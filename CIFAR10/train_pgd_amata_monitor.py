import argparse
import logging
import os
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from wrn_madry import Wide_ResNet_Madry
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--net', default='pre_act_resnet18', type=str, choices=['pre_act_resnet18', 'WRN-32-10'])
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--amata-kmin', default=2, type=float)
    parser.add_argument('--amata-kmax', default=10, type=float)
    parser.add_argument('--amata-tao', default=20, type=float)
    parser.add_argument('--amata-eta', default=0.05, type=float)
    parser.add_argument('--amata-alpha-min', default=2, type=float)
    parser.add_argument('--amata-mode', default=0, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

def amataCompute(Kmin, Kmax, T, tao, eta=None, Alphamin=None, mode=0):
    def function(x, eta, T):
        return 8/(1+np.exp((x-(T/2))*eta))+2

    Klist = []
    Alphalist = []
    for t in range(1, T+1):
        if mode == 0:
            coeff = t/T
        elif mode == 1:
            coeff = (1-1/np.exp(eta*t))/(1-1/np.exp(eta*T))
        elif mode == 2 or mode == 3:
            coeff = (np.exp(eta*t))/(np.exp(eta*T))
        Kt = Kmin + (Kmax-Kmin)*coeff

        if mode == 3:
            Alphat = function(t, 0.08, T)
        else:
            Alphat = tao/Kt if Alphamin is None else max(2, tao/Kt)

        Klist.append(Kt)
        Alphalist.append(Alphat)
    Klist = list(map(round, Klist))
    Alphalist = list(map(round, Alphalist))

    return Klist, Alphalist

def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    if args.net == "pre_adc_resnet18":
        model = PreActResNet18().cuda()
    elif args.net == "WRN-32-10":
        model = Wide_ResNet_Madry(depth=32, num_classes=10, widen_factor=10, dropRate=0.0).cuda()
    else:
        exit(0)

    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # compute amata scheduler
    Klist, Alphalist = amataCompute(args.amata_kmin, args.amata_kmax, args.epochs, args.amata_tao, eta=args.amata_eta, Alphamin=args.amata_alpha_min, mode=args.amata_mode)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Kt \t Alpha \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        # get amata attack pares:
        Kt, Alphat = Klist[epoch], Alphalist[epoch]
        alpha = (Alphat / 255.) / std

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for i in range(len(epsilon)):
                    delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            for _ in range(Kt):
                output = model(X + delta)
                loss = criterion(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.grad.zero_()

            delta = delta.detach()
            output = model(X + delta)
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %d \t %d \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, Kt, Alphat, train_loss/train_n, train_acc/train_n)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_'+str(epoch)+ '.pth'))

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 3)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

if __name__ == "__main__":
    main()
