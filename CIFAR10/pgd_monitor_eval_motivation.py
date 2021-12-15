import argparse
import logging
import os
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wrn_madry import Wide_ResNet_Madry
from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgdV2, evaluate_standard)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--data-type', default='test', type=str)
    parser.add_argument('--net', default='pre_act_resnet18', type=str, choices=['pre_act_resnet18', 'WRN-32-10'])
    # parser.add_argument('--epochs', default=15, type=int)
    # parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    # parser.add_argument('--lr-min', default=0., type=float)
    # parser.add_argument('--lr-max', default=0.2, type=float)
    # parser.add_argument('--weight-decay', default=5e-4, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--min-attack-iters', default=1, type=int, help='MIN Attack iterations')
    parser.add_argument('--max-attack-iters', default=15, type=int, help='MAX Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    # parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--start-model', type=int, help='Start model number')
    parser.add_argument('--end-model', type=int, help='End model number')
    parser.add_argument('--skip-step', type=int, help='Skip step size')
    parser.add_argument('--model-dir', type=str, help='Eval model directory')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--out-file-name', default='eval_output.log', type=str, help='Log file name')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
    #     help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    # parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
    #     help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    # parser.add_argument('--master-weights', action='store_true',
    #     help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, args.out_file_name)

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

    data_loader = train_loader if args.data_type == "train" else test_loader

    # Evaluation
    if args.net == "pre_act_resnet18":
        model_test = PreActResNet18().cuda()
    elif args.net == "WRN-32-10":
        model_test = Wide_ResNet_Madry(depth=32, num_classes=10, widen_factor=10, dropRate=0.0).cuda()
    else:
        print("Error: model error!!!")
        exit(0)

    # model_test = PreActResNet18().cuda()
    start_time = time.time()
    logger.info('Model \t Atk Iter \t Alpha \t PGD Loss \t PGD Acc')
    start, end, step = args.start_model, args.end_model, args.skip_step
    for i in range(start, end+1, step):
        model_path = "model_"+str(i)+".pth"
        model_full_path = os.path.join(args.model_dir, model_path)
        if not os.path.exists(model_full_path):
            print("Warning: {} not exists.".format(model_full_path))
            continue
        model_test.load_state_dict(torch.load(model_full_path))
        model_test.float()
        model_test.eval()
        # print("test 1")
        for attack_iters in range(args.min_attack_iters, args.max_attack_iters+1):
            # print("test 2")
            alpha = max(np.ceil(16/attack_iters), 2)
            pgd_loss, pgd_acc = evaluate_pgdV2(data_loader, model_test, attack_iters, alpha, args.restarts)
            # test_loss, test_acc = evaluate_standard(data_loader, model_test)

            logger.info('%d \t %d \t %d \t %.4f \t %.4f', i, attack_iters, alpha, pgd_loss, pgd_acc)

    eval_time = time.time()
    logger.info('Total eval time: %.4f minutes', (eval_time - start_time)/60)

if __name__ == "__main__":
    main()
