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
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    # parser.add_argument('--epochs', default=15, type=int)
    # parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    # parser.add_argument('--lr-min', default=0., type=float)
    # parser.add_argument('--lr-max', default=0.2, type=float)
    # parser.add_argument('--weight-decay', default=5e-4, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=50, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    # parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--start-model', type=int, help='Start model number')
    parser.add_argument('--end-model', type=int, help='End model number')
    parser.add_argument('--skip-step', type=int, help='Skip step size')
    parser.add_argument('--model-dir', type=str, help='Eval model directory')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
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
    logfile = os.path.join(args.out_dir, 'eval_output.log')
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

    # Evaluation
    model_test = PreActResNet18().cuda()
    start_time = time.time()
    logger.info('Model \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    start, end, step = args.start_model, args.end_model, args.skip_step
    for i in range(start, end, step):
        model_path = "model_"+str(i)+".pth"
        model_test.load_state_dict(torch.load(os.path.join(args.model_dir, model_path)))
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.attack_iters, args.restarts)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('%d \t %.4f \t \t %.4f \t %.4f \t %.4f', i, test_loss, test_acc, pgd_loss, pgd_acc)
    eval_time = time.time()
    logger.info('Total eval time: %.4f minutes', (eval_time - start_time)/60)

if __name__ == "__main__":
    main()
