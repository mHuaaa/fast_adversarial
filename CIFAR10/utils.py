import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from collections import defaultdict

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n

def evaluate_pgdV2(test_loader, model, attack_iters, alpha=2, restarts=1):
    epsilon = (8 / 255.) / std
    alpha = (alpha / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n

def evaluate_pgdV3(test_loader, model, restarts=1):
    # epsilon = (8 / 255.) / std
    # alpha = (alpha / 255.) / std
    # pgd_loss = 0
    # pgd_acc = 0
    # n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        # atk #1
        attack_iters = 1
        epsilon = (8 / 255.) / std
        alpha = (16 / 255.) / std
        pgd_loss = 0
        pgd_acc = 0
        n = 0
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        loss1, acc1, output1 = pgd_loss/n, pgd_acc/n, output
        print("Finished atk#1")
        # atk #2
        attack_iters = 8
        epsilon = (8 / 255.) / std
        alpha = (2 / 255.) / std
        pgd_loss = 0
        pgd_acc = 0
        n = 0
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        loss2, acc2, output2 = pgd_loss/n, pgd_acc/n, output
        print("Finished atk#2")
        break

    return loss1, acc1, output1, loss2, acc2, output2, y


def evaluate_pgd4train(train_loader, model, attack_parses, sample_limit=500):
    def refine_eval_value(pgd_loss, pgd_acc, n):
        for key in pgd_loss.keys():
            pgd_loss[key] = list(map(lambda x: x/n, pgd_loss[key]))
            pgd_acc[key] = list(map(lambda x: x/n, pgd_acc[key]))
        return pgd_loss, pgd_acc

    epsilon = (8 / 255.) / std

    pgd_loss = {}
    pgd_acc = {}

    n = 0
    model.eval()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        for attack_iters in attack_parses.keys():
            if attack_iters not in pgd_loss:
                pgd_loss[attack_iters] = [0]*len(attack_parses[attack_iters])
                pgd_acc[attack_iters] = [0]*len(attack_parses[attack_iters])

            for j in range(len(attack_parses[attack_iters])):
                alpha = attack_parses[attack_iters][j]
                alpha = (alpha / 255.) / std
                pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, 1)
                with torch.no_grad():
                    output = model(X + pgd_delta)
                    loss = F.cross_entropy(output, y)
                    pgd_loss[attack_iters][j] += loss.item() * y.size(0)
                    pgd_acc[attack_iters][j] += (output.max(1)[1] == y).sum().item()

        n += y.size(0)

        if n >= sample_limit:
            break

    # print("test evaluate_pgd4train before", pgd_loss, pgd_acc, n)
    pgd_loss, pgd_acc = refine_eval_value(pgd_loss, pgd_acc, n)
    # print("test evaluate_pgd4train after", pgd_loss, pgd_acc, n)
    return pgd_loss, pgd_acc

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def validAttackParses(iters, alpha, args):
    return args.max_iters >= iters > 0 and args.epsilon*2 >= alpha > 0

def addNextAttackParses(next_attack_parses, iters, alpha, args):
    if validAttackParses(iters, 2, args) and iters not in next_attack_parses:
        next_attack_parses[iters] = []

    for cur in [-1, 0, 1]:
        cur_alpha = alpha + cur*args.alpha_interval
        if validAttackParses(iters, cur_alpha, args):
            next_attack_parses[iters].append(cur_alpha)

def chooseAttackParses(attack_parses, loss, acc, args, others=None):
    next_candidate = []
    next_candidate_rob = []
    next_candidate_err = []
    next_candidate_loss = []
    next_candidate_eval = []

    if args.mode == 1:
        print("history acc: ", others["history_acc"]/others['history_n'])
        print("pre acc: ", others["pre_acc"])
        # print("pre loss: ", others["pre_loss"])

    for atk_iter in sorted(list(attack_parses.keys())):
        alphas = attack_parses[atk_iter]
        atk_loss = loss[atk_iter]
        atk_err = [1-a for a in acc[atk_iter]]
        if args.mode == 0:
            atk_eval = [(1-a)/atk_iter for a in acc[atk_iter]]
        elif args.mode == 1:
            atk_eval = np.add(atk_loss, args.penalty_iters_coeff/atk_iter)
            # atk_eval = np.subtract(others['history_acc']/others['history_n'], acc[atk_iter])/atk_iter
            # atk_eval = np.exp(atk_loss)/atk_iter
            # atk_eval = np.exp(1*np.subtract(others["pre_acc"], acc[atk_iter]))/atk_iter
            # atk_eval = [np.exp(others["pre_acc"]-a)/atk_iter for a in acc[atk_iter]]
            # atk_eval = [(1-a)/atk_iter for a in acc[atk_iter]]
        elif args.mode == 2:
            penalty_loss = np.add(args.penalty_iters_coeff/atk_iter, np.divide(args.penalty_alpha_coeff, alphas))
            atk_eval = np.add(atk_loss, penalty_loss)

        else:
            print("Error chooseAttackParses mode!!")
            exit(0)
        print("tmp iters loss: ", atk_loss)
        print("tmp iters eval: ", atk_eval)
        # print("cost compute:", atk_iters, acc[atk_iters], atk_eval)
        best_alpha_index = np.argmax(atk_eval)
        next_candidate.append((atk_iter, alphas[best_alpha_index]))
        next_candidate_loss.append(atk_loss[best_alpha_index])
        next_candidate_rob.append(acc[atk_iter][best_alpha_index])
        next_candidate_err.append(atk_err[best_alpha_index])
        next_candidate_eval.append(atk_eval[best_alpha_index])

    assert next_candidate == sorted(next_candidate)
    print("candidate parses: ", next_candidate)
    print("candidate parses loss: ", next_candidate_loss)
    print("candidate parses robust rate: ", next_candidate_rob)
    print("candidate parses error rate: ", next_candidate_err)
    print("candidate parses eval: ", next_candidate_eval)

    best_parse_index = np.argmax(next_candidate_eval)
    choose_atk_iters, choose_alpha = next_candidate[best_parse_index]

    next_attack_parses = {}
    addNextAttackParses(next_attack_parses, choose_atk_iters, choose_alpha, args)
    if best_parse_index == 0:
        addNextAttackParses(next_attack_parses, choose_atk_iters-1, choose_alpha, args)
    else:
        addNextAttackParses(next_attack_parses, next_candidate[best_parse_index-1][0], next_candidate[best_parse_index-1][1], args)

    if best_parse_index == len(next_candidate)-1:
        addNextAttackParses(next_attack_parses, choose_atk_iters+1, choose_alpha, args)
    else:
        addNextAttackParses(next_attack_parses, next_candidate[best_parse_index+1][0], next_candidate[best_parse_index+1][1], args)

    print("choose attack parses", choose_atk_iters, choose_alpha, next_attack_parses)
    print("generate next attack parses", choose_atk_iters, choose_alpha, next_attack_parses)
    return choose_atk_iters, choose_alpha, next_attack_parses

def chooseAttackParsesV2(attack_parses, loss, acc, alpha_interval=1, mode=0):
    next_candidate = []
    next_candidate_err = []
    next_candidate_loss = []
    next_candidate_eval = []

    if mode == 0:
        for atk_iters in sorted(list(attack_parses.keys())):
            alphas = attack_parses[atk_iters]
            atk_loss = loss[atk_iters]
            atk_err = [1-a for a in acc[atk_iters]]
            atk_eval = [(1-a)/atk_iters for a in acc[atk_iters]]
            # print("cost compute:", atk_iters, acc[atk_iters], atk_eval)

            best_alpha_index = np.argmax(atk_eval)
            next_candidate.append((atk_iters, alphas[best_alpha_index]))
            next_candidate_loss.append(atk_loss[best_alpha_index])
            next_candidate_err.append(atk_err[best_alpha_index])
            next_candidate_eval.append(atk_eval[best_alpha_index])
        print("candidate parses: ", next_candidate)
        print("candidate parses loss: ", next_candidate_loss)
        print("candidate parses error rate: ", next_candidate_err)
        print("candidate parses eval: ", next_candidate_eval)
    else:
        print("Error chooseAttackParses mode!!")
        exit(0)
    assert next_candidate == sorted(next_candidate)
    best_parse_index = np.argmax(next_candidate_eval)
    choose_atk_iters, choose_alpha = next_candidate[best_parse_index]

    next_attack_parses = {}
    addNextAttackParses(next_attack_parses, choose_atk_iters, choose_alpha, alpha_interval)
    if best_parse_index == 0:
        addNextAttackParses(next_attack_parses, choose_atk_iters-1, choose_alpha, alpha_interval)
    else:
        addNextAttackParses(next_attack_parses, next_candidate[best_parse_index-1][0], next_candidate[best_parse_index-1][1], alpha_interval)

    if best_parse_index == len(next_candidate)-1:
        addNextAttackParses(next_attack_parses, choose_atk_iters+1, choose_alpha, alpha_interval)
    else:
        addNextAttackParses(next_attack_parses, next_candidate[best_parse_index+1][0], next_candidate[best_parse_index+1][1], alpha_interval)

    print("choose attack parses", choose_atk_iters, choose_alpha, next_attack_parses)
    print("generate next attack parses", choose_atk_iters, choose_alpha, next_attack_parses)

    sum_eval = sum(next_candidate_eval)
    next_candidate_probs = [e/sum_eval for e in next_candidate_eval]
    # print("next candidate", next_candidate)
    print("next candidate probs", next_candidate_probs)
    return next_candidate, next_candidate_probs, next_attack_parses

