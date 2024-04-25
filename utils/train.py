import torch
from utils.eval import run_evaluation, get_accuracy
import os
import time
from mixup import UMixup
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def mixup_criterion(logits, l_y, r_y, lam):
    loss = lam * F.cross_entropy(logits, l_y) + (1 - lam) * F.cross_entropy(logits, r_y)
    return loss.mean()


def train(model, optimizer, criterion, data, mu=1.0, mixer=None):
    model.train()
    optimizer.zero_grad()
    # classification loss
    loss = 0
    # mixup loss
    mixup_loss = 0
    mixed_data = None

    out = model(data)
    sup_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    ori_embed = model.get_embedding(-2).detach()
    ori_pred = model.get_logits().softmax(-1)
   
    if mixer is not None and mixer.shrink_ratio > 0:
        mixed_data_list = mixer.mixup(data, ori_embed, pred=ori_pred.detach())
        for mixed_data in mixed_data_list:
            mixed_out = model(mixed_data)
            mix_logits = model.get_logits()
            mixup_loss += mixup_criterion(logits=mix_logits[mixed_data.mixed_index],
                                        l_y=mixed_data.l_y,
                                        r_y=mixed_data.r_y,
                                        lam=mixed_data.lam)

        mixup_loss /= len(mixed_data_list)
    
    loss = sup_loss + mu * mixup_loss

    mixup_training_end_time = time.time()

    loss.backward()
    optimizer.step()
    
    acc = get_accuracy(out[data.train_mask], data.y[data.train_mask])
    return sup_loss, mixup_loss, acc


def run_training(model, optimizer, criterion, g, args, logger, device=None):
    es = 0
    best_acc = 0
    best_val_acc = 0

    if args.mix_method == "none":
        mixer = None
    else:
        mixer = UMixup(alpha=args.alpha, 
                       shrink_ratio=args.shrink_ratio,
                       hop=args.hop, 
                       batch_size=args.batch_size,
                        mix_method=args.mix_method, 
                       lam_type=args.lam_type, 
                       pair_type=args.pair_type,
                       beta=args.beta, 
                       gamma=args.gamma, 
                       lam=args.lam, 
                       name=args.dataset, 
                       device=device)

    num_epochs = args.epochs 
    for epoch in tqdm(range(num_epochs)):
        sup_loss, mixup_loss, train_acc = train(model, optimizer, criterion, g, args.mu, mixer)
        val_loss, val_acc, test_loss, test_acc = run_evaluation(model, criterion, g)
    
        # select the result based on valdation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc = test_acc
            es = 0
        
        else:
            es += 1

        # if es == 50:
        #     print("Early stopping!")
        #     break

    return best_acc * 100