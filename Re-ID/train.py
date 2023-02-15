import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools

from backbone import ResNet50_bb
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger



def train():
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    net = ResNet50_bb().cuda()
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(net.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000)

    ## dataloader
    selector = BatchHardTripletSelector()
    ds = Market1501('/mnt/analyticsvideo/DensePoseData/market1501/Market-1501-v15.09.15/bounding_box_train', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        embds = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss_avg, optim.lr, time_interval))
            loss_avg = []
            t_start = t_end

        count += 1
        if count == 1000: break

    ## dump model
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')


if __name__ == '__main__':
    train()