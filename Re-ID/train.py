import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools

from backbone import ResNet50_bb, ResNet18_bb
from head import Head, MainHead, DenseHead
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from Market1501 import Market1501, DensePose1501
from optimizer import AdamOptimWrapper
from logger import logger



def train():
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    mainNet = ResNet50_bb().cuda()
    mainNet = nn.DataParallel(mainNet)
    DSAGNet = ResNet18_bb().cuda()
    DSAGNet = nn.DataParallel(DSAGNet)

    mainHead = MainHead().cuda()
    mainHead = nn.DataParallel(mainHead)
    denseHead = DenseHead().cuda()
    denseHead = nn.DataParallel(denseHead)

    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(mainNet.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000)

    ## /mnt/analyticsvideo/DensePoseData/market1501/SegmentedMarket1501train
    selector = BatchHardTripletSelector()
    ds = Market1501('/mnt/analyticsvideo/DensePoseData/market1501/Market-1501-v15.09.15/bounding_box_train', is_train = True)
    ds_dense = DensePose1501('/mnt/analyticsvideo/DensePoseData/market1501/SegmentedMarket1501train/uv_maps', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    sampler_dense = BatchSampler(ds_dense,18,4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    dl_dense = DataLoader(ds_dense, batch_sampler = sampler_dense, num_workers = 4)
    diter = iter(dl)
    diter_dense = iter(dl_dense)

    # train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = next(diter)
            imgs_dense, lbs_dense, _ = next(diter_dense)
        except StopIteration:
            diter = iter(dl)
            diter_dense = iter(dl_dense)
            imgs, lbs, _ = next(diter)
            imgs_dense, lbs_dense, _ = next(diter_dense)
        mainNet.train()
        DSAGNet.train()
        
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        imgs_dense = imgs_dense.cuda()
        lbs_dense = lbs_dense.cuda()

        mainEmbds = mainNet(imgs)
        DSAGEmbds = DSAGNet(imgs_dense)

        mainGlobalEmbds, mainLocalEmbds = mainHead(mainEmbds)
        denseGlobalEmbds, denseLocalEmbds = denseHead(DSAGEmbds)
        #print(mainGlobalEmbds.shape)
        #print(mainLocalEmbds.shape)
        anchor, positives, negatives = selector(mainGlobalEmbds, lbs)

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

    # dump model
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')


if __name__ == '__main__':
    train()