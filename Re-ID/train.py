from pathlib import Path
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
from head import MainHead, DenseHead, Classifier
from loss import TripletLoss, ArcFace
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from dataWrapper import Wrapper
from optimizer import AdamOptimWrapper
from logger import logger

from embed import embed, evaluate

from pytorch_metric_learning import losses
from torchvision.models import resnet50, ResNet50_Weights


def train(name = None, num_it = 30000):
    # setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    # model and loss
    path_log = Path("log/" + name + "train.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    try:
        if path_log.exists():
            raise FileExistsError
    except FileExistsError:
        print("Already exist log file: {}".format(path_log))
        raise
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=path_log.__str__(),
            filemode="w",
        )
        print("Create log file: {}".format(path_log))

    logger.info('setting up backbone model and loss')

    rn50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    mainNet = ResNet50_bb().cuda()
    #mainNet.load_state_dict(torch.load('res/arcface30k_mainNet.pkl'))
    for mainlayer,reslayer,numlayersinblock in [[mainNet.layer1,rn50.layer1, 3], [mainNet.layer2,rn50.layer2,4],[mainNet.layer3,rn50.layer3,6]]:
         for conv in ["conv1","conv2","conv3"]:
              for i in range(numlayersinblock):
                  mainlayer[i]._modules[conv].weight.data.copy_(reslayer[i]._modules[conv].weight.data)

    mainNet = nn.DataParallel(mainNet)
    DSAGNet = ResNet18_bb().cuda()
    #DSAGNet.load_state_dict(torch.load('res/arcface30k_DSAGNet.pkl'))
    DSAGNet = nn.DataParallel(DSAGNet)

    mainHead = MainHead().cuda()
    #mainHead.load_state_dict(torch.load('res/arcface30k_mainHead.pkl'))
    mainHead = nn.DataParallel(mainHead)
    DSAGHead = DenseHead().cuda()
    #DSAGHead.load_state_dict(torch.load('res/arcface30k_DSAGHead.pkl'))
    DSAGHead = nn.DataParallel(DSAGHead)

    classifier = Classifier().cuda()
    classifier = nn.DataParallel(classifier)

    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin
    ID_loss = nn.CrossEntropyLoss().cuda()

    num_classes=1501
    embedding_size=2048
    arcface_loss = losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64).cuda() # 28.6 and 64

    model_name = ["mainNet", "DSAGNet", "mainHead", "denseHead", "classifier"]
    optimizer = {
    "mainNet": AdamOptimWrapper(mainNet.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000),
    "DSAGNet": AdamOptimWrapper(DSAGNet.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000),
    "mainHead": AdamOptimWrapper(mainHead.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000),
    "denseHead": AdamOptimWrapper(DSAGHead.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000),
    "classifier": AdamOptimWrapper(classifier.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000)
    # "mainNet": torch.optim.SGD(mainNet.parameters(), lr = 0.025, momentum = 0.9, weight_decay=5e-4),
    # "DSAGNet": torch.optim.SGD(DSAGNet.parameters(), lr = 0.025, momentum = 0.9, weight_decay=5e-4),
    # "mainHead": torch.optim.SGD(mainHead.parameters(), lr = 0.025, momentum = 0.9, weight_decay=5e-4),
    # "denseHead": torch.optim.SGD(denseHead.parameters(), lr = 0.025, momentum = 0.9, weight_decay=5e-4),
    # "classifier": torch.optim.SGD(classifier.parameters(), lr = 0.025, momentum = 0.9, weight_decay=5e-4)
    }


    # optimizer
    logger.info('creating optimizer')
    #optim = AdamOptimWrapper(mainNet.parameters(), lr = 3e-3, wd = 0, t0 = 15000, t1 = 25000)

    # /mnt/analyticsvideo/DensePoseData/market1501/SegmentedMarket1501train
    selector = BatchHardTripletSelector()
    ds = Wrapper('/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_train',data_path_dense='/mnt/analyticsvideo/DensePoseData/market1501/uv_maps_train', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    # For test error
    ds_test = Wrapper('/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_test',data_path_dense='/mnt/analyticsvideo/DensePoseData/market1501/uv_maps_test', is_train = True)
    sampler_test = BatchSampler(ds_test, 18, 4)
    dl_test = DataLoader(ds_test, batch_sampler = sampler_test, num_workers = 4)
    diter_test = iter(dl_test)

    # train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    epochs = 0
    losses_train = []
    losses_test = []
    epochs_counter = []
    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_test'
    store_path = 'res/embd_res'
    load_path2 = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    store_path2 = 'res/embd_query'
    t_start = time.time()
    while True:
        try:
            imgs, imgs_dense, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, imgs_dense, lbs, _ = next(diter)
        mainNet.train()
        DSAGNet.train()
        mainHead.train()
        DSAGHead.train()
        classifier.train()

        imgs = imgs.cuda()
        lbs = lbs.cuda()
        imgs_dense = imgs_dense.cuda()

        mainEmbds = mainNet(imgs)
        DSAGEmbds = DSAGNet(imgs_dense)

        mainGlobalEmbds, mainLocalEmbds = mainHead(mainEmbds)
        denseGlobalEmbds, denseLocalEmbds = DSAGHead(DSAGEmbds)
        globalEmbds = mainGlobalEmbds + denseGlobalEmbds
        localEmbds = mainLocalEmbds + denseLocalEmbds
    
        # anchor, positives, negatives = selector(globalEmbds, lbs)
        # trip_global_loss = triplet_loss(anchor, positives, negatives)
        # anchor, positives, negatives = selector(localEmbds, lbs)
        # trip_local_loss = triplet_loss(anchor, positives, negatives)

        #lbs = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17]).cuda()

        # global_main_ID_loss = ID_loss(classifier(mainGlobalEmbds),lbs)
        # local_main_ID_loss = ID_loss(classifier(mainLocalEmbds),lbs)
        # global_ID_loss = ID_loss(classifier(globalEmbds),lbs)
        # local_ID_loss = ID_loss(classifier(localEmbds),lbs)

        global_main_arcface_loss = arcface_loss(mainGlobalEmbds,lbs)
        local_main_arcface_loss = arcface_loss(mainLocalEmbds,lbs)
        global_arcface_loss = arcface_loss(globalEmbds, lbs)
        local_arcface_loss = arcface_loss(localEmbds,lbs)

        # weights
        c1, c2, c3 = 1.5, 0.5, 1
        
        # Update model
        model = [mainNet, DSAGNet, mainHead, DSAGHead, classifier]
        for m in model:
            m.zero_grad()

        for m in model_name:
            optimizer[m].zero_grad()

        #loss = c1*(trip_global_loss+trip_local_loss)
        #loss = c1*(trip_global_loss+trip_local_loss)+c2*(global_main_arcface_loss+local_main_arcface_loss)+c3*(global_arcface_loss+local_arcface_loss)
        #optim.zero_grad()
        loss = c2*(global_main_arcface_loss+local_main_arcface_loss) + c3*(global_arcface_loss+local_arcface_loss)
        #loss = c2*(global_main_ID_loss+local_main_ID_loss)+c3*(global_ID_loss+local_ID_loss)
        #loss = c1*(trip_global_loss+trip_local_loss)+c2*(global_main_ID_loss+local_main_ID_loss)+c3*(global_ID_loss+local_ID_loss)
        #print(loss)
        loss.backward()

        for m in model_name:
            optimizer[m].step()

        #optim.step()))+

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, time: {:3f}'.format(count, loss_avg, time_interval))
            loss_avg = []
            t_start = t_end

            if count == 20:
                for m in model_name:
                    optimizer[m].set_lr(1e-2)

            if count == 500:
                for m in model_name:
                    optimizer[m].set_lr(1e-3)

        count += 1
        if count % 5 == 0:
            epochs += 1
            if True: #epochs % 10 == 0:
                # embd_global, embd_local, lbs = embed(load_path,store_path, Net = mainNet, Head = mainHead)
                # embd_query, lbs = embed(load_path2, store_path2, Net = mainNet, Head = mainHead)
                # print(type(embd_global))
                # print(type(embd_local))
                # print(type(lbs))
                # print(embd_global.shape)
                # print(embd_local.shape)
                # print(lbs.shape)
                try:
                    imgs, imgs_dense, lbs, _ = next(diter_test)
                except StopIteration:
                    diter_test = iter(dl_test)
                    imgs, imgs_dense, lbs, _ = next(diter_test)

                imgs = imgs.cuda()
                lbs = lbs.cuda()
                imgs_dense = imgs_dense.cuda()
                mainEmbds = mainNet(imgs)
                DSAGEmbds = DSAGNet(imgs_dense)

                mainGlobalEmbds, mainLocalEmbds = mainHead(mainEmbds)
                denseGlobalEmbds, denseLocalEmbds = DSAGHead(DSAGEmbds)
                embd_global = mainGlobalEmbds + denseGlobalEmbds
                embd_local = mainLocalEmbds + denseLocalEmbds

                loss_global_main = arcface_loss(mainGlobalEmbds, lbs)
                loss_local_main = arcface_loss(mainLocalEmbds, lbs)
                loss_global = arcface_loss(embd_global,lbs)
                loss_local = arcface_loss(embd_local,lbs)

                loss_test = c2*(loss_global_main+loss_local_main) + c3*(loss_global+loss_local)
                losses_test.append(loss_test/72)
                losses_train.append(loss/72)
                epochs_counter.append(epochs)

        if count == num_it: 
            break

    # dump model
    logger.info('saving trained model')
    torch.save(mainNet.module.state_dict(), 'res/' + name + '_mainNet.pkl')
    torch.save(DSAGNet.module.state_dict(), 'res/' + name + '_DSAGNet.pkl')
    torch.save(mainHead.module.state_dict(), 'res/' + name + '_mainHead.pkl')
    torch.save(DSAGHead.module.state_dict(), 'res/' + name + '_DSAGHead.pkl')
    torch.save(classifier.module.state_dict(), 'res/' + name + '_classifier.pkl')


    logger.info('everything finished')

    return losses_test, losses_train, epochs


if __name__ == '__main__':
    train(name = 'arcface80k', num_it=50000)

    load_path = '/mnt/analyticsvideo/DensePoseData/market1501/bounding_box_test'
    store_path = 'res/embd_res'
    load_path2 = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    store_path2 = 'res/embd_query'
    embed(load_path,store_path)
    embed(load_path2, store_path2)
    evaluate('res/embd_res', 'res/embd_query')