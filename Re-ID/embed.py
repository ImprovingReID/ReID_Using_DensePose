import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import pickle
import numpy as np
import sys
import logging
import argparse
import cv2

from backbone import ResNet18_bb, ResNet50_bb
from head import DenseHead, MainHead
from dataWrapper import Wrapper

from utils import pdist_np as pdist


torch.multiprocessing.set_sharing_strategy('file_system')

def embed(load_path, store_path):

    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## restore model
    logger.info('restoring model')
    mainNet = ResNet50_bb().cuda()
    mainNet.load_state_dict(torch.load('../../res/mainNet2.pkl'))
    mainNet = nn.DataParallel(mainNet)
    mainHead = MainHead().cuda()
    mainHead.load_state_dict(torch.load('../../res/mainHead2.pkl'))
    mainHead = nn.DataParallel(mainHead)
    mainNet.eval()
    mainHead.eval()

    ## load gallery dataset
    batchsize = 32
    ds = Wrapper(load_path, is_train = False)
    dl = DataLoader(ds, batch_size = batchsize, drop_last = False, num_workers = 4)


    ## embedding samples
    logger.info('start embedding')
    all_iter_nums = len(ds) // batchsize + 1
    embeddingsGlobal = []
    embeddingsLocal = []
    label_ids = []
    for it, (img, lb_id) in enumerate(dl):
        print('\r=======>  processing iter {} / {}'.format(it, all_iter_nums),
                end = '', flush = True)
        label_ids.append(lb_id)
        embdsLocal = []
        embdsGlobal = []
        for im in img:
            im = im.cuda()
            embd = mainNet(im).cpu()#.numpy()
            G, L = mainHead(embd)#.detach().cpu().numpy()
            G = G.detach().cpu().numpy()
            L = L.detach().cpu().numpy()
            embdsGlobal.append(G)
            embdsLocal.append(L)
        embedGlobal = sum(embdsGlobal) / len(embdsGlobal)
        embedLocal = sum(embdsLocal) / len(embdsLocal)
        embeddingsGlobal.append(embedGlobal)
        embeddingsLocal.append(embedLocal)

    print('  ...   completed')

    embeddingsGlobal = np.vstack(embeddingsGlobal)
    embeddingsLocal = np.vstack(embeddingsLocal)
    label_ids = np.hstack(label_ids)

    ## dump results
    logger.info('dump embeddings')
    embd_res = {'embeddingsGlobal': embeddingsGlobal, 'embeddingLocal': embeddingsLocal, 'label_ids': label_ids}

    with open(store_path, 'wb') as fw:
        pickle.dump(embd_res, fw)

    logger.info('embedding finished')


def evaluate(load_path1, load_path2):
    cmc_rank = 1

    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    ## load embeddings
    logger.info('loading gallery embeddings')
    with open(load_path1, 'rb') as fr:
        gallery_dict = pickle.load(fr)
        embGlobal, embLocal, lb_ids = gallery_dict['embeddingsGlobal'], gallery_dict['embeddingLocal'], gallery_dict['label_ids']
    logger.info('loading query embeddings')
    with open(load_path2 , 'rb') as fr:
        query_dict = pickle.load(fr)
        embGlobal_query, embLocal_query, lb_ids_query = query_dict['embeddingsGlobal'], query_dict['embeddingLocal'], query_dict['label_ids']

    ## compute and clean distance matrix
    embGallery = np.concatenate((embGlobal,embLocal),1)
    embQuery = np.concatenate((embGlobal_query, embLocal_query),1)
    print(embGallery.shape)
    print(embQuery.shape)
    dist_mtx = pdist(embQuery, embGallery)
    print(dist_mtx.shape)
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis = 1)
    print(indices.shape)
    print(lb_ids.shape)
    print(lb_ids_query.shape)
    matches = lb_ids[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    print(matches)
    all_aps = []
    all_cmcs = []
    logger.info('starting evaluating ...')
    for qidx in range(n_q): #tqdm(range(n_q)):
        qpid = lb_ids_query[qidx]

        order = indices[qidx]
        pid_diff = lb_ids[order] != qpid
        useful = lb_ids[order] != -1
        #keep = np.logical_or(pid_diff, cam_diff)
        keep = np.logical_and(pid_diff, useful)
        print(keep)
        match = matches[qidx][keep]
        print(match)

        if not np.any(match): continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:cmc_rank])

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    print('mAP is: {}, cmc is: {}'.format(mAP, cmc))


if __name__ == '__main__':
    #load_path = '/mnt/analyticsvideo/DensePoseData/market1501/'
    #store_path = '../../res/embd_res'
    #load_path = '/mnt/analyticsvideo/DensePoseData/market1501/query'
    #store_path = '../../res/embd_query'
    #embed(load_path,store_path)
    evaluate('../../res/embd_res', '../../res/embd_query')