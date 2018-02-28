# -*- coding: utf-8 -*-
"""
Python script for training FW-Net
@author: Ke Sun
@e-mail: sunk@mail.ustc.edu.cn
@Date:   Feb 26, 2018
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import caffe
from multiprocessing import Process, Queue, freeze_support
import logging
from random import *
from scipy.io import loadmat
import argparse


def cal_PSNR(pred,gt):
    diff = np.array(gt,dtype=np.float32) - np.array(pred,dtype=np.float32)
    mse = np.multiply(diff,diff).sum() / diff.size
    psnr = 10*np.log10(255.*255./mse)
    return psnr

def test_valid(net):
    valid_img_path = '/data8/sunk/FW/deno_data/bsd68_noise_%d/' % sigma
    valid_gt_path = '/data8/sunk/FW/deno_data/bsd68_gt/'
    aver_psnr, loss = 0.0,0.0
    psnr_L = []
    num = 19
    for i in xrange(1,num+1):
        im_Name = 'im%04d' % i
        valid_img = loadmat(valid_img_path+im_Name+'.mat')['z']
        valid_gt = Image.open(valid_gt_path+('test%03d'%i)+'.png')
        valid_gt = np.asarray(valid_gt,dtype=np.float32)
        img_H = valid_img.shape[0]
        img_W = valid_img.shape[1]
        net.blobs['data'].reshape(1,1,img_H,img_W)
        net.blobs['data'].data[...] = valid_img
        net.blobs['label'].reshape(1,1,img_H,img_W)
        net.blobs['label'].data[...] = valid_img
        net.forward()
        pred = net.blobs['map_fc'].data
        pred = pred * 255.
        pred[pred>255] = 255
        pred[pred<0] = 0
        pred = np.array(pred,dtype=np.int)
        psnr = cal_PSNR(pred,valid_gt)
        psnr_L.append(psnr)
        aver_psnr = aver_psnr + psnr / num
    psnr_set5 = np.sum(np.array(psnr_L)[0:5]) / 5.
    psnr_set14 = np.sum(np.array(psnr_L)[5:19]) / 14
    return psnr_L, aver_psnr, psnr_set5, psnr_set14

# randomly crop fixed-size patch for training
# gray-image
def rand_crop(img,gt,crop_H,crop_W,x=-1,y=-1):
    ori_Img = np.array(img,dtype=np.float32).copy()
    ori_gt = np.array(gt,dtype=np.float32).copy()
    img_H = ori_Img.shape[0]
    img_W = ori_Img.shape[1]
    if x==-1 or y==-1:
        x = randint(0,img_W-crop_W-1)
        y = randint(0,img_H-crop_H-1)
    crop_Img = ori_Img[y:y+crop_H,x:x+crop_W]
    crop_gt = ori_gt[y:y+crop_H,x:x+crop_W]
    return crop_Img,crop_gt

# generate training data
def genData(Data_Q,Label_Q,batchsize,crop_H,crop_W):
    name_list = range(1,433)
    shuffle(name_list)
    data_L = np.zeros((batchsize,1,crop_H,crop_W))
    target_L = np.zeros((batchsize,1,crop_H,crop_W))
    count = 0
    key = 0
    while True:
        if Data_Q.full() != True:
            img = loadmat('/data8/sunk/FW/deno_data/BSD432_sgm_25/im%04d.mat' % name_list[key])['z']
            gt = loadmat('/data8/sunk/FW/deno_data/BSD432_gt/im%04d.mat' % name_list[key])['y']
            crop_img,crop_gt = rand_crop(img,gt,crop_H,crop_W)
            data_L[count,0,:,:] = crop_img
            target_L[count,0,:,:] = crop_gt
            count += 1
            if count == batchsize:
                Data_Q.put(data_L)
                Label_Q.put(target_L)
                count = 0
            key += 1
            if key == 432:
                shuffle(name_list)
                key = 0

def train(args):

    # generate and store training data online
    Data_Q = Queue(50)
    Label_Q = Queue(50)
    pData = Process(target=genData, 
                    args=(Data_Q,Label_Q,args.batchsize, args.crop_H,args.crop_W))
    pData.start()

    caffe.set_mode_gpu()
    caffe.set_device(args.GPU_ID)
    logging.basicConfig(filename=args.log,level=logging.INFO)

    solver = caffe.AdamSolver(args.solver)

    # initialization
    if args.init_model:
        logging.info('loading'+args.init_model)
        solver.net.copy_from(args.init_model)
        logging.info('loaded')
    solver.net.params['init_norm'][0].data[...] = args.init_p
    solver.net.params['init_pool'][0].data[...] = args.init_p
    norm_param_ul = ['norm_%d' % i for i in xrange(1,args.T)]
    pool_param_ul = ['pool_%d' % i for i in xrange(1,args.T)]
    for i in xrange(args.T-1):
        solver.net.params[norm_param_ul[i]][0].data[...] = args.init_p
        solver.net.params[pool_param_ul[i]][0].data[...] = args.init_p

    solver.net.blobs['data'].reshape(args.batchsize,1,args.crop_H,args.crop_W)
    solver.net.blobs['label'].reshape(args.batchsize,1,args.crop_H,args.crop_W)

    # training process
    epoch = args.num_epochs
    epoch_iters = 750
    step_sum = epoch_iters*epoch+1
    step = 0
    valid_acc = 0
    aver_psnr = 0.0
    best_psnr = 0.0
    best_iter = 0
    best_each_psnr = [0 for i in xrange(19)]
    while step < step_sum:
        if Data_Q.empty() != True:
            # training
            solver.net.blobs['data'].data[...] = Data_Q.get()
            solver.net.blobs['label'].data[...] = Label_Q.get()
            solver.step(1)
            p = solver.net.params['init_norm'][0].data[...]
            log_out = 'iter: %d p: %.4f loss: %f' %  (step,p,solver.net.blobs['loss'].data)
            logging.info(log_out)
            step += 1

            # evaluated on validation or test set.
            if step % epoch_iters == 0:
                psnr, aver_psnr, psnr_set5, psnr_set14 = test_valid(solver.net)
                if (aver_psnr > best_psnr):
                    best_psnr = aver_psnr
                    best_each_psnr = psnr
                    best_iter = step
                log_out = 'test: iter: %d aver_psnr/best_psnr %f/%f psnr_set5: %f psnr_set14: %f\n' % (step,aver_psnr,best_psnr,psnr_set5,psnr_set14)
                logging.info(log_out)
                log_psnr = 'test: iter: %d ' % step
                log_best_psnr = 'best: iter: %d ' % best_iter
                for j in xrange(19):
                    log_psnr = log_psnr + '%d: %f ' % (j,psnr[j-1])
                    log_best_psnr = log_best_psnr + '%d: %f ' % (j,best_each_psnr[j-1])
                logging.info(log_psnr)
                logging.info(log_best_psnr)
                solver.net.blobs['data'].reshape(args.batchsize,1,args.crop_H,args.crop_W)
                solver.net.blobs['label'].reshape(args.batchsize,1,args.crop_H,args.crop_W)

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description="fw_denoising",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fw_parser = parser.add_argument_group('settings')
    fw_parser.add_argument('--GPU_ID', type=int, default=0,
                           help='ID number of GPU card.')
    fw_parser.add_argument('--solver', type=str,
                           help='the solver file, e.g. fw_deno_solver.prototxt')
    fw_parser.add_argument('--T', type=int, default=12,
                           help='number of layers in the neural network.')
    fw_parser.add_argument('--init_p', type=float, default=1.5,
                           help='the initial value of prior p')
    fw_parser.add_argument('--num_epochs', type=int, default=100,
                           help='max num of epochs')
    fw_parser.add_argument('--batchsize', type=int, default=64,
                           help='the batch size')
    fw_parser.add_argument('--init_model', type=str, default=None,
                           help='the pre-trained model.')
    fw_parser.add_argument('--crop_H', type=int, default=42,
                           help='the height of training patch.')
    fw_parser.add_argument('--crop_W', type=int, default=42,
                           help='the width of training patch.')
    fw_parser.add_argument('--log', type=str, default='fw_train.log',
                           help='the path of log file.')
    fw_args = parser.parse_args()

    train(fw_args)
