from dataset import ETRIDataset_color
from networks import *
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed

import logging

from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from sam import SAM
import random
from adamp import AdamP


def seed_everything(seed=214):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_everything(214)

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='Baseline_MNet_color')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--lr', default=0.0001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=214, type=int,
                    help='seed for initializing training. ')

a = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create logs directory if it does not exist
if not os.path.exists('logs'):
    os.makedirs('logs')
# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove existing handlers, if any
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(f"logs/{a.version}_training.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)





def main():
    """ The main function for model training. """

    if os.path.exists('model') is False:
        os.makedirs('model')

    save_path = './model/' + a.version
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    net = efficientformer_color().to(DEVICE)
    
    df = pd.read_csv('./Dataset/Fashion-How24_sub2_train.csv')

    train_dataset = ETRIDataset_color(df, base_path='./Dataset/train/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=a.batch_size, num_workers=4)
    
    df = pd.read_csv('./Dataset/Fashion-How24_sub2_val.csv') 
    val_dataset = ETRIDataset_color(df, base_path='./Dataset/val/', mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    if a.optimizer not in ['Adam', 'AdamW', 'SAM', 'Lion', 'AdamP']:
        raise
    elif a.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=a.lr, amsgrad=True)
    elif a.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr)
    elif a.optimizer == 'SAM':
        optim_ = AdamP
        optimizer = SAM(net.parameters(), optim_, lr=a.lr, weight_decay=0.01, nesterov=True)
    elif a.optimizer == 'AdamP':
        optimizer = AdamP(net.parameters(), lr=a.lr)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    t0_epoch = time.time()

    logger.info("initial learning rate=%.4f" % (optimizer.param_groups[0]['lr']))
    logger.info("initial weight decay rate=%.8f" % (optimizer.param_groups[0]['weight_decay']))
    for epoch in range(a.epochs):

        net.train()
        loss_list = []
        for i, sample in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            for key in sample:
                sample[key] = sample[key].to(DEVICE)
            out = net(sample)
            loss = criterion(out, sample['color_label'])
            loss.backward()
            loss_list.append(loss.item())
            optimizer.first_step(zero_grad=True)
            
            out2 = net(sample)
            loss2 = criterion(out2, sample['color_label'])
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            
        scheduler.step()        
        net.eval()

        val_loss_list = []
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataloader)):
                for key in sample:
                    sample[key] = sample[key].to(DEVICE)
                
                out = net(sample)                
                val_loss = criterion(out, sample['color_label'])
                val_loss_list.append(val_loss.item())

        top_1, acsa = check_performance(net, val_dataloader)

        logger.info('Epoch [{}/{}], Loss: {:.4f}, Time : {:2.3f}, lr : {:.8f}'
                    .format(epoch + 1, a.epochs, np.mean(loss_list), time.time() - t0_epoch, optimizer.param_groups[0]['lr']))
        logger.info("Color: Top-1=%.5f, ACSA=%.5f, Val_loss=%.4f" % (top_1, acsa, np.mean(val_loss_list)))
        t0_epoch = time.time()

        print('Saving Model....')
        torch.save(net.state_dict(), save_path + '/model_' + str(epoch + 1) + '.pt')    # teacher net
        print('OK.')

def check_performance(net, val_dataloader):
    net.eval()

    gt_list = np.array([])
    pred_list = np.array([])

    for j, sample in enumerate(val_dataloader):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        out = net(sample)

        gt = np.array(sample['color_label'].cpu())
        gt_list = np.concatenate([gt_list, gt], axis=0)

        _, indx = out.max(1)
        pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)

    top_1, acsa = get_test_metrics(gt_list, pred_list)

    net.train()

    return top_1, acsa

def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()

if __name__ == '__main__':
    
    main()

