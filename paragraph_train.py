import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from SRAM_dataset_gbatch import SRAMDatasetList
from focal_loss_pytorch.focalloss import FocalLoss
# import copy
# from dgl.dataloading import DataLoader
import dgl
# from tqdm import tqdm
# import torch.nn.functional as F
from datetime import datetime
import os 
import matplotlib as mpl
from regression import validation_caps, ud_loss, plot_errors
from models import NetCapClassifier, NetCapRegressor, NetCapRegressorEnsemble

def evaluation_r(dataloader: dgl.dataloading.DataLoader, cmodel: NetCapClassifier, 
                 modelens: NetCapRegressorEnsemble, class_distr: torch.Tensor, 
                 pltname, epoch):
    with torch.no_grad():
        acc = 0
        precisions = torch.zeros((1,cmodel.num_classes))
        recalls = torch.zeros((1, cmodel.num_classes))
        loaderIdx = 0
        mape = 0
        mcm = torch.zeros((cmodel.num_classes, 2 , 2))
        mape_max = 0
        tar_all = []
        err_all = []
        lab_all = []
        for input_nodes, output_nodes, blocks in dataloader:
            loaderIdx += 1
    
            ## combine the predicted values hi from the ensemble model into h
            for i, modelr in enumerate(modelens.layers):
                hi = modelens(blocks, i)
                cmask = l_pred == i
                h[cmask] = hi[cmask]
            targets = blocks[-1].dstdata['y']
            metrics, errors = validation_caps(h, targets, 0)
            tar_all.append(targets)
            err_all.append(errors)
            mape += metrics['mean_err']
            mape_max = max(mape_max, metrics['max_err'])
        acc /= loaderIdx
        
        ### plot error scatters
        err_all = torch.cat(err_all, dim=0)
        tar_all = torch.cat(tar_all, dim=0)
        lab_all = torch.cat(lab_all, dim=0)
        # print("err size:", err_all.size(), "tar_all size:",tar_all.size())
        plot_errors(err_all.squeeze().tolist(), tar_all.squeeze().tolist(), 
                    lab_all.squeeze().tolist(), pltname, epoch)
        mape /= loaderIdx
        return mape, mape_max

def train(dataset: SRAMDatasetList, datasetTest: SRAMDatasetList, cmodel, modelens, device):
    start = datetime.now()
    max_epoch = 200
    """ create a dataloader """
    sampler = dgl.dataloading.MultiLayerNeighborSampler([1024, 128, 32]) # change for GAT
    # # change for num of layers
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) 
    train_ids = dataset.bgs.ndata["train_mask"].nonzero().squeeze().int().to(device)
    val_ids = dataset.bgs.ndata["val_mask"].nonzero().squeeze().int().to(device)
    batch_size = 32
    dataloader_train = dgl.dataloading.DataLoader(dataset.bgs, train_ids, 
                                                  sampler, batch_size=batch_size, 
                                                  shuffle=True, drop_last=False,
                                                  device=device)
    dataloader_val = dgl.dataloading.DataLoader(dataset.bgs, val_ids, 
                                                sampler, batch_size=batch_size, 
                                                shuffle=True, drop_last=False,
                                                device=device)
    test_ids = datasetTest.bgs.ndata["train_mask"].nonzero().squeeze().int().to(device)
    dataloader_test = dgl.dataloading.DataLoader(datasetTest.bgs, test_ids, 
                                                sampler, batch_size=batch_size, 
                                                shuffle=True, drop_last=False,
                                                device=device)

    start = datetime.now()
    loss_fcn2 = ud_loss
    optimizers = []
    schedulers = []
    for rmodel in modelens.layers:
        optimizers.append(torch.optim.Adam(rmodel.parameters(), lr=1e-3, weight_decay=5e-4))
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[-1], float(max_epoch), 1e-4))
    print("Training " + modelens.name + " regression at " + start.strftime("%d-%m-%Y_%H:%M:%S"))
    print('Entering training loop...')
    ### training loop of regressors ###
    for epoch in range(max_epoch):
        for rmodel in modelens.layers: 
            rmodel.train() 
        train_loss_r = 0
        loaderIdx = 0
        
        ## dataset loop
        for input_nodes, output_nodes, blocks in dataloader_train:
            loaderIdx += 1
            h, _ = cmodel(blocks)
            prob_t, l_pred = h.detach().max(dim=1, keepdim=True)
            targets = blocks[-1].dstdata['y']
            for i, optimizer in enumerate(optimizers): 
                optimizer.zero_grad()
                h = modelens(blocks, i)
                cmask = (l_pred == i) 
                ## backpropagation will start only if #samples >= 5
                if cmask.sum() < 5:
                    continue
                # ## add noisy data samples into training set, on 2023-10-16
                # if cmask.sum()*1.2 < len(cmask) and (~cmask).any():
                #     rand_idx = (~cmask).squeeze().nonzero().squeeze()
                #     rand_idx = rand_idx[torch.randperm(len(rand_idx))]
                #     noise_num = int(cmask.sum()*0.2) 
                #     rand_idx = rand_idx[:noise_num]
                #     cmask[rand_idx] = True

                loss = loss_fcn2(h[cmask], targets[cmask])#, weights=weights[cmask0])
                # print("loss item:", loss.item(), "tran_loss_r", train_loss_r)
                train_loss_r += loss.item()
                loss.backward()
                optimizer.step()

        for i, scheduler in enumerate(schedulers): 
            scheduler.step()
        ## evaluation part
        for rmodel in modelens.layers: 
            rmodel.eval() 
        print('Validating...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_r(dataloader_val, cmodel, modelens, 
                                                                  dataset.class_distr, 
                                                                  dataset.name+cmodel.name, epoch)
        print("|| Epoch {:05d} | Loss {:.4f} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} | mape {:.2f}% | mape_max {:.2f}% ||"
              .format(epoch, train_loss_r/loaderIdx, acc*100, f1_weighted, f1_macro, mape*100, mape_max*100))
            
        ## testing
        print('Testing...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_r(dataloader_test, cmodel, modelens, 
                                                                  datasetTest.class_distr, 
                                                                  datasetTest.name+cmodel.name, epoch)
        print("|| Test | runtime {:s} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} | mape {:.2f}% | mape_max {:.2f}% ||"
              .format(str(datetime.now()-start), acc*100, f1_weighted, f1_macro, mape*100, mape_max*100))