import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from SRAM_dataset import SRAMDataset
from focal_loss_pytorch.focalloss import FocalLoss
# import dgl
# from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import os 
import matplotlib as mpl
from regression import validation_caps, ud_loss
from models import NetClassifier
from bgnn import BGNNPredictor

def train(dataset: SRAMDataset, gnn_model, device):
    start = datetime.now()
    bg = dataset._bg.to(device)
    h_dict = dataset.get_feat_dict()
    labels = dataset.get_labels().to(device)
    targets = dataset.get_targets().to(device)
    # get train/validation split
    train_nids, val_nids, test_nids = dataset.get_nids()
    train_nids = torch.cat((train_nids, val_nids))
    loss_fcn1 = FocalLoss(gamma=2, alpha=dataset.alpha)
    weights = dataset.alpha.expand(len(labels), len(dataset.alpha)).gather(1,labels)
    loss_fcn2 = ud_loss

    bgnn = BGNNPredictor(gnn_model, task='regression',
                         gnn_loss_fn=loss_fcn1)
                        #  trees_per_epoch=trees_per_epoch,
                        #  backprop_per_epoch=backprop_per_epoch,
                        #  lr=lr,
                        #  append_gbdt_pred=append_gbdt_pred,
                        #  train_input_features=train_input_features,
                        #  gbdt_depth=gbdt_depth,
                        #  gbdt_lr=gbdt_lr)
    X = h_dict['net']
    # train
    metrics = bgnn.fit(dataset._bg, h_dict, labels, targets, train_nids, val_nids, test_nids, 
                       num_epochs=200, patience=50)

    # ### training loop ###
    # for epoch in range(max_epoch):
    #     model.train()
    
    #     val_loss = 0
    #     optimizer.zero_grad()
        
    #     t, h = model(h_dict, bg)
    #     loss_c = loss_fcn1(t[train_nids], labels[train_nids].squeeze()) 
    #     loss_r = loss_fcn2(h[train_nids], targets[train_nids], weights=weights[train_nids])
    #     print("loss c:", loss_c.item(), "loss r:", loss_r.item())
    #     loss = loss_c + loss_r
    #     val_loss = loss.item()
    #     loss.backward()
    #     optimizer.step()
    #     # scheduler.step()
        
    #     ### do validations and evaluations ###
    #     model.eval()

    #     print('Validating...')
    #     logits, clogits = model(h_dict, bg)
    #     metrics = validation(logits, labels, mask=test_nids, pltname=model.name+"_conf_matrix_valid")
    #     metric_log['train_loss'].append(val_loss)
    #     metric_log['val_acc'].append(metrics['acc'])
    #     metric_log['val_f1'].append(metrics['f1_macro'])
    #     print("|| Epoch {:05d} | Loss {:.4f} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
    #           .format(epoch,     val_loss,     metrics['acc']*100, 
    #                   metrics['f1_weighted'],  metrics['f1_macro']))

    #     metrics = validation_caps(clogits, targets, 1, mask=test_nids, pltname="err_in_eval_"+str(1))
    #     print("|| train/test time {:s}/{:s} | mean error {:.2f}%  | max error {:.2f}% ||"
    #             .format(str(datetime.now()-start), str(datetime.now()-start), 
    #                     metrics['mean_err']*100, metrics['max_err']*100))
        
    # return test_metrics

if __name__ == '__main__':
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dataset = SRAMDataset(name='sandwich', raw_dir='/data1/shenshan/RCpred')
    linear_dict = {'device': [dataset._d_feat_dim, 64, 64], 
                   'inst':   [dataset._i_feat_dim, 64, 64], 
                   'net':    [dataset._n_feat_dim+1, 64, 64]}
    gnn_model = NetClassifier(num_class=dataset._num_classes, proj_dim_dict=linear_dict, 
                            gnn='sage-mean', has_l2norm=False, has_bn=False, dropout=0.1, 
                            device=device)
    train(dataset, gnn_model, device)

    