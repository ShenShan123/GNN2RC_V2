import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from SRAM_dataset_gbatch import SRAMDatasetList
from focal_loss_pytorch.focalloss import FocalLoss
# import copy
from dgl.dataloading import DataLoader
import dgl
# from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import os 
import matplotlib as mpl
from regression import validation_caps, ud_loss, plot_errors
from models import NetCapPredictor
import random

def plot_confmat(y_true, y_pred, metrics, pltname):
    plt.style.use('seaborn-darkgrid')
    # mpl.rcParams['font.sans-serif'] = ['Calibri']
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['axes.labelsize'] = 13
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.figsize'] = (8.47, 4.3)
    cf_matrix = sklearn.metrics.confusion_matrix(y_true.squeeze(), y_pred, normalize='true')
    fig, ax = plt.subplots()
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    sns.heatmap(cf_matrix, annot=True, fmt='.2%', cbar=True)#,  cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    title = ' '.join(key + ":" + "{:.4f}".format(val) for key, val in metrics.items())
    plt.title(title)
    # plt.savefig('./data/plots/'+pltname, dpi=400)
    # plt.savefig("data/plots/conf_matrix_pool_infr.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

""" quick validation in trainning epoch """
def validation(logits, labels, mask=None, num_classes=5, pltname="conf_matrix_valid"):
    with torch.no_grad():
        # use the labels in validation set
        if mask is not None:
            logits = logits[mask].cpu().detach()
            labels = labels[mask].cpu().detach().numpy()
        else:
            logits = logits.cpu().detach()
            labels = labels.cpu().detach().numpy()

        indices = torch.argmax(logits, dim=1)
        metrics = {'acc': sklearn.metrics.accuracy_score(labels, indices),
                   'mcm': sklearn.metrics.multilabel_confusion_matrix(labels, indices, labels=[i for i in range(num_classes)]),
                #    'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                  }
        # plot_confmat(labels, indices, metrics, pltname)
        return metrics

""" Just use in testing """
# def evaluation(bg, h_dict, model_list: nn.ModuleList()):
#     with torch.no_grad():
#         trans_h = model_list[0](h_dict)#[input_nodes.long()]
#         dst_h = model_list[1](bg, trans_h)
#         net_h = dst_h[-h_dict['net'].shape[0]:]
#         logits = model_list[2](net_h)
#         return logits, net_h


def evaluation_c(dataloader: dgl.dataloading.DataLoader, model: NetCapPredictor, 
                 class_distr: torch.Tensor, pltname, epoch):
    # l_pred = torch.empty((0,5), dtype=torch.int32, device=dataloader.device)
    # h_pred = torch.empty((0,1), dtype=torch.float32, device=dataloader.device)
    # labels = torch.empty((0,1), dtype=torch.int32, device=dataloader.device)
    # targets = torch.empty((0,1), dtype=torch.float32, device=dataloader.device)
    with torch.no_grad():
        acc = 0
        precisions = torch.zeros((1,model.num_classes))
        recalls = torch.zeros((1, model.num_classes))
        loaderIdx = 0
        mape = 0
        mcm = torch.zeros((model.num_classes, 2 , 2))
        mape_max = 0
        # label_tmp = torch.tensor([[0],[1],[2],[3],[4]])
        for input_nodes, output_nodes, blocks in dataloader:
            # logits, clogits = model(h_dict, bg)
            l, h = model([dataset._d_feat_dim, dataset._i_feat_dim, dataset._n_feat_dim],
                        blocks)
            # l_pred = torch.cat((l_pred, l), dim=0)
            # h_pred = torch.cat((h_pred, h), dim=0)
            labels = blocks[-1].dstdata['label']
            # targets = blocks[-1].dstdata['y']
            loaderIdx += 1
            # print("loaderIdx in eval", loaderIdx)
            metrics = validation(l, labels, num_classes=len(class_distr))
            acc += metrics['acc']
            # print("mcm:", metrics["mcm"])
            mcm += torch.tensor(metrics["mcm"])
            # print("targets:", targets)
            # print("tar pred:", h)
            # metrics = validation_caps(h, targets, 0)
            # mape += metrics['mean_err']
            # mape_max = max(mape_max, metrics['max_err'])
            # if loaderIdx % 400 == 0:
            #     print("labels:", labels.squeeze()[:10])
            #     print("l pred:", l.argmax(dim=1).squeeze()[:10])
                # print("targets:", targets.squeeze()[:10])
                # print("t pred:", h.squeeze()[:10])
            # assert 0
        # return l_pred, h_pred, labels, targets
        acc /= loaderIdx
        (tn, fn, tp, fp) = (mcm[:, 0, 0], mcm[:, 1, 0], mcm[:, 1, 1], mcm[:, 0, 1])
        # print("tn, fn, tp, fp", tn, fn, tp, fp)
        precisions = tp / (tp + fp + 1e-3)
        recalls = tp / (tp + fn + 1e-3)
        # print("precisions:", precisions, "recalls:", recalls)
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-3)
        f1_macro = f1.mean()
        f1_weighted = torch.sum(f1 * class_distr) / class_distr.sum()
        mape /= loaderIdx
        return acc, f1_weighted.item(), f1_macro.item(), mape, mape_max

def evaluation_r(dataloader: dgl.dataloading.DataLoader, model: NetCapPredictor, 
                 class_distr: torch.Tensor, pltname, epoch):
    with torch.no_grad():
        acc = 0
        precisions = torch.zeros((1,model.num_classes))
        recalls = torch.zeros((1, model.num_classes))
        loaderIdx = 0
        mape = 0
        mcm = torch.zeros((model.num_classes, 2 , 2))
        mape_max = 0
        tar_all = []
        err_all = []
        # label_tmp = torch.tensor([[0],[1],[2],[3],[4]])
        for input_nodes, output_nodes, blocks in dataloader:
            # logits, clogits = model(h_dict, bg)
            l, h = model([dataset._d_feat_dim, dataset._i_feat_dim, dataset._n_feat_dim],
                        blocks)

            targets = blocks[-1].dstdata['y']
            tar_all.append(targets)
            loaderIdx += 1
            metrics, errors = validation_caps(h, targets, 0)
            err_all.append(errors)
            mape += metrics['mean_err']
            mape_max = max(mape_max, metrics['max_err'])
            # if loaderIdx % 400 == 0:
            #     print("targets:", targets.squeeze()[:10])
            #     print("t pred:", h.squeeze()[:10])

        err_all = torch.cat(err_all, dim=0)
        tar_all = torch.cat(tar_all, dim=0)
        # print("err size:", err_all.size(), "tar_all size:",tar_all.size())
        plot_errors(err_all.squeeze(), tar_all.squeeze(), pltname, epoch)
        mape /= loaderIdx
        return mape, mape_max

def classifier_save(model_list: nn.ModuleList(), val_metrics=None, test_metrics=None, epoch=None):
    name = '_'.join(model.__class__.__name__ for model in model_list)
    if "GraphSAGE" in name:
        name = name.replace("GraphSAGE", "GraphSAGE_"+model_list[1].layers[0]._aggre_type) 
    if val_metrics is not None:
        val_results = '_'.join(key+"_{:.2f}".format(val) for key, val in val_metrics.items())
        test_results = '_'.join(key+"_{:.2f}".format(val) for key, val in test_metrics.items())
        # here we remove the saved models with similar results
        os.system("rm " + "data/models/"+name+"_"+val_results+"_"+test_results+"_[0-9]*")
        torch.save(model_list, "data/models/"+name+"_"+val_results+
                            "_"+test_results+"_"+str(epoch)+".pt")
    else:
        torch.save(model_list, "data/models/"+name+".pt")

def plot_metric_log(metric_log: dict, name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(25, 11)
    x = range(len(metric_log['train_loss']))
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_title('training epochs')
    ax1.plot(x, metric_log['val_acc'],'--', label='acc')
    ax1.plot(x, metric_log['val_f1'],'-.', label='f1')
    ax1r = ax1.twinx()
    ax1r.plot(x, metric_log['train_loss'],'-', label='loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('metrics')
    ax1r.set_ylabel('loss')

    ax2.set_title('testing epochs')
    x = metric_log['test_epoch']
    ax2.plot(x, metric_log['test_acc'],'--', label='acc')
    ax2.plot(x, metric_log['test_f1'],'-.', label='f1')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('metrics')
    plt.title(name, loc='left')
    plt.legend()
    plt.savefig("data/plots/train_log_" + name + ".png", dpi=500, bbox_inches='tight')
    plt.close(fig)

def train(dataset: SRAMDatasetList, datasetTest: SRAMDatasetList, model, device):
    start = datetime.now()
    ### create a dataloader ###
    # net_ids = dataset.hgs.nodes['net'].data['train_mask'].nonzero()
    # print("train id:", net_ids)
    # assert 0
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) # change for 3-layer graphSAGE
    # print("train ids:", dataset.bgs.ndata["train_mask"].nonzero())
    # assert 0
    train_ids = dataset.bgs.ndata["train_mask"].nonzero().squeeze().int().to(device)
    val_ids = dataset.bgs.ndata["val_mask"].nonzero().squeeze().int().to(device)
    # print("train_ids shape:", train_ids.shape)
    # print("val_ids shape:", val_ids.shape)

    batch_size = 128
    dataloader = dgl.dataloading.DataLoader(dataset.bgs, train_ids, 
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

    ### other settings ###
    # train_nids = torch.cat((train_nids, val_nids))
    loss_fcn1 = FocalLoss(gamma=2, alpha=dataset.alpha)
    
    # print("weights:", weights)
    # print("weights shape:", weights.shape)
    # assert 0
    # loss_fcn2 = ud_loss
    # loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    max_epoch = 500
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch), 1e-3)
    # alpha_weights = 1.0 / dataset.alpha.expand(batch_size, len(dataset.alpha)).detach()
    # # print("weights", alpha_weights[0])
    # # alpha_weights2 = F.softmax(alpha_weights, dim=1)
    # # print("weights after softmax", alpha_weights2[0])
    # alpha_weights = (1-alpha_weights)**2 
    # print("alpha_weights after gamma", alpha_weights[0])
    # # assert 0
    # alpha_weights = alpha_weights.to(device)
    # best_val_loss = torch.inf
    # best_test_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    # best_val_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    # best_loss_metrics = {}
    # bad_count = 0
    # best_count = 0
    # patience = 25
    # test_metrics = {}
    # val_metrics = {}
    metric_log = {'train_loss':[], 'val_f1': [], 'val_acc':[], 
                  'test_epoch':[], 'test_f1':[], 'test_acc':[], 
                  'time': 0.0}
    print("Training " + model.name + " classifier at " + start.strftime("%d-%m-%Y_%H:%M:%S"))
    print('Entering training loop...')

    ### training loop ###
    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        loaderIdx = 0
        
        # dataset loop
        for input_nodes, output_nodes, blocks in dataloader:
            optimizer.zero_grad()
            # print("in nodes:", input_nodes)
            # print("out nodes:", output_nodes)
            # print("block[0] ndata:", blocks[0].ndata)
            # print(blocks)
            # feats = blocks[0].ndata['x']
            # ntypes = blocks[0].ndata['_TYPE']
            # dim_list = [dataset._d_feat_dim, dataset._i_feat_dim, dataset._n_feat_dim]
            # bg = dataset.bgList[i]
            # h_dict = dataset.featDictList[i]
            # labels = dataset.labelList[i]
            # targets = dataset.targetList[i]
            # # get train/validation split
            # train_nids = dataset.nidTrainList[i]
            # val_nids = dataset.nidValList[i]
            # t, h = model(h_dict, bg)
            l, h = model([dataset._d_feat_dim, dataset._i_feat_dim, dataset._n_feat_dim],
                         blocks)
            # print("h size", h.shape)
            labels = blocks[-1].dstdata['label'].long()
            # targets = blocks[-1].dstdata['y']
            
            # weights /= weights.squeeze().sum()
            # weights = alpha_weights[:len(labels)]
            # weights = weights.gather(1,labels)
            # print("labels:", labels)
            # print("l", l)
            # print("h", h)
            # print("targets:", targets)
            # print("weights after gather", weights[0])
            # assert 0
            loss_c = loss_fcn1(l, labels.squeeze()) 
            # loss_r = loss_fcn2(h, targets, weights=weights)
            # if loaderIdx % 400 == 0:
            #     # print("loaderIdx", loaderIdx, "loss c:", loss_c.item(), "loss r:", loss_r.item())
            #     print("loaderIdx", loaderIdx, "loss r:", loss_r.item())
            loaderIdx += 1
            loss = loss_c
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # scheduler.step()
        ### do validations and evaluations ###
        model.eval()
        print('Validating...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_c(dataloader_val, model, dataset.class_distr, 
                                                                  dataset.name+model.name+"_downsample", epoch)
        # assert 0
        print("|| Epoch {:05d} | Loss {:.4f} | class acc {:.2f}%  | class f1_weighted {:.2f} | f1_macro {:.2f} ||"
              .format(epoch, train_loss/loaderIdx,    acc*100,             f1_weighted,             f1_macro))
        # print("|| Epoch {:05d} | Loss {:.4f} | mape {:.2f}% | mape_max {:.2f}% ||"
        #       .format(epoch, train_loss/loaderIdx,  mape*100, mape_max*100))
        
        # print("|| train/test time {:s}/{:s} | mean error {:.2f}%  | max error {:.2f}% ||"
        #         .format(str(datetime.now()-start), str(datetime.now()-start), 
        #                 metrics['mean_err']*100, metrics['max_err']*100))
        
        # testing
        print('Testing...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_c(dataloader_test, model, dataset.class_distr, 
                                                                  datasetTest.name+model.name+"_downsample", epoch)

        # print("l_pred:", l_pred.squeeze())
        # print("labels:", labels.squeeze())
        print("|| Test Epoch {:05d} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(epoch,    acc*100,  f1_weighted,  f1_macro))
        
        # print("|| Test | runtime {:s} | class acc {:.2f}%  | class f1_weighted {:.2f} | f1_macro {:.2f} | mape {:.2f}% | mape_max {:.2f}% ||"
        #       .format(str(datetime.now()-start), acc*100,    f1_weighted,               f1_macro,         mape*100,      mape_max*100))
        # print("|| Test | runtime {:s} | mape {:.2f}% | mape_max {:.2f}% ||"
        #       .format(str(datetime.now()-start), mape*100, mape_max*100))

        # metrics = validation_caps(h_pred, targets, 1, mask=None, pltname="err_in_eval_"+str(1))
        # print("h_pred:", h_pred.squeeze())
        # print("targets:", targets.squeeze())
        # print("|| Test time {:s}/{:s} | mean error {:.2f}%  | max error {:.2f}% ||"
        #         .format(str(datetime.now()-start), str(datetime.now()-start), 
        #                 metrics['mean_err']*100, metrics['max_err']*100))
            
    # return test_metrics

if __name__ == '__main__':
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    # mcm = sklearn.metrics.multilabel_confusion_matrix([0,1,2,3,3], [0,0,2,3,3], labels=[i for i in range(5)])
    # print(mcm)
    # assert 0
    device = torch.device('cuda:0') 
    dataset = SRAMDatasetList(device=device, test_ds=False)
    datasetTest = SRAMDatasetList(device=device, test_ds=True, featMax=dataset.featMax)
    # assert 0
    linear_dict = {'device': [dataset._d_feat_dim, 64, 64], 
                   'inst':   [dataset._i_feat_dim, 64, 64], 
                   'net':    [dataset._n_feat_dim, 64, 64]}
    model = NetCapPredictor(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
                            gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
                            device=device)
    train(dataset, datasetTest, model, device)