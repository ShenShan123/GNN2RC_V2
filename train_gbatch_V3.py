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
from models_V3 import NetCapClassifier, NetCapRegressor, NetCapRegressorEnsemble
from transformers import AutoModel
# import random

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
                   'mcm': sklearn.metrics.multilabel_confusion_matrix(labels, indices, 
                                                                      labels=[i for i in range(num_classes)]),
                #    'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                  }
        # plot_confmat(labels, indices, metrics, pltname)
        return metrics


def evaluation_c(dataloader: dgl.dataloading.DataLoader, 
                 model: NetCapClassifier, stmodel: AutoModel,
                 class_distr: torch.Tensor, pltname, epoch):
    with torch.no_grad():
        acc = 0
        precisions = torch.zeros((1,model.num_classes))
        recalls = torch.zeros((1, model.num_classes))
        loaderIdx = 0
        mcm = torch.zeros((model.num_classes, 2 , 2))
        for input_nodes, output_nodes, blocks in dataloader:
            loaderIdx += 1
            h_embd = get_sent_embedings(blocks[0].srcdata['tid'].to(stmodel.device), 
                                        blocks[0].srcdata['tid_len'].to(stmodel.device),
                                        stmodel).to(model.device)
            h, _ = model(blocks, h_embd)
            labels = blocks[-1].dstdata['label']
            metrics = validation(h, labels, num_classes=len(class_distr))
            acc += metrics['acc']
            mcm += torch.tensor(metrics["mcm"])
        acc /= loaderIdx
        (tn, fn, tp, fp) = (mcm[:, 0, 0], mcm[:, 1, 0], mcm[:, 1, 1], mcm[:, 0, 1])
        # print("tn, fn, tp, fp", tn, fn, tp, fp)
        precisions = tp / (tp + fp + 1e-3)
        recalls = tp / (tp + fn + 1e-3)
        # print("precisions:", precisions, "recalls:", recalls)
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-3)
        f1_macro = f1.mean()
        f1_weighted = torch.sum(f1 * class_distr) / class_distr.sum()
        return acc, f1_weighted.item(), f1_macro.item()
    
def evaluation_r(dataloader: dgl.dataloading.DataLoader, cmodel: NetCapClassifier, 
                 modelens: NetCapRegressorEnsemble, stmodel: AutoModel,
                 class_distr: torch.Tensor, pltname, epoch):
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
            h_embd = get_sent_embedings(blocks[0].srcdata['tid'].to(stmodel.device), 
                                        blocks[0].srcdata['tid_len'].to(stmodel.device), 
                                        stmodel).to(cmodel.device)
            h, _ = cmodel(blocks, h_embd)
            labels = blocks[-1].dstdata['label'].long()
            metrics = validation(h, labels, num_classes=len(class_distr))
            acc += metrics['acc']
            mcm += torch.tensor(metrics["mcm"])
            prob_t, l_pred = h.max(dim=1, keepdim=True)
            lab_all.append(l_pred)
    
            ## combine the predicted values hi from the ensemble model into h
            h = torch.zeros(l_pred.shape, device=l_pred.device)
            for i, modelr in enumerate(modelens.regressors):
                hi = modelens(blocks, i, h_embd)
                cmask = l_pred == i
                h[cmask] = hi[cmask]
            targets = blocks[-1].dstdata['y']
            metrics, errors = validation_caps(h, targets, 0)
            tar_all.append(targets)
            err_all.append(errors)
            mape += metrics['mean_err']
            mape_max = max(mape_max, metrics['max_err'])
        acc /= loaderIdx
        (tn, fn, tp, fp) = (mcm[:, 0, 0], mcm[:, 1, 0], mcm[:, 1, 1], mcm[:, 0, 1])
        # print("tn, fn, tp, fp", tn, fn, tp, fp)
        precisions = tp / (tp + fp + 1e-3)
        recalls = tp / (tp + fn + 1e-3)
        # print("precisions:", precisions, "recalls:", recalls)
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-3)
        f1_macro = f1.mean()
        f1_weighted = torch.sum(f1 * class_distr) / class_distr.sum()
        ### plot error scatters
        err_all = torch.cat(err_all, dim=0)
        tar_all = torch.cat(tar_all, dim=0)
        lab_all = torch.cat(lab_all, dim=0)
        # print("err size:", err_all.size(), "tar_all size:",tar_all.size())
        plot_errors(err_all.squeeze().tolist(), tar_all.squeeze().tolist(), 
                    lab_all.squeeze().tolist(), pltname, epoch)
        mape /= loaderIdx
        return acc, f1_weighted.item(), f1_macro.item(), mape, mape_max

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

def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

def get_sent_embedings(tids, tid_lens, modelst):
#Compute token embeddings
    with torch.no_grad():
        # tids = datasetTest.bgs.ndata['tid'][:10]
        # tid_lens = datasetTest.bgs.ndata['tid_len'][:10]
        attention_mask = torch.zeros(tids.shape, dtype=torch.int32, device=tids.device)
        for i in range(tids.shape[0]):
            attention_mask[i][:tid_lens[i]] = 1

        model_output = modelst(input_ids=tids, attention_mask=attention_mask)
        # print("model_output[last_hidden_state] shape:", model_output["last_hidden_state"].shape)
        # print("model_output[pooler_output] shape:", model_output["pooler_output"].shape)
        # print("model_output[0] shape:", model_output[0].shape)
        st_embeddings = mean_pooling(model_output, attention_mask)
        # print("st_embeddings shape:", st_embeddings.shape)
        return st_embeddings

def train(dataset: SRAMDatasetList, datasetTest: SRAMDatasetList, 
          cmodel: NetCapClassifier, modelens: NetCapRegressorEnsemble, 
          stmodel: AutoModel):
    device = cmodel.device
    # device2 = stmodel.device
    start = datetime.now()
    """ create a dataloader """
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([256, 32]) # change for num of layers
    train_ids = dataset.bgs.ndata["train_mask"].nonzero().squeeze().int().to(device)
    val_ids = dataset.bgs.ndata["val_mask"].nonzero().squeeze().int().to(device)
    batch_size = 8
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

    """ configuring the optimizer of the gnn classifier """
    loss_fcn1 = FocalLoss(gamma=2, alpha=dataset.alpha)
    # loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam(cmodel.parameters(), lr=1e-3, weight_decay=5e-4)
    max_epoch = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch), 1e-4)
    metric_log = {'train_loss':[], 'val_f1': [], 'val_acc':[], 
                  'test_epoch':[], 'test_f1':[], 'test_acc':[], 
                  'time': 0.0}
    print("Training " + cmodel.name + " classifier at " + start.strftime("%d-%m-%Y_%H:%M:%S"))
    print('Entering training loop of classifier ...')

    """ training loop of classifier """
    for epoch in range(max_epoch):
        cmodel.train()
        train_loss_c = 0
        loaderIdx = 0
        ## minibatch loop
        for input_nodes, output_nodes, blocks in dataloader_train:
            # print("num nodes in", len(blocks), "layers:", blocks[0].num_src_nodes(), 
            #       blocks[0].num_dst_nodes(), blocks[1].num_dst_nodes())
            loaderIdx += 1
            optimizer.zero_grad()
            h_embd = get_sent_embedings(blocks[0].srcdata['tid'].to(stmodel.device), 
                                       blocks[0].srcdata['tid_len'].to(stmodel.device), 
                                       stmodel)
            h, _ = cmodel(blocks, h_embd.to(device))
            labels = blocks[-1].dstdata['label'].long()
            loss_c = loss_fcn1(h, labels.squeeze()) 
            train_loss_c += loss_c.item()
            loss_c.backward()
            optimizer.step()
            # assert 0
        scheduler.step()
        ## do validations and evaluations ###
        cmodel.eval()
        print('Validating ...')
        acc, f1_weighted, f1_macro = evaluation_c(dataloader_val, cmodel, stmodel,
                                                  dataset.class_distr,
                                                  dataset.name+cmodel.name, epoch)
        print("|| Epoch {:05d} | Loss {:.4f} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} ||"
            .format(epoch, train_loss_c/loaderIdx,    acc*100,      f1_weighted,             f1_macro))
            
        ## testing
        print('Testing...')
        acc, f1_weighted, f1_macro_test = evaluation_c(dataloader_test, cmodel, stmodel,
                                                       dataset.class_distr, 
                                                       datasetTest.name+cmodel.name, epoch)
        print("|| Test | runtime {:s} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} ||"
            .format(str(datetime.now()-start), acc*100,      f1_weighted,               f1_macro_test))

        if f1_macro_test > 0.8 or f1_macro_test > 0.8*f1_macro:
            break

    print("Classifier training is finished.")

    """ configuring optimizers of gnn regressors """
    start = datetime.now()
    loss_fcn2 = ud_loss
    optimizers = []
    schedulers = []
    for rmodel in modelens.regressors:
        optimizers.append(torch.optim.Adam(rmodel.parameters(), lr=1e-3, weight_decay=5e-4))
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[-1], float(max_epoch), 1e-4))
    print("Training " + modelens.name + " regression at " + start.strftime("%d-%m-%Y_%H:%M:%S"))
    print('Entering training loop of regressors ...')
    ### training loop of regressors ###
    for epoch in range(max_epoch):
        for rmodel in modelens.regressors: 
            rmodel.train() 
        train_loss_r = 0
        loaderIdx = 0
        
        ## dataset loop
        for input_nodes, output_nodes, blocks in dataloader_train:
            loaderIdx += 1
            h_embd = get_sent_embedings(blocks[0].srcdata['tid'].to(stmodel.device), 
                                        blocks[0].srcdata['tid_len'].to(stmodel.device), 
                                        stmodel).to(device)
            h, _ = cmodel(blocks, h_embd)
            prob_t, l_pred = h.detach().max(dim=1, keepdim=True)
            targets = blocks[-1].dstdata['y']
            for i, optimizer in enumerate(optimizers): 
                optimizer.zero_grad()
                
                h = modelens(blocks, i, h_embd)
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
        for rmodel in modelens.regressors: 
            rmodel.eval() 
        print('Validating...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_r(dataloader_val, cmodel, modelens, 
                                                                  stmodel, dataset.class_distr, 
                                                                  dataset.name+cmodel.name, epoch)
        print("|| Epoch {:05d} | Loss {:.4f} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} | mape {:.2f}% | mape_max {:.2f}% ||"
              .format(epoch, train_loss_r/loaderIdx, acc*100, f1_weighted, f1_macro, mape*100, mape_max*100))
            
        ## testing
        print('Testing...')
        acc, f1_weighted, f1_macro, mape, mape_max = evaluation_r(dataloader_test, cmodel, modelens, 
                                                                  stmodel, datasetTest.class_distr, 
                                                                  datasetTest.name+cmodel.name, epoch)
        print("|| Test | runtime {:s} | class acc {:.2f}%  | class f1_weighted {:.4f} | f1_macro {:.4f} | mape {:.2f}% | mape_max {:.2f}% ||"
              .format(str(datetime.now()-start), acc*100, f1_weighted, f1_macro, mape*100, mape_max*100))