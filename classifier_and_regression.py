import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Hete2HomoLayer, GAT, GATv2, MLP3, GCN, GraphSAGE, MLPN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics
from SRAM_dataset import SRAMDataset
from focal_loss_pytorch.focalloss import FocalLoss
import copy
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

def plot_confmat(y_true, y_pred, metrics, pltname):
    cf_matrix = sklearn.metrics.confusion_matrix(y_true.squeeze(), y_pred, normalize='true')
    fig, ax = plt.subplots()
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    sns.heatmap(cf_matrix, annot=True, fmt='.2%', cbar=True)#,  cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    title = ' '.join(key + ":" + "{:.4f}".format(val) for key, val in metrics.items())
    plt.title(title)
    plt.savefig('./data/plots/'+pltname, dpi=400)
    plt.close(fig)

""" quick validation in trainning epoch """
def validation(logits, targets, mask, pltname="conf_matrix_valid"):
    with torch.no_grad():
        # use the labels in validation set
        logits = logits[mask]
        # print("logits in val")
        # print(logits)
        # logits_norm = F.normalize(logits, p=1, dim=1).cpu().detach().numpy()
        # print('logits_norm:', logits_norm)
        targets = targets[mask].cpu().detach().numpy()
        _, indices = torch.max(logits, dim=1)

        metrics = {'acc': sklearn.metrics.accuracy_score(targets, indices),
                   'f1_weighted': sklearn.metrics.f1_score(targets, indices, average='weighted'),
                   'f1_macro': sklearn.metrics.f1_score(targets, indices, average='macro'),
                #    'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                  }
        plot_confmat(targets, indices, metrics, pltname)
        return metrics

""" Just use in testing """
def evaluation(shg, bg, h_dict, model_list: nn.ModuleList()):
    with torch.no_grad():
        trans_h = model_list[0](shg, h_dict)
        dst_h = model_list[1](bg, trans_h)
        # we only consider the dst nodes' labels
        dst_h = dst_h[-shg.num_nodes('net'):]
        logits = model_list[2](dst_h)
        # print("model parameters in eval:")
        # print_params(model_list)
        # print("logits in eval:", logits[mask])
        return logits, dst_h
    # return validation(logits, targets, mask, pltname)

def print_params(model_list: nn.ModuleList()):
    # for model in model_list:
    for name, params in model_list[0].named_parameters():
        # if name == "layers.1.bias":
        #     print(name, ":", params[:10])
        # if name == "layers.1.fc_self.weight":
        #     print(name, ":", params[0][:10])
        # if name == "layers.1.fc_neigh.weight":
        print(name, ":", params[0])

def classifier_save(model_list: nn.ModuleList(), val_metrics=None, test_metrics=None, epoch=None):
    name = '_'.join(model.__class__.__name__ for model in model_list)
    if val_metrics is not None:
        val_results = '_'.join(key+"_{:.2f}".format(val) for key, val in val_metrics.items())
        test_results = '_'.join(key+"_{:.2f}".format(val) for key, val in test_metrics.items())
        torch.save(model_list, "data/models/"+name+"_"+val_results+
                            "_"+test_results+"_"+str(epoch)+".pt")
    else:
        torch.save(model_list, "data/models/"+name+"_baseConf.pt")

def train(dataset: SRAMDataset, model_list: nn.ModuleList()):
    shg = dataset._shg
    bg = dataset._bg
    h_dict = dataset.get_feat_dict()
    labels = dataset.get_labels()
    test_mask = dataset.get_test_mask()
    # define train/val samples, loss function and optimizer
    loss_fcn = FocalLoss(gamma=2, alpha=dataset.alpha)
    optimizer = torch.optim.Adam([{'params' : model.parameters()} for model in model_list], 
                                 lr=5e-3, weight_decay=5e-4)
    
    best_val_loss = torch.inf
    bad_count = 0
    best_count = 0
    best_epoch = -1
    best_val_epoch = -1
    best_test_epoch = -1
    test_metrics = {}
    best_test_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    val_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    best_val_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    patience = 25
    model_list_cp = nn.ModuleList()
    # training loop
    for epoch in range(500):
        for i, model in enumerate(model_list):
            model.train()

        optimizer.zero_grad()
        trans_h = model_list[0](shg, h_dict)
        dst_h = model_list[1](bg, trans_h)
        # we only consider the net nodes' labels
        dst_h = dst_h[-labels.shape[0]:]
        assert dst_h.shape[0] == dataset._num_n
        logits = model_list[2](dst_h)
        
        val_mask, train_mask = dataset.get_val_mask(test_mask)
        loss = loss_fcn(logits[train_mask], labels[train_mask].squeeze())
        val_loss = loss.item()
        loss.backward()
        optimizer.step()

        name = model_list[1].__class__.__name__
        
        # do validations and evaluations
        for i, model in enumerate(model_list):
            model.eval()

        metrics = validation(logits, labels, val_mask, name+"_conf_matrix_valid")
        print("|| Epoch {:05d} | Loss {:.4f} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(epoch, loss.item(), metrics['acc']*100, metrics['f1_weighted'], metrics['f1_macro']))
        
        # for ealy stop with 25 patience
        if ((best_val_loss > val_loss) or (best_val_metrics['f1_macro'] < metrics['f1_macro'])) and (epoch > 50):
            if (best_val_loss > val_loss):
                best_val_loss = val_loss
                best_epoch = epoch
                val_metrics = metrics
            if best_val_metrics['f1_macro'] < metrics['f1_macro']:
                best_val_epoch = epoch
                best_val_metrics = metrics
            
            bad_count = 0
            print('Testing...')
            logits, _ = evaluation(shg, bg, h_dict, model_list)
            test_metrics = validation(logits, labels, test_mask, name+"_conf_matrix_eval")
            print("|| test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
                  .format(test_metrics['acc']*100, test_metrics['f1_weighted'], test_metrics['f1_macro']))
            # keep track the best testing result
            if best_test_metrics['f1_macro'] < test_metrics['f1_macro']:
                best_test_metrics = test_metrics
                best_test_epoch = epoch
            
            if (metrics['f1_macro'] > 0.8 and test_metrics['f1_macro'] > 0.8 and
                metrics['acc'] > 0.9 and test_metrics['acc'] > 0.9):
                best_count += 1
                model_list_cp = copy.deepcopy(model_list)
                classifier_save(model_list, metrics, test_metrics, epoch)
            else:
                best_count = 0
            model_list_cp = copy.deepcopy(model_list)
        # training this model at leat (200+patience) times
        elif epoch > 100:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break
    
    classifier_save(model_list_cp)
    print("|* Best epoch:{:d} loss: {:.4f} *|" \
          .format(best_val_epoch, best_val_loss))
    print("|* Best validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          .format(best_val_metrics['acc']*100, best_val_metrics['f1_weighted'], best_val_metrics['f1_macro']))
    print("|* Best test epoch:{:d} *|".format(best_test_epoch))
    print("|* Best test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          .format(best_test_metrics['acc']*100, best_test_metrics['f1_weighted'], best_test_metrics['f1_macro']))
    print("|* Best epoch continue iterations: {:d} *|".format(best_count))
    return test_metrics

def plot_errors(x, y, pltname, c):
    plt.grid()
    plt.yscale('log')
    fig, axs = plt.subplots()
    axs.scatter(x.tolist(), y.tolist(), s=5, marker=".", alpha=0.5)
    # for ax in axs.flat:
    axs.set(xlabel='relative errors', ylabel='cap (fF)')

    absx = torch.abs(x)
    max_err, _ = torch.max(absx, dim=0)
    fig.suptitle('Class:{:d} mean error:{:.2f}%, max error:{:.2f}%'
                 .format(c, torch.mean(absx).item()*100, max_err.item()*100))
    plt.savefig('./data/plots/'+pltname, dpi=400)
    plt.close(fig)

def ud_loss(logits, targets):
    mask = (targets != torch.inf) &  (logits != torch.inf) &  (targets != 0.0)
    logits = torch.squeeze(logits[mask])
    targets = torch.squeeze(targets[mask])
    diffs = logits - targets
    rel_w = torch.div(diffs, targets)
    squ_rel_w = torch.square(rel_w)
    loss_value = torch.mean(squ_rel_w)
    max_loss, max_i = torch.max(loss_value, dim=0)
    # print('in loss with max loss: logits:{:.4f}, targets:{:.4f}, loss:{:.4f}, with index:{:05d}'
    #       .format(logits[max_i].item(), targets[max_i].item(), max_loss.item(), max_i.item()))
    return loss_value

def validation_caps(logits, targets, mask, category, pltname="err_in_val"):
    with torch.no_grad():
        # use the labels in validation set
        logits = logits[mask]
        targets = targets[mask]
        # print('logits shape:', logits.shape, 'targets shape:', targets.shape)
        err_vec = torch.abs((logits - targets) / targets).squeeze()
        # print("err_vec:", err_vec)
        mean_err =  torch.mean(err_vec, dim=0)
        max_err, max_i = torch.max(err_vec, dim=0)
        # print("max_err:", max_err)
        plot_errors(((logits - targets) / targets).squeeze(), targets, pltname, category)
        metrics = {"mean_err": mean_err.item(), "max_err": max_err.item()}

        return metrics

def evaluation_caps(h, model):
    with torch.no_grad():
        logits = model(h)
        return logits

def train_cap(dataset, net_h, model, cmask, category):
    net_h = torch.cat([net_h, dataset._n_feat], dim=1)
    # net_h = dataset._n_feat
    net_h = net_h[cmask]
    # we may train samples from multiple classes 
    targets = dataset.get_targets()[cmask]
    # define train/val samples, loss function and optimizer
    test_mask = dataset.get_test_mask()
    # here we test only one class 
    test_mask = test_mask[cmask]
    loss_fcn = ud_loss
    optimizer = torch.optim.Adam([{'params' : model.parameters()}], lr=5e-3, weight_decay=5e-4)
    best_val_loss = torch.inf
    bad_count = 0
    best_epoch = -1
    test_metrics = {}
    val_metrics = {}
    patience = 25
    # training loop
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        logits = model(net_h)
        val_mask, train_mask = dataset.get_val_mask(test_mask)
        # print('train#samples:', train_mask.sum().item(), 'val#samples:', val_mask.sum().item(), \
        #       'test#samples:', test_mask.sum().item())
        # print('max targets in training:', targets[train_mask].max().item())
        loss = loss_fcn(logits[train_mask], targets[train_mask])
        val_loss = loss.item()
        loss.backward()
        optimizer.step()

        # do validations and evaluations
        model.eval()

        metrics = validation_caps(logits, targets, val_mask, category,
                                  "err_in_val_"+str(category))
        print("|| Epoch {:05d} | Loss {:.4f} | mean error {:.2f}%  | max_error {:.2f}% ||"
              .format(epoch, loss.item(), metrics['mean_err']*100, metrics['max_err']*100))
        # for ealy stop with 20 patience
        if (best_val_loss > val_loss) and (epoch > 40):
            best_val_loss = val_loss
            best_epoch = epoch
            val_metrics = metrics
            bad_count = 0
            print('Testing...')
            logits = evaluation_caps(net_h, model)
            test_metrics = validation_caps(logits, targets, test_mask, category,
                                           "err_in_eval_"+str(category))
            print("|| test metrics | mean error {:.2f}%  | max error {:.2f}% ||"
                  .format(test_metrics['mean_err']*100, test_metrics['max_err']*100))
        elif epoch > 100:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break

    print("|* Category:{:d} | Best epoch:{:d} | loss: {:.4f} | # train samples: {:d} | # val samples: {:d} *|" \
          .format(category, best_epoch, best_val_loss, train_mask.sum(), val_mask.sum()))
    print("|* validation metrics | mean error {:.2f}% | max error {:.2f}% *|"
          .format(val_metrics['mean_err']*100, val_metrics['max_err']*100))
    print("|* test metrics | mean error {:.2f}% | max error {:.2f}% | #samples: {:d} *|"
          .format(test_metrics['mean_err']*100, test_metrics['max_err']*100, test_mask.sum()))
    return test_metrics

def generate_samples(h, feat, targets, labels):
        h = torch.cat([h, feat], dim=1)
        h_res, labels_res = SMOTE().fit_resample(h, labels)
        print(sorted(Counter(labels_res).items()))
        return h_res, labels_res


if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dataset = SRAMDataset(name='sandwich', raw_dir='/data1/shenshan/RCpred')
    # assert 0
    ### feature of instances transform ###
    proj_h_feat = 64
    proj_feat = 64
    gnn_h_feat = 128
    gnn_feat = 64
    out_feat = dataset._num_classes
    mlp_feats = [64, 64]
    linear_dict = {'device': [dataset._d_feat_dim, proj_h_feat, proj_feat], 
                   'inst':   [dataset._i_feat_dim, proj_h_feat, proj_feat], 
                   'net':    [dataset._n_feat_dim, proj_h_feat, proj_feat]}
    
    for i in range(1):
        """ projection layer model """ 
        model_list = nn.ModuleList()
        model_list.append(Hete2HomoLayer(linear_dict, act=nn.ReLU(), has_l2norm=False, has_bn=True, dropout=0.2).to(device))

        """ create GNN model """   
        model_list.append(GCN(proj_feat, gnn_feat, gnn_feat, dropout=0.2).to(device))
        # model_list.append(GAT(proj_feat, gnn_feat, gnn_feat, heads=[4, 4, 1], feat_drop=0.0, attn_drop=0.0).to(device))
        # model_list.append(GATv2(1, proj_feat, gnn_feat, gnn_feat, [4, 4, 1], nn.ReLU(), feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False))
        # model_list.append(GraphSAGE(proj_feat, gnn_h_feat, gnn_feat, 1, activation=nn.ReLU(), dropout=0.0, aggregator_type="mean"))
        
        """ MLP model """
        model_list.append(MLPN(gnn_feat, mlp_feats, out_feat, act=nn.ReLU(), use_bn=True, has_l2norm=False, dropout=0.2).to(device))
        # model_list.append(mlp_block(mlp_feat, 2, nonlinear=None, use_bn=True))
        
        """ model training """
        print('Training...')
        train(dataset, model_list)
        # torch.save(model_list, "data/models/"+name+".pt")
        assert 0
        # """ models saving """
        # name = '_'.join(model.__class__.__name__ for model in model_list)
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.96_f1_weighted_0.96_f1_macro_0.92_acc_0.98_f1_weighted_0.98_f1_macro_0.97_295'
        name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.98_f1_weighted_0.98_f1_macro_0.93_acc_0.92_f1_weighted_0.92_f1_macro_0.94_183'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.97_f1_weighted_0.97_f1_macro_0.86_acc_0.98_f1_weighted_0.98_f1_macro_0.84_238'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.93_f1_weighted_0.93_f1_macro_0.80_acc_0.92_f1_weighted_0.92_f1_macro_0.86_150'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_baseConf'
        # load classifier
        model_list_ld = torch.load("data/models/"+name+".pt")
        # print(model_list_ld)

        """ inference the whole nets """
        print('Inferencing whole nodes...')
        for i, model in enumerate(model_list_ld):
            model.eval()
        logits, net_h = evaluation(dataset._shg, dataset._bg, 
                                   dataset.get_feat_dict(), model_list_ld)
        mask = torch.ones(dataset._num_n, dtype=torch.bool)
        mask[dataset._zero_idx] = False
        name = '_'.join(model.__class__.__name__ for model in model_list)
        metrics = validation(logits, dataset.get_labels(), mask, name+"_conf_matrix_infr")
        print("|| inference metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(metrics['acc']*100, metrics['f1_weighted'], metrics['f1_macro']))
        # assert 0
        class_distr = torch.zeros(dataset._num_classes)
        for i in range(dataset._num_classes):
            class_distr[i] = (logits.argmax(dim=1).squeeze() == i).sum().long()
        print('predicted distr:', class_distr)
        # assert 0

        """ Define regression models """
        model_list_c = nn.ModuleList()
        for i in range(dataset._num_classes):
            mlp_feats = [64, 32]
            model_list_c.append(MLPN(dataset._n_feat_dim+gnn_feat, mlp_feats, 1, 
            # model_list_c.append(MLPN(dataset._n_feat_dim, mlp_feats, 1, 
                                     act=nn.ReLU(), use_bn=False, has_l2norm=False, 
                                     dropout=0.5).to(device))
        
        """ model training """
        mean_err = torch.zeros(dataset._num_classes)
        max_err = torch.zeros(dataset._num_classes)
        
        for i in range(dataset._num_classes):
            print('Training class {:d} ...'.format(i))
            model = model_list_c[i]
            class_mask = logits.argmax(dim=1).squeeze() == i
            test_metrics = train_cap(dataset, net_h, model, class_mask, i)
            mean_err[i] = test_metrics['mean_err']
            max_err[i] = test_metrics['max_err']

        print('mean_err:', mean_err, 'avg:', mean_err.mean())
        print('max_err:', max_err, 'avg:', max_err.mean())
        torch.save(model_list_c, "data/models/5_MLPs_2.pt")
        assert 0

        model_list_c = torch.load("data/models/5_MLPs_2.pt")
        print('load the mlp regression models')
        