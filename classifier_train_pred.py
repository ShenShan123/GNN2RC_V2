import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Hete2HomoLayer, GAT, GATv2, MLP3, GCN, GraphSAGE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics
from SRAM_dataset import SRAMDataset
from focal_loss_pytorch.focalloss import FocalLoss
# from focal_loss import FocalLoss

def plot_errors(y_true, y_pred, pltname):
    # print('y_true:', y_true)
    # print('y_pred:', y_pred)
    cf_matrix = sklearn.metrics.confusion_matrix(y_true.squeeze(), y_pred, normalize='true')
    # print(cf_matrix)
    fig, ax = plt.subplots()
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    sns.heatmap(cf_matrix, annot=True, fmt='.2%', cbar=True)#,  cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.savefig('./data/'+pltname, dpi=400)
    plt.close(fig)

""" quick validation in trainning epoch """
def validation(logits, targets, mask, pltname="err_in_val"):
    with torch.no_grad():
        # use the labels in validation set
        logits = logits[mask]
        # print("logits in val")
        # print(logits)
        # logits_norm = F.normalize(logits, p=1, dim=1).cpu().detach().numpy()
        # print('logits_norm:', logits_norm)
        targets = targets[mask].cpu().detach().numpy()
        _, indices = torch.max(logits, dim=1)

        plot_errors(targets, indices, pltname)
        metrics = {'acc': sklearn.metrics.accuracy_score(targets, indices),
                'f1_weighted': sklearn.metrics.f1_score(targets, indices, average='weighted'),
                'f1_macro': sklearn.metrics.f1_score(targets, indices, average='macro'),
                'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                }
        return metrics

""" Just use in testing """
def evaluation(shg, bg, h_dict, targets, mask, 
               model_list: nn.ModuleList(), pltname="err_in_eval"):
    with torch.no_grad():
        trans_h = model_list[0](shg, h_dict)
        dst_h = model_list[1](bg, trans_h)
        # we only consider the dst nodes' labels
        dst_h = dst_h[-targets.shape[0]:]
        logits = model_list[2](dst_h)
        # print("model parameters in eval:")
        # print_params(model_list)
        # print("logits in eval:", logits[mask])
    return validation(logits, targets, mask, pltname)

def print_params(model_list: nn.ModuleList()):
    # for model in model_list:
    for name, params in model_list[0].named_parameters():
        # if name == "layers.1.bias":
        #     print(name, ":", params[:10])
        # if name == "layers.1.fc_self.weight":
        #     print(name, ":", params[0][:10])
        # if name == "layers.1.fc_neigh.weight":
        print(name, ":", params[0])

def train(dataset: SRAMDataset, model_list: nn.ModuleList()):
    shg = dataset._shg
    bg = dataset._bg
    h_dict = dataset.get_feat_dict()
    # labels = dataset.get_labels()
    labels = dataset._classes
    masks = dataset.get_test_mask()
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    # val_mask, train_mask = dataset.get_val_mask()
    loss_fcn = FocalLoss(gamma=2, alpha=dataset.alpha)
    optimizer = torch.optim.Adam([
                                    {'params' : model_list[0].parameters()},
                                    {'params' : model_list[1].parameters()}, 
                                    {'params' : model_list[2].parameters()}
                                 ], lr=5e-3, weight_decay=5e-4)
    
    best_val_loss = torch.inf
    bad_count = 0
    best_epoch = -1
    test_metrics = {}
    val_metrics = {}
    patience = 20
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
        #new_h = torch.cat([dst_h, h_dict['node']], dim=1)
        logits = model_list[2](dst_h)
        
        val_mask, train_mask = dataset.get_val_mask()
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask].squeeze())
        loss = loss_fcn(logits[train_mask], labels[train_mask].squeeze())
        
        loss.backward()
        optimizer.step()

        name = model_list[1].__class__.__name__
        
        # do validations and evaluations
        for i, model in enumerate(model_list):
            model.eval()

        metrics = validation(logits, labels, val_mask, name+"_conf_matrix_valid")
        print("|| Epoch {:05d} | Loss {:.4f} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              . format(epoch, loss.item(), metrics['acc']*100, metrics['f1_weighted'], metrics['f1_macro']))
        
        # for ealy stop with 20 patience
        val_loss = loss.item()
        if (best_val_loss > val_loss) and (epoch > 50):
            best_val_loss = val_loss
            best_epoch = epoch
            val_metrics = metrics
            bad_count = 0
            print('Testing...')
            test_metrics = evaluation(shg, bg, h_dict, labels, 
                                      test_mask, model_list, 
                                      name+"_conf_matrix_eval")
            # test_metrics = validation(logits, labels, test_mask, name+"_conf_matrix_eval")
            print("|| test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
                  .format(test_metrics['acc']*100, test_metrics['f1_weighted'], test_metrics['f1_macro']))
        elif epoch > 100:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break

    print("|* Best epoch:{:d} loss: {:.4f} *|" \
          .format(best_epoch, best_val_loss))
    print("|* validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          . format(val_metrics['acc']*100, val_metrics['f1_weighted'], val_metrics['f1_macro']))
    print("|* test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          . format(test_metrics['acc']*100, test_metrics['f1_weighted'], test_metrics['f1_macro']))
    return test_metrics

if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dataset = SRAMDataset(name='sandwich', raw_dir='/data1/shenshan/RCpred')
    shg = dataset._shg
    bg = dataset._bg
    # assert 0
    ### feature of instances transform ###
    proj_feat = 64
    gnn_feat = 64
    mlp_feat = 64
    out_feat = dataset._num_classes
    linear_dict = {'device': [dataset._d_feat_dim, proj_feat, proj_feat], 
                   'inst':   [dataset._i_feat_dim, proj_feat, proj_feat], 
                   'net':    [dataset._n_feat_dim, proj_feat, proj_feat]}
    
    for i in range(1):
        """ projection layer model """ 
        model_list = nn.ModuleList()
        model_list.append(Hete2HomoLayer(linear_dict, act=nn.ReLU(), has_l2norm=True, has_bn=True).to(device))

        """ create GNN model """   
        # model_list.append(GCN(proj_feat, gnn_feat, gnn_feat, dropout=0.1).to(device))
        # model_list.append(GAT(proj_feat, gnn_feat, gnn_feat, heads=[4, 4, 1], feat_drop=0.1, attn_drop=0.1).to(device))
        # model_list.append(GATv2(1, proj_feat, gnn_feat, gnn_feat, [4, 4, 1], nn.ReLU(), feat_drop=0.1, attn_drop=0.1, negative_slope=0.2, residual=False))
        model_list.append(GraphSAGE(proj_feat, gnn_feat, gnn_feat, 1, activation=nn.ReLU(), dropout=0.1, aggregator_type="gcn"))
        
        """ MLP model """
        mlp_feats = [64, 64]
        model_list.append(MLP3(gnn_feat, mlp_feats, out_feat, nonlinear='relu', use_bn=True).to(device))
        # model_list.append(mlp_block(mlp_feat, 2, nonlinear=None, use_bn=True))
        
        """ model training """
        print('Training...')
        train(dataset, model_list)