import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from SRAM_dataset import SRAMDataset
from focal_loss_pytorch.focalloss import FocalLoss
import copy
from dgl.dataloading import DataLoader
# import dgl
# from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import os 
import matplotlib as mpl
from regression import validation_caps, ud_loss
from models import NetCapPredictor

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
def validation(logits, labels, mask=None, pltname="conf_matrix_valid"):
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
                   'f1_weighted': sklearn.metrics.f1_score(labels, indices, average='weighted'),
                   'f1_macro': sklearn.metrics.f1_score(labels, indices, average='macro'),
                #    'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                  }
        plot_confmat(labels, indices, metrics, pltname)
        return metrics

""" Just use in testing """
def evaluation(bg, h_dict, model_list: nn.ModuleList()):
    with torch.no_grad():
        trans_h = model_list[0](h_dict)#[input_nodes.long()]
        dst_h = model_list[1](bg, trans_h)
        net_h = dst_h[-h_dict['net'].shape[0]:]
        logits = model_list[2](net_h)
        return logits, net_h

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

def train(dataset: SRAMDataset, model, device):
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
    # print("weights:", weights)
    # print("weights shape:", weights.shape)
    # assert 0
    loss_fcn2 = ud_loss
    # loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=5e-4)
    max_epoch = 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch), 4e-4)

    best_val_loss = torch.inf
    best_test_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    best_val_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    best_loss_metrics = {}
    bad_count = 0
    best_count = 0
    patience = 25
    test_metrics = {}
    val_metrics = {}

    metric_log = {'train_loss':[], 'val_f1': [], 'val_acc':[], 
                  'test_epoch':[], 'test_f1':[], 'test_acc':[], 
                  'time': 0.0}
    print("Training " + model.name + " classifier at " + start.strftime("%d-%m-%Y_%H:%M:%S"))
    print('Entering training loop...')

    ### training loop ###
    for epoch in range(max_epoch):
        model.train()
    
        val_loss = 0
        optimizer.zero_grad()
        
        t, h = model(h_dict, bg)
        loss_c = loss_fcn1(t[train_nids], labels[train_nids].squeeze()) 
        loss_r = loss_fcn2(h[train_nids], targets[train_nids], weights=weights[train_nids])
        print("loss c:", loss_c.item(), "loss r:", loss_r.item())
        loss = loss_c + loss_r
        val_loss = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        ### do validations and evaluations ###
        model.eval()

        print('Validating...')
        logits, clogits = model(h_dict, bg)
        metrics = validation(logits, labels, mask=test_nids, pltname=model.name+"_conf_matrix_valid")
        metric_log['train_loss'].append(val_loss)
        metric_log['val_acc'].append(metrics['acc'])
        metric_log['val_f1'].append(metrics['f1_macro'])
        print("|| Epoch {:05d} | Loss {:.4f} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(epoch,     val_loss,     metrics['acc']*100, 
                      metrics['f1_weighted'],  metrics['f1_macro']))

        metrics = validation_caps(clogits, targets, 1, mask=test_nids, pltname="err_in_eval_"+str(1))
        print("|| train/test time {:s}/{:s} | mean error {:.2f}%  | max error {:.2f}% ||"
                .format(str(datetime.now()-start), str(datetime.now()-start), 
                        metrics['mean_err']*100, metrics['max_err']*100))
        
        # # for ealy stop with 25 patience
        # if ((best_val_loss > val_loss) or 
        #     (best_val_metrics['f1_macro'] < metrics['f1_macro'])) and (epoch > 50):
        #     if (best_val_loss > val_loss):
        #         best_loss = val_loss
        #         best_loss_epoch = epoch
        #         best_loss_metrics = metrics
        #     if best_val_metrics['f1_macro'] < metrics['f1_macro']:
        #         best_val_epoch = epoch
        #         best_val_loss = val_loss
        #         best_val_metrics = metrics
            
        #     bad_count = 0
        #     print('Testing...')
        #     test_start = datetime.now()
        #     logits, _ = evaluation(bg, h_dict, model_list)
        #     test_metrics = validation(logits, labels, 
        #                               mask=test_nids, pltname=name+"_conf_matrix_eval")
        #     metric_log['test_epoch'].append(epoch)
        #     metric_log['test_acc'].append(test_metrics['acc'])
        #     metric_log['test_f1'].append(metrics['f1_macro'])
        #     metric_log['time'] = (datetime.now() - start)
        #     print("|| train/test time {:s}/{:s} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
        #           .format(str(metric_log['time']), str(datetime.now()-test_start), 
        #                   test_metrics['acc']*100, test_metrics['f1_weighted'], test_metrics['f1_macro']))
            
        #     # keep track the best testing result
        #     if best_test_metrics['f1_macro'] < test_metrics['f1_macro']:
        #         best_test_metrics = test_metrics
        #         best_test_epoch = epoch
            
        #     if (metrics['f1_macro'] > 0.9 and test_metrics['f1_macro'] > 0.9 and
        #         metrics['acc'] > 0.9 and test_metrics['acc'] > 0.9):
        #         best_count += 1
        #         model_list_cp = copy.deepcopy(model_list)
        #         classifier_save(model_list, metrics, test_metrics, epoch)
        #     else:
        #         best_count = 0
        #     model_list_cp = copy.deepcopy(model_list)
        # # training this model at leat (200+patience) times
        # elif epoch > 200:
        #     bad_count += 1 
        
        # if bad_count > patience:
        #     print("Ending training ...")
        #     break

        # plot_metric_log(metric_log, name+'_'+start.strftime("%d-%m-%Y_%H:%M:%S"))
        # end training epoch
    
    # classifier_save(model_list_cp)
    # end_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    # print("|* Min loss epoch:{:d} | loss: {:.4f} | train hours: {:s} | end time:{:s} *|" \
    #       .format(best_loss_epoch, best_loss, str(metric_log['time']), end_date))
    # print("|* Best validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
    #       .format(best_loss_metrics['acc']*100, best_loss_metrics['f1_weighted'], best_loss_metrics['f1_macro']))
    # print("|* Best epoch:{:d} loss: {:.4f} *|" \
    #       .format(best_val_epoch, best_val_loss))
    # print("|* Best validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
    #       .format(best_val_metrics['acc']*100, best_val_metrics['f1_weighted'], best_val_metrics['f1_macro']))
    # print("|* Best test epoch:{:d} *|".format(best_test_epoch))
    # print("|* Best test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
    #       .format(best_test_metrics['acc']*100, best_test_metrics['f1_weighted'], best_test_metrics['f1_macro']))
    # print("|* Best epoch continue iterations: {:d} *|".format(best_count))
    return test_metrics

if __name__ == '__main__':
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dataset = SRAMDataset(name='sandwich', raw_dir='/data1/shenshan/RCpred')
    linear_dict = {'device': [dataset._d_feat_dim, 64, 64], 
                   'inst':   [dataset._i_feat_dim, 64, 64], 
                   'net':    [dataset._n_feat_dim, 64, 64]}
    model = NetCapPredictor(num_class=dataset._num_classes, proj_dim_dict=linear_dict, 
                            gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
                            device=device)
    train(dataset, model, device)