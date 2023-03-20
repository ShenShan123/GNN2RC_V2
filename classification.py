import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from SRAM_dataset import SRAMDataset
from focal_loss_pytorch.focalloss import FocalLoss
import copy
from dgl.dataloading import DataLoader
import dgl
from tqdm import tqdm
# import torch.nn.functional as F
import time
from datetime import datetime

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
def validation(logits, targets, mask=None, pltname="conf_matrix_valid"):
    with torch.no_grad():
        # use the labels in validation set
        if mask is not None:
            logits = logits[mask]
        # print("logits in val")
        # print(logits)
        # logits_norm = F.normalize(logits, p=1, dim=1).cpu().detach().numpy()
        # print('logits_norm:', logits_norm)
            targets = targets[mask].cpu().detach().numpy()
        else:
            targets = targets.cpu().detach().numpy()

        _, indices = torch.max(logits, dim=1)
        metrics = {'acc': sklearn.metrics.accuracy_score(targets, indices),
                   'f1_weighted': sklearn.metrics.f1_score(targets, indices, average='weighted'),
                   'f1_macro': sklearn.metrics.f1_score(targets, indices, average='macro'),
                #    'auc': 0, #sklearn.metrics.roc_auc_score(targets, logits_norm, multi_class='ovr'),
                  }
        plot_confmat(targets, indices, metrics, pltname)
        return metrics

""" Just use in testing """
def evaluation(dataset, dataloader, h_dict, model_list: nn.ModuleList(), device):
    with torch.no_grad():
        id_offset = dataset._num_d + dataset._num_i
        logits = []
        net_h = []
        for input_nodes, output_nodes, blocks in dataloader:
            d_ids = input_nodes[input_nodes < dataset._num_d].long()
            i_ids = input_nodes[(input_nodes >= dataset._num_d) & (input_nodes < id_offset)].long() - dataset._num_d
            n_ids = input_nodes[input_nodes >= id_offset].long() - id_offset
            block_h_dict = {'device': h_dict['device'][d_ids].to(device), 'inst': h_dict['inst'][i_ids].to(device),
                            'net': h_dict['net'][n_ids].to(device)}
            trans_h = model_list[0](block_h_dict)#[input_nodes.long()]
            dst_h = model_list[1](blocks, trans_h)
            block_logits = model_list[2](dst_h)
            # block_nids = (output_nodes - id_offset).long()
            logits.append(block_logits.cpu().detach())
            net_h.append(dst_h.cpu().detach())
            # print("model parameters in eval:")
            # print_params(model_list)
            # print("logits in eval:", logits[mask])
        return torch.cat(logits, dim=0), torch.cat(net_h, dim=0)
    # return validation(logits, targets, mask, pltname)

""" Only be used in debugging """
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
        torch.save(model_list, "data/models/"+name+".pt")

def plot_metric_log(metric_log: dict, date, name):
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
    plt.title(name+" training log", loc='left')
    plt.legend()
    plt.savefig("data/plots/train_test_epochs_" + date + ".png", dpi=500, bbox_inches='tight')
    plt.close(fig)

def train(dataset: SRAMDataset, model_list: nn.ModuleList(), device):
    start_time = time.clock()
    start_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    bg = dataset._bg.to(device)
    h_dict = dataset.get_feat_dict()
    # h_dict = {key: value.to(device) for key, value in h_dict.items()}
    labels = dataset.get_labels().to(device)
    # test_mask = dataset.get_test_mask()
    # get train/validation split
    # train_mask, val_mask, test_mask = dataset.get_masks()

    train_nids, val_nids, test_nids = dataset.get_nids()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    id_offset = dataset._num_d + dataset._num_i
    train_nids = (train_nids + id_offset).to(device)
    val_nids = (val_nids + id_offset).to(device)
    test_nids = (test_nids + id_offset).to(device)
    trainDataloader = DataLoader(bg, train_nids, sampler, batch_size=1024, shuffle=False, 
                            drop_last=False, device=device, num_workers=0)#, use_uva=True)
    valDataloader = DataLoader(bg, val_nids, sampler, batch_size=1024, shuffle=False, 
                            drop_last=False, device=device, num_workers=0)#, use_uva=True)
    testDataloader = DataLoader(bg, test_nids, sampler, batch_size=1024, shuffle=False, 
                            drop_last=False, device=device, num_workers=0)#, use_uva=True)
    
    ### define train/val samples, loss function and optimizer ###
    loss_fcn = FocalLoss(gamma=2, alpha=dataset.alpha)
    # loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam([{'params' : model.parameters()} for model in model_list], 
                                 lr=1e-2, weight_decay=5e-4)
    max_epoch = 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(max_epoch), 1e-5)

    best_val_loss = torch.inf
    best_test_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    best_val_metrics = {'acc': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
    best_loss_metrics = {}
    bad_count = 0
    best_count = 0
    patience = 25
    model_list_cp = nn.ModuleList()
    name = model_list[1].__class__.__name__
    metric_log = {'train_loss':[], 'val_f1': [], 'val_acc':[], 
                  'test_epoch':[], 'test_f1':[], 'test_acc':[], 
                  'time': 0.0}

    print('  finish dataloader preparing, entering training loop...')

    ### training loop ###
    for epoch in range(max_epoch):
        for i, model in enumerate(model_list):
            model.train()
        
        # train_mask = train_mask.to(device)
        val_loss = 0
        logits = []

        ### inner loop: train batches ###
        for input_nodes, output_nodes, blocks  in trainDataloader:
            # print("blocks:", blocks)
            # print("input_nodes:", input_nodes)
            # print("output_nodes:", output_nodes)
        
            optimizer.zero_grad()
            d_ids = input_nodes[input_nodes < dataset._num_d].long()
            i_ids = input_nodes[(input_nodes >= dataset._num_d) & (input_nodes < id_offset)].long() - dataset._num_d
            n_ids = input_nodes[input_nodes >= id_offset].long() - id_offset
            block_h_dict = {'device': h_dict['device'][d_ids].to(device), 'inst': h_dict['inst'][i_ids].to(device),
                            'net': h_dict['net'][n_ids].to(device)}
            
            trans_h = model_list[0](block_h_dict)#[input_nodes.long()]
            dst_h = model_list[1](blocks, trans_h)
            # print('dst_h shape:', dst_h.shape)
            # print('output_nodes shape:', output_nodes.shape)
            block_logits = model_list[2](dst_h)

            block_nids = (output_nodes - id_offset).long()
            logits.append(block_logits.cpu().detach())
            # block_train_mask = train_mask[block_nids]
            block_labels = labels[block_nids]#.to(device)
            loss = loss_fcn(block_logits, block_labels.squeeze())
            val_loss += loss.item()
            loss.backward()
            optimizer.step()
            # end for
        scheduler.step()
        logits = torch.cat(logits, dim=0)
        
        ### do validations and evaluations ###
        for i, model in enumerate(model_list):
            model.eval()

        print('Validating...')
        logits, _ = evaluation(dataset, valDataloader, h_dict, model_list, device)
        metrics = validation(logits, labels[(val_nids - id_offset).long()], 
                            #  val_mask[(train_nids - id_offset).long()], 
                             mask=None, pltname=name+"_conf_matrix_valid")
        metric_log['train_loss'].append(val_loss)
        metric_log['val_acc'].append(metrics['acc'])
        metric_log['val_f1'].append(metrics['f1_macro'])
        print("|| Epoch {:05d} | Loss {:.4f} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(epoch,     val_loss,     metrics['acc']*100, 
                      metrics['f1_weighted'],  metrics['f1_macro']))
        
        # for ealy stop with 25 patience
        if ((best_val_loss > val_loss) or 
            (best_val_metrics['f1_macro'] < metrics['f1_macro'])) and (epoch > 50):
            if (best_val_loss > val_loss):
                best_loss = val_loss
                best_loss_epoch = epoch
                best_loss_metrics = metrics
            if best_val_metrics['f1_macro'] < metrics['f1_macro']:
                best_val_epoch = epoch
                best_val_loss = val_loss
                best_val_metrics = metrics
            
            bad_count = 0
            print('Testing...')
            logits, _ = evaluation(dataset, testDataloader, h_dict, model_list, device)
            test_metrics = validation(logits, labels[(test_nids - id_offset).long()], 
                                      mask=None, pltname=name+"_conf_matrix_eval")
            metric_log['test_epoch'].append(epoch)
            metric_log['test_acc'].append(test_metrics['acc'])
            metric_log['test_f1'].append(metrics['f1_macro'])
            metric_log['time'] = (time.clock() - start_time) / 3600
            print("|| training time {:.2f}h | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
                  .format(metric_log['time'], test_metrics['acc']*100, 
                          test_metrics['f1_weighted'], test_metrics['f1_macro']))
            
            # keep track the best testing result
            if best_test_metrics['f1_macro'] < test_metrics['f1_macro']:
                best_test_metrics = test_metrics
                best_test_epoch = epoch
            
            if (metrics['f1_macro'] > 0.9 and test_metrics['f1_macro'] > 0.9 and
                metrics['acc'] > 0.9 and test_metrics['acc'] > 0.9):
                best_count += 1
                model_list_cp = copy.deepcopy(model_list)
                classifier_save(model_list, metrics, test_metrics, epoch)
            else:
                best_count = 0
            model_list_cp = copy.deepcopy(model_list)
        # training this model at leat (200+patience) times
        elif epoch > 200:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break

        plot_metric_log(metric_log, start_date, name)
        # end training epoch
    
    classifier_save(model_list_cp)
    print("|* Min loss epoch:{:d} | loss: {:.4f} | train hours: {:.2f} *|" \
          .format(best_loss_epoch, best_loss, metric_log['time']))
    print("|* Best validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          .format(best_loss_metrics['acc']*100, best_loss_metrics['f1_weighted'], best_loss_metrics['f1_macro']))
    print("|* Best epoch:{:d} loss: {:.4f} *|" \
          .format(best_val_epoch, best_val_loss))
    print("|* Best validation metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          .format(best_val_metrics['acc']*100, best_val_metrics['f1_weighted'], best_val_metrics['f1_macro']))
    print("|* Best test epoch:{:d} *|".format(best_test_epoch))
    print("|* Best test metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} *|"
          .format(best_test_metrics['acc']*100, best_test_metrics['f1_weighted'], best_test_metrics['f1_macro']))
    print("|* Best epoch continue iterations: {:d} *|".format(best_count))
    return test_metrics