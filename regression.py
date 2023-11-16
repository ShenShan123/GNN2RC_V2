import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from SRAM_dataset import SRAMDataset
from datetime import datetime

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

def ud_loss(logits, targets, weights=None):
    mask = (targets != torch.inf) & (logits != torch.inf) &  (targets != 0.0)
    logits = logits[mask].squeeze()
    targets = targets[mask].squeeze()
    diffs = logits - targets
    rel_w = torch.div(diffs, targets)
    if weights is None:
        squ_rel_w = torch.square(rel_w)
    else:
        squ_rel_w = torch.square(rel_w)* weights[mask].squeeze()
    loss_value = torch.mean(squ_rel_w)
    max_loss, max_i = torch.max(loss_value, dim=0)
    # print('in loss with max loss: logits:{:.4f}, targets:{:.4f}, loss:{:.4f}, with index:{:05d}'
    #       .format(logits[max_i].item(), targets[max_i].item(), max_loss.item(), max_i.item()))
    return loss_value

def validation_caps(logits, targets, category, mask=None, pltname="err_in_val"):
    with torch.no_grad():
        if mask is not None:
            # use the labels in validation set
            logits = logits[mask].squeeze()
            targets = targets[mask].squeeze()
        # print('logits shape:', logits.shape, 'targets shape:', targets.shape)
        err_vec = torch.abs((logits - targets) / targets).squeeze()
        # print("err_vec:", err_vec)
        mean_err =  torch.mean(err_vec, dim=0)
        max_err, max_i = torch.max(err_vec, dim=0)
        # print("max_err:", max_err)
        # plot_errors(((logits - targets) / targets).squeeze(), targets, pltname, category)
        metrics = {"mean_err": mean_err.item(), "max_err": max_err.item()}
        return metrics

def evaluation_caps(h, model):
    with torch.no_grad():
        logits = model(h)
        return logits

def train_cap(dataset: SRAMDataset, net_h, model: nn.Module(), cmask, category):
    start = datetime.now()
    train_mask, val_mask = dataset.get_masks()
    if net_h is not None:
        net_h = torch.cat([net_h, dataset._n_feat], dim=1)
    else:
        net_h = dataset._n_feat
    # net_h = dataset._n_feat
    net_h_test = net_h[val_mask & cmask]
    net_h = net_h[train_mask & cmask]
    # we may train samples from multiple classes 
    targets = dataset.get_targets()[(train_mask) & cmask]
    targets_test = dataset.get_targets()[val_mask & cmask]
    # define train/val samples, loss function and optimizer
    # test_mask = dataset.get_test_mask()
    # here we test only one class 
    # test_mask = test_mask[cmask_train]
    # test_mask = test_mask[cmask_test]
    loss_fcn = ud_loss
    optimizer = torch.optim.Adam([{'params' : model.parameters()}], lr=1e-2, weight_decay=5e-4)
    best_val_loss = torch.inf
    bad_count = 0
    best_epoch = -1
    test_metrics = {}
    val_metrics = {}
    patience = 25
    # training loop
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        logits = model(net_h)
        # loss = loss_fcn(logits[train_mask], targets[train_mask])
        loss = loss_fcn(logits, targets)
        val_loss = loss.item()
        loss.backward()
        optimizer.step()

        # do validations and evaluations
        model.eval()

        metrics = validation_caps(logits, targets, category, mask=None, 
                                  pltname="err_in_val_"+str(category))
        print("|| Epoch {:05d} | Loss {:.4f} | mean error {:.2f}%  | max_error {:.2f}% ||"
              .format(epoch, loss.item(), metrics['mean_err']*100, metrics['max_err']*100))
        # for ealy stop with 20 patience
        if (best_val_loss > val_loss) and (epoch > 40):
            best_val_loss = val_loss
            best_epoch = epoch
            val_metrics = metrics
            bad_count = 0
            test_start = datetime.now()
            print('Testing...')
            logits = evaluation_caps(net_h_test, model)
            test_metrics = validation_caps(logits, targets_test, category, mask=None,
                                           pltname="err_in_eval_"+str(category))
            print("|| train/test time {:s}/{:s} | mean error {:.2f}%  | max error {:.2f}% ||"
                  .format(str(datetime.now()-start), str(datetime.now()-test_start), 
                          test_metrics['mean_err']*100, test_metrics['max_err']*100))
        elif epoch > 100:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break

    print("|* Category:{:d} | Best epoch:{:d} | loss: {:.4f} | # train samples: {:d} | # val samples: {:d} *|" \
          .format(category, best_epoch, best_val_loss, (train_mask & cmask).sum(), (val_mask & cmask).sum()))
    print("|* validation metrics | mean error {:.2f}% | max error {:.2f}% *|"
          .format(val_metrics['mean_err']*100, val_metrics['max_err']*100))
    print("|* test metrics | mean error {:.2f}% | max error {:.2f}% | #samples: {:d} *|"
          .format(test_metrics['mean_err']*100, test_metrics['max_err']*100, (val_mask & cmask).sum()))
    return test_metrics