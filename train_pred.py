import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Hete2HomoLayer, GAT, MLP3, GCN, GraphSAGE
import seaborn as sns
import matplotlib.pyplot as plt
from SRAM_dataset import SRAMDataset, SRAMLargeCapDataset

""" User defined Loss function """
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

def plot_errors(x, y, pltname):
    plt.cla()
    plt.yscale("log")
    # kwargs = {'cumulative': True}
    # fig = sns.distplot(errors.tolist(), hist_kws=kwargs, kde_kws=kwargs)
    fig = plt.scatter(x.tolist(), y.tolist(), s=5, marker=".", alpha=0.5)
    plt.xlabel('relative errors')
    plt.ylabel('cap (fF)')
    plt.grid()
    absx = torch.abs(x)
    max_err, _ = torch.max(absx, dim=0)
    freq5 = torch.sum((absx < 0.05).int()) / absx.shape[0]
    freq10 = torch.sum((absx < 0.1).int()) / absx.shape[0]
    freq20 = torch.sum((absx < 0.2).int()) / absx.shape[0]
    freq50 = torch.sum((absx < 0.5).int()) / absx.shape[0]
    plt.title('mean error:{:.2f}%, max error:{:.2f}%, 95% quantile:{:.2f}%\n \
               0.05 fraction:{:.2f}%, 0.1 fraction:{:.2f}%,\n \
               0.2 fraction:{:.2f}%, 0.5 fraction:{:.2f}%' \
              .format(torch.mean(absx).item()*100, max_err.item()*100, \
                      torch.quantile(x, 0.95).item()*100, freq5.item()*100, \
                      freq10.item()*100, freq20.item()*100, freq50.item()*100))
    fig.get_figure().savefig('./data/'+pltname, dpi=400)

""" quick validation in trainning epoch """
def validation(logits, targets, mask, pltname="err_in_val"):
    # use the labels in validation set
    logits = logits[mask]
    targets = targets[mask]
    err_vec = torch.abs((logits - targets) / targets).squeeze()
    mean_err =  torch.mean(err_vec, dim=0)
    # print('logits with inf:', torch.nonzero((logits == torch.inf)))
    max_err, max_i = torch.max(err_vec, dim=0)
    print('err_vec:', err_vec)
    print('in validation with max err: logits:{:.4f}, targets:{:.4f}, error:{:.2f}%, with index:{:05d}'
          .format(logits[max_i].item(), targets[max_i].item(), max_err.item()*100, max_i.item()))
    err_vec = ((logits - targets) / targets).squeeze()
    plot_errors(err_vec, targets, pltname)
    return mean_err.item(), max_err.item()

""" Just use in testing """
def evaluation(shg, bg, h_dict, targets, mask, 
               model_list: nn.ModuleList(), pltname="err_in_eval"):
    with torch.no_grad():
        for i, model in enumerate(model_list):
            model.eval()
        src_h_dict = {'inst': h_dict['inst']}
        trans_h = model_list[0](shg, h_dict)
        dst_h = model_list[1](bg, trans_h)
        # we only consider the dst nodes' labels
        dst_h = dst_h[-targets.shape[0]:]
        # new_h = torch.cat([dst_h, h_dict['node']], dim=1)
        logits = model_list[2](dst_h)
    return validation(logits, targets, mask, pltname)

def train(dataset: SRAMDataset, model_list: nn.ModuleList()):
    shg = dataset._shg
    bg = dataset._bg
    h_dict = dataset.get_feat_dict()
    labels = dataset.get_labels()
    masks = dataset.get_test_mask()
    src_h_dict = {'inst': h_dict['inst']}
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    loss_fcn = ud_loss
    optimizer = torch.optim.Adam([
                                    {'params' : model_list[0].parameters()},
                                    {'params' : model_list[1].parameters()}, 
                                    {'params' : model_list[2].parameters()}
                                 ], lr=1e-3, weight_decay=5e-4)
    
    best_val_loss = torch.inf
    bad_count = 0
    best_epoch = -1
    test_err_max = torch.inf
    test_err = torch.inf
    val_err_max = torch.inf
    val_err = torch.inf
    patience = 50
    # training loop
    for epoch in range(500):
        for i, model in enumerate(model_list):
            model.train()
        
        trans_h = model_list[0](shg, h_dict)
        dst_h = model_list[1](bg, trans_h)
        # we only consider the dst nodes' labels
        dst_h = dst_h[dataset._num_src : ]
        #new_h = torch.cat([dst_h, h_dict['node']], dim=1)
        logits = model_list[2](dst_h)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        name = model_list[1].__class__.__name__
        err, max_err = validation(logits, labels, val_mask, name+"_err_in_val")
        print("|| Epoch {:05d} | Loss {:.4f} | mean error {:.2f}% | max error {:.2f}% ||"
              . format(epoch, loss.item(), err*100, max_err*100))
        # for ealy stop with 20 patience
        val_loss = loss.item()
        if (best_val_loss > val_loss) and (epoch > 100):
            best_val_loss = val_loss
            best_epoch = epoch
            val_err = err
            val_err_max = max_err
            bad_count = 0
            print('Testing...')
            test_err, test_err_max = evaluation(shg, bg, h_dict, labels, 
                                                test_mask, model_list, 
                                                name+"_err_in_eval")
            print("|| Test error: {:.2f}%, max error: {:.2f}% ||".format(test_err*100, test_err_max*100))
        elif epoch > 100:
            bad_count += 1 
        
        if bad_count > patience:
            print("Ending training ...")
            break

    print("|| Best epoch:{:d} loss: {:.4f}, Validate error:{:.2f}%, max:{:.2f}%, Test error: {:.2f}%, max: {:.2f}% ||" \
          .format(best_epoch, best_val_loss, val_err*100, val_err_max*100, test_err*100, test_err_max*100))
    return test_err, test_err_max

if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # dataset = SRAMDataset()
    dataset = SRAMLargeCapDataset()
    # assert(0)
    shg = dataset._shg
    bg = dataset._bg
    """ feature of instances transform """
    in_feat = shg.nodes['node'].data['x'].shape[1]
    proj_feat = 64
    gnn_feat = 64
    mlp_feat = 64
    out_feat = 1
    linear_dict = {'inst': [in_feat, proj_feat, proj_feat], 'node': [in_feat, proj_feat, proj_feat]}
    test_err = torch.zeros([10])
    test_err_max = torch.zeros([10])
    for i in range(1):
        """ projection layer model """ 
        model_list = nn.ModuleList()
        model_list.append(Hete2HomoLayer(linear_dict, act=nn.ReLU(), has_l2norm=False, has_bn=True).to(device))

        """ create GNN model """   
        model_list.append(GCN(proj_feat, gnn_feat, gnn_feat).to(device))
        # model_list.append(GAT(proj_feat, gnn_feat, gnn_feat, heads=[4,1], feat_drop=0.6, attn_drop=0.6).to(device))
        # model_list.append(GraphSAGE(proj_feat, gnn_feat, gnn_feat, 2, activation=nn.ReLU(), dropout=0.5, aggregator_type="mean"))
        
        """ MLP model """
        mlp_feats = [64, 64]
        model_list.append(MLP3(gnn_feat, mlp_feats, out_feat, nonlinear='relu', use_bn=True).to(device))
        
        """ model training """
        print('Training...')
        err, err_max = train(dataset, model_list)
        test_err[i] = err
        test_err_max[i] = err_max

    print("|| Final test error: {:.2f}%, max: {:.2f}% ||" \
          .format(torch.mean(test_err[:i+1]).item()*100, torch.mean(test_err_max[:i+1]).item()*100))
    print('errors in each training:', test_err[:i+1], 'max errors', test_err_max[:i+1])