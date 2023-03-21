import torch
import torch.nn as nn
from models import Hete2HomoLayer, GAT, GATv2, MLP3, GCN, GraphSAGE, MLPN
from SRAM_dataset import SRAMDataset
from regression import train_cap
from classification import train, validation, evaluation

if __name__ == '__main__':
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dataset = SRAMDataset(name='sandwich', raw_dir='/data1/shenshan/RCpred')
    # assert 0
    ### feature of instances transform ###
    proj_h_feat = 64
    proj_feat = 64
    gnn_h_feat = 64
    gnn_feat = 64
    out_feat = dataset._num_classes
    mlp_feats = [64, 64]
    linear_dict = {'device': [dataset._d_feat_dim, proj_h_feat, proj_feat], 
                   'inst':   [dataset._i_feat_dim, proj_h_feat, proj_feat], 
                   'net':    [dataset._n_feat_dim, proj_h_feat, proj_feat]}
    
    for i in range(1):
        """ projection layer model """ 
        model_list = nn.ModuleList()
        model_list.append(Hete2HomoLayer(linear_dict, act=nn.ReLU(), has_bn=True, has_l2norm=False, dropout=0.1).to(device))

        """ create GNN model """   
        # model_list.append(GCN(proj_feat, gnn_h_feat, gnn_feat, dropout=0.1).to(device))
        # model_list.append(GAT(proj_feat, gnn_feat, gnn_feat, heads=[4, 4, 1], feat_drop=0.0, attn_drop=0.0).to(device))
        # model_list.append(GATv2(1, proj_feat, gnn_feat, gnn_feat, [4, 4, 1], nn.ReLU(), feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False).to(device))
        model_list.append(GraphSAGE(proj_feat, gnn_h_feat, gnn_feat, 1, activation=nn.ReLU(), dropout=0.1, aggregator_type="mean").to(device))
        
        """ MLP model """
        model_list.append(MLPN(gnn_feat, mlp_feats, out_feat, act=nn.ReLU(), use_bn=True, has_l2norm=False, dropout=0.1).to(device))

        """ model training """
        train(dataset, model_list, device)
        # torch.save(model_list, "data/models/"+name+".pt")
        assert 0
        # """ models saving """
        # name = '_'.join(model.__class__.__name__ for model in model_list)
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.96_f1_weighted_0.96_f1_macro_0.92_acc_0.98_f1_weighted_0.98_f1_macro_0.97_295'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.98_f1_weighted_0.98_f1_macro_0.93_acc_0.92_f1_weighted_0.92_f1_macro_0.94_183'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.97_f1_weighted_0.97_f1_macro_0.86_acc_0.98_f1_weighted_0.98_f1_macro_0.84_238'
        # name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.93_f1_weighted_0.93_f1_macro_0.80_acc_0.92_f1_weighted_0.92_f1_macro_0.86_150'
        # name = 'Hete2HomoLayer_GCN_MLPN_acc_0.90_f1_weighted_0.90_f1_macro_0.88_acc_0.90_f1_weighted_0.90_f1_macro_0.89_245'
        # name = 'Hete2HomoLayer_GCN_MLPN_acc_0.96_f1_weighted_0.96_f1_macro_0.90_acc_0.97_f1_weighted_0.97_f1_macro_0.93_277'
        name = 'Hete2HomoLayer_GCN_MLPN_acc_0.98_f1_weighted_0.98_f1_macro_0.94_acc_0.95_f1_weighted_0.95_f1_macro_0.93_158'
        # load classifier
        model_list_ld = torch.load("data/models/"+name+".pt")
        # print(model_list_ld)

        """ inference the whole nets """
        print('Inferencing whole nodes...')
        for i, model in enumerate(model_list_ld):
            model.eval()
        logits, net_h = evaluation(dataset._bg, dataset._num_n,
                                   dataset.get_feat_dict(), model_list_ld)
        mask = torch.ones(dataset._num_n, dtype=torch.bool)
        mask[dataset._zero_idx] = False
        name = '_'.join(model.__class__.__name__ for model in model_list_ld)
        metrics = validation(logits, dataset.get_labels(), mask, name+"_conf_matrix_infr")
        print("|| inference metrics | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(metrics['acc']*100, metrics['f1_weighted'], metrics['f1_macro']))
        # do statistics for the sample distribution
        class_distr = torch.zeros(dataset._num_classes)
        for i in range(dataset._num_classes):
            class_distr[i] = (logits.argmax(dim=1).squeeze() == i).sum().long()
        print('predicted distr:', class_distr)

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
        # class_merge = [[0,1], [0,1], [1,2], [2,3], [3,4]]
        for i in range(dataset._num_classes):
            print('Training class {:d} ...'.format(i))
            model = model_list_c[i]
            class_mask_train = torch.zeros(net_h.shape[0], dtype=torch.bool)
            # for class_idx in class_merge[i]:
            class_mask_train |= logits.argmax(dim=1).squeeze() == i
            class_mask_test = logits.argmax(dim=1).squeeze() == i
            test_metrics = train_cap(dataset, net_h, model, class_mask_train, class_mask_test, i)
            mean_err[i] = test_metrics['mean_err']
            max_err[i] = test_metrics['max_err']

        print('mean_err:', mean_err, 'avg:', mean_err.mean())
        print('max_err:', max_err, 'avg:', max_err.mean())
        torch.save(model_list_c, "data/models/5_MLPs_gcn2.pt")
        assert 0

        model_list_c = torch.load("data/models/5_MLPs_2.pt")
        print('load the mlp regression models')
        