import torch
import torch.nn as nn
from models import Hete2HomoLayer, GAT, GCN, GraphSAGE, MLPN
from SRAM_dataset import SRAMDataset
from regression import train_cap
from classification import train, validation, evaluation
from datetime import datetime
import os
# from utils.circuits2graph import run_struct2g


if __name__ == '__main__':
    device = torch.device('cpu')
    dataset = SRAMDataset(name='ssram', raw_dir='/data1/shenshan/SPF_examples_cdlspf/Python_data/')

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
        model_list.append(Hete2HomoLayer(linear_dict, act=nn.ReLU(), has_bn=False, has_l2norm=True, dropout=0.1).to(device))

        """ create GNN model """   
        # model_list.append(GCN(proj_feat, gnn_h_feat, gnn_feat, dropout=0.1).to(device))
        model_list.append(GAT(proj_feat, gnn_feat, gnn_feat, heads=[1, 1, 1], feat_drop=0.1, attn_drop=0.1).to(device))
        # model_list.append(GATv2(1, proj_feat, gnn_feat, gnn_feat, [4, 4, 1], nn.ReLU(), feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False).to(device))
        # model_list.append(GraphSAGE(proj_feat, gnn_h_feat, gnn_feat, 1, activation=nn.ReLU(), dropout=0.1, aggregator_type="pool").to(device))
        
        """ MLP model """
        model_list.append(MLPN(gnn_feat, mlp_feats, out_feat, act=nn.ReLU(), use_bn=False, has_l2norm=True, dropout=0.1).to(device)) 

        """ model training """
        print("PID =", os.getpid())
        # train(dataset, model_list, device)
        # torch.save(model_list, "data/models/"+name+".pt")
        # exit()
        # """ models saving """
        # name = '_'.join(model.__class__.__name__ for model in model_list)
        # name = 'sp_8192w/Hete2HomoLayer_GCN_MLPN'
        # name = 'sp_8192w/Hete2HomoLayer_GraphSAGE_pool_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.80_acc_0.99_f1_weighted_0.99_f1_macro_0.83_439'
        # name = 'sp_8192w/Hete2HomoLayer_GraphSAGE_mean_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.89_acc_0.99_f1_weighted_0.99_f1_macro_0.88_198'
        # name = 'sp_8192w/Hete2HomoLayer_GraphSAGE_pool_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.81_acc_0.99_f1_weighted_0.99_f1_macro_0.82_440'
        # model_name = 'Hete2HomoLayer_GraphSAGE_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.98_acc_0.98_f1_weighted_0.99_f1_macro_0.98_242' # sandwich dataset
        # name = 'ultra8T/Hete2HomoLayer_GraphSAGE_MLPN_acc_0.98_f1_weighted_0.98_f1_macro_0.96_acc_0.99_f1_weighted_0.99_f1_macro_0.96_384'
        # model_name = 'Hete2HomoLayer_GraphSAGE_mean_MLPN_acc_1.00_f1_weighted_1.00_f1_macro_0.99_acc_1.00_f1_weighted_1.00_f1_macro_0.99_446' # ultra_8T dataset
        # model_name = "Hete2HomoLayer_GCN_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.97_acc_0.99_f1_weighted_0.99_f1_macro_0.96_349" # 
        # model_name = "ssram_Hete2HomoLayer_GCN_MLPN_acc_0.95_f1_weighted_0.96_f1_macro_0.88_acc_0.95_f1_weighted_0.96_f1_macro_0.88_472"
        # model_name = "ssram_Hete2HomoLayer_GraphSAGE_mean_MLPN_acc_0.98_f1_weighted_0.98_f1_macro_0.93_acc_0.98_f1_weighted_0.98_f1_macro_0.93_465"
        # model_name = "ssram_Hete2HomoLayer_GraphSAGE_pool_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.97_acc_0.99_f1_weighted_0.99_f1_macro_0.97_246"
        # model_name = "ssram_Hete2HomoLayer_GAT_MLPN_acc_0.93_f1_weighted_0.94_f1_macro_0.83_acc_0.93_f1_weighted_0.94_f1_macro_0.83_499"
        # model_name = "sram_sp_8192w_Hete2HomoLayer_GAT_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.72_acc_0.99_f1_weighted_0.99_f1_macro_0.72_471"
        # model_name = "ultra_8T_Hete2HomoLayer_GAT_MLPN_acc_0.97_f1_weighted_0.97_f1_macro_0.90_acc_0.97_f1_weighted_0.97_f1_macro_0.90_403"
        # model_name = "sandwich_Hete2HomoLayer_GAT_MLPN"

        # load classifier
        model_list_ld = torch.load("data/models/"+model_name+".pt")
        # print(model_list_ld)

        """ inference the whole nets """
        print('Inferencing whole nodes...')
        infr_start = datetime.now()

        for i, model in enumerate(model_list_ld):
            model.eval()
        logits, net_h = evaluation(dataset._bg, dataset.get_feat_dict(), model_list_ld)
        
        model_name = '_'.join(model.__class__.__name__ for model in model_list_ld)
        if "GraphSAGE" in model_name:
            model_name += "_" + model_list_ld[1].layers[0]._aggre_type
            print(model_list_ld[1].layers[0]._aggre_type)
        _, mask = dataset.get_masks()
        metrics = validation(logits, dataset.get_labels(), mask=mask, pltname=model_name+"_conf_matrix_infr")

        print("|| inference time {:s} | mean accuracy {:.2f}%  | weighted f1 score {:.2f} | f1 macro {:.2f} ||"
              .format(str(datetime.now() - infr_start), metrics['acc']*100, metrics['f1_weighted'], metrics['f1_macro']))
        # do statistics for the sample distribution
        class_distr = torch.zeros(dataset._num_classes)
        for i in range(dataset._num_classes):
            # class_distr[i] = (dataset.get_labels().squeeze() == i).sum().item()
            class_distr[i] = (logits.argmax(dim=1).squeeze() == i).sum().item()
        print('predicted distr:', class_distr)

        # net_h = None
        # if net_h is None:
        #     gnn_feat = 0
        
        """ Define regression models """
        model_list_c = nn.ModuleList()
        for i in range(dataset._num_classes):
            mlp_feats = [128, 128, 64]
            if i > 2:
                use_bn = False
                dropout = 0.0
            else:
                use_bn = False
                dropout = 0.5
            model_list_c.append(MLPN(dataset._n_feat_dim+gnn_feat, mlp_feats, 1, 
            # model_list_c.append(MLPN(dataset._n_feat_dim, mlp_feats, 1, 
                                     act=nn.ReLU(), use_bn=use_bn, has_l2norm=False, 
                                     dropout=dropout).to(device))
        
        """ model training """
        test_mean_err = torch.zeros(dataset._num_classes)
        test_max_err = torch.zeros(dataset._num_classes)
        # class_merge = [[0,1], [0,1], [1,2], [2,3], [3,4]]
        for i in range(dataset._num_classes):
            print('Training class {:d} ...'.format(i))
            model = model_list_c[i]
            # class_mask_train = torch.zeros(net_h.shape[0], dtype=torch.bool)
            # for class_idx in class_merge[i]:
            if net_h is not None:
                class_mask = logits.argmax(dim=1).squeeze() == i
            else:
                class_mask = dataset.get_labels().squeeze() == i
            test_metrics = train_cap(dataset, net_h, model, class_mask, i)
            test_mean_err[i] = test_metrics['mean_err']
            test_max_err[i] = test_metrics['max_err']
        weighted_err = test_mean_err * class_distr / class_distr.sum()
        weighted_max_err = test_max_err * class_distr / class_distr.sum()
        print('test_mean_err:', test_mean_err, 'avg:', weighted_err.sum().item())
        print('test_max_err:', test_max_err, 'avg:', weighted_max_err.sum().item())
        # model_name="ssram_nognn"
        torch.save(model_list_c, "data/models/"+model_name+"_regressors.pt")
        exit()

        model_list_c = torch.load("data/models/5_MLPs_2.pt")
        print('load the mlp regression models')
        