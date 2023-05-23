import torch
import torch.nn as nn
from models import Hete2HomoLayer, GAT, GATv2, MLP3, GCN, GraphSAGE, MLPN
from SRAM_dataset import SRAMDataset
from regression import train_cap, evaluation_caps
from classification import train, validation, evaluation
from datetime import datetime
from utils.circuits2graph import run_cir2g


if __name__ == '__main__':
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # 8T_digitized_timing_top_fast array_128_32_8t
    dataset = SRAMDataset(name='8T_digitized_timing_top_fast', raw_dir='/data1/shenshan/RCpred')
    train_ds_name = "sandwich"
    for i in range(1):
        # model_name = 'Hete2HomoLayer_GraphSAGE_mean_MLPN_acc_1.00_f1_weighted_1.00_f1_macro_0.99_acc_1.00_f1_weighted_1.00_f1_macro_0.99_446'
        model_name = "Hete2HomoLayer_GraphSAGE_MLPN_acc_0.99_f1_weighted_0.99_f1_macro_0.98_acc_0.98_f1_weighted_0.99_f1_macro_0.98_242"
        # load classifier
        model_list_ld = torch.load("data/models/"+train_ds_name+"/"+model_name+".pt")
        # print(model_list_ld)

        """ inference the whole nets """
        print('Inferencing whole nodes...')
        infr_start = datetime.now()

        for i, model in enumerate(model_list_ld):
            model_list_ld.eval()
        logits, net_h = evaluation(dataset._bg, dataset.get_feat_dict(), model_list_ld)
        
        # do statistics for the sample distribution
        class_distr = torch.zeros(dataset._num_classes)
        for i in range(dataset._num_classes):
            # class_distr[i] = (dataset.get_labels().squeeze() == i).sum().item()
            class_distr[i] = (logits.argmax(dim=1).squeeze() == i).sum().item()
        print('predicted distr:', class_distr)
        
        """ Define regression models """
        model_name = '5_MLPs_sage_mean_sandwich' # "5_MLPs_sage_mean_ultra_8t"
        model_list_c_ld = torch.load("data/models/"+train_ds_name+ "/"+ model_name+".pt")
        if net_h is not None:
            net_h = torch.cat([net_h, dataset._n_feat], dim=1)
        
        c_pred = torch.zeros(logits.shape[0], 1)

        for i in range(dataset._num_classes):
            class_mask = logits.argmax(dim=1).squeeze() == i
            class_id = torch.nonzero(class_mask.squeeze()).squeeze()
            model_list_c_ld[i].eval()
            c_pred[class_id] = evaluation_caps(net_h[class_id], model_list_c_ld[i])
            
        print("c_pred:", c_pred)
        
        # assert 0
        # front_8T_digitized_timing_top_256_fast # array_128_32_8t
        run_cir2g(dataset.name, "front_8T_digitized_timing_top_256_fast", feat_Y=c_pred)