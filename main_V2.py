import torch
from models import NetCapClassifier, NetCapRegressorEnsemble
from SRAM_dataset_gbatch import SRAMDatasetList
from datetime import datetime
# from utils.circuits2graph import run_struct2g
from train_gbatch_V2 import train

if __name__ == '__main__':
    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3') 
    dataset = SRAMDatasetList(nameList=['ssram', 'ultra_8T', 'sandwich',], #'sram_sp_8192w'], #   
                              device=device, test_ds=False)
    datasetTest = SRAMDatasetList(nameList=['sram_sp_8192w', ], device=device, #'array_128_32_8t',
                                  test_ds=True, featMax=dataset.featMax)
    # assert 0
    linear_dict = {'device': [dataset._d_feat_dim, 64, 64], 
                   'inst':   [dataset._i_feat_dim, 64, 64], 
                   'net':    [dataset._n_feat_dim, 64, 64]}
    model = NetCapClassifier(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
                             gnn='sage-mean', has_l2norm=True, has_bn=False, dropout=0.1, 
                             device=device)
    # modelr = MLPRegressor(num_classes=dataset._num_classes, 
    #                          reg_dim_list=[64+dataset._n_feat_dim+1, 128, 128, 64, 1],
    #                          has_l2norm=False, 
    #                          has_bn=True, device=device)
    # modelr = []
    # modelr.append(NetCapRegressor(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
    #                         gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
    #                         device=device))
    # modelr.append(NetCapRegressor(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
    #                         gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
    #                         device=device))
    modelens = NetCapRegressorEnsemble(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
                                       gnn='sage-mean', has_l2norm=True, has_bn=False, dropout=0.1, 
                                       device=device)
    train(dataset, datasetTest, model, modelens, device)