import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
# from HeteroLinear import HeteroMLPLayer
from models import MLPN

class ParaGraphLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, act=None):
        super(ParaGraphLayer, self).__init__()
        # W_r for each relation
        self.gatDict = nn.ModuleDict({
                name : nn.GATConv(in_size, out_size, num_heads=1, bias=False) for name in etypes
            })
        self.wl = nn.Linear(in_size*2, out_size, bias=False)
        self.activation = act
        self.bias = nn.Parameter(torch.FloatTensor(size=(out_size)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.wl.weight, gain=gain)
        nn.init.constant_(self.bias, 0)

    def forward(self, graph, feat_dict):
        if isinstance(feat_dict, tuple) or graph.is_block:
            src_inputs, dst_inputs = feat_dict
            if isinstance(feat_dict, tuple):
                src_inputs, dst_inputs = feat_dict
            else:
                src_inputs = feat_dict
                dst_inputs = {k: v[:graph.number_of_dst_nodes(k)] for k, v in feat_dict.items()}
        else:
            src_inputs = dst_inputs = feat_dict

            dst_output = {k : torch.zeros(v.shape) for k, v in dst_inputs.items()}
            # The input is a dictionary of node features for each type
            for srctype, etype, dsttype in graph.canonical_etypes:
                rel_graph = graph[srctype, etype, dsttype]
                if (srctype not in src_inputs) or (dsttype not in dst_inputs):
                        continue
                gatlayer = self.gatDict[etype]
                ## GATLayer computer line 5-8 in Algorithm1
                h_i_t = gatlayer(rel_graph, (src_inputs[srctype], dst_inputs[dsttype]))
                ## reduce function, line 9, is sum
                dst_output[dsttype] += h_i_t
                for ntype in dst_output.keys():
                    dst_output[ntype] = self.activation(
                                            self.wl(
                                                torch.cat((dst_inputs[ntype], 
                                                           dst_output[ntype]+self.bias
                                                          ), dim=1)))
        return dst_output

class ParaGraphNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 act):
        super(ParaGraphNN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(ParaGraphLayer(in_feats, n_hidden, act=act))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(ParaGraphLayer(n_hidden, n_hidden, act=act))
        # output layer
        self.layers.append(ParaGraphLayer(n_hidden, out_feats, act=act)) # activation None

    def forward(self, blocks):
        # h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, (block.srcdata, block.dstdata))
        return h

class ParaGraph(nn.Module):
    def __init__(self, proj_dim_dict, gnn_dim_list=[32, 32, 32],
                 reg_dim_list=[32, 32, 32, 32, 1],
                 has_l2norm=False, has_bn=False, dropout=0.1, 
                 device=torch.device('cuda:0')):
        super(ParaGraph, self).__init__()
        self.device = device
        # self.num_classes = num_classes
        self.name = "paragraph"+str(len(proj_dim_dict['net'])-1)+"_mlp"+ \
                    "_regmlp"+str(len(reg_dim_list)-1)
        """ feature projection """ 
        self.feat_proj = nn.ModuleDict({ntype: nn.Linear(proj_dim_dict[ntype][0], 
                                                         proj_dim_dict[ntype][-1], 
                                                         bias=False) 
                                        for ntype in proj_dim_dict.keys()})
        
        """ create GNN model """ 
        # a 5-layer ParaGraphNN
        self.gnn = ParaGraphNN(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 
                               n_layers=5, dropout=0.1).to(device)

        """ MLP regressor """
        self.regr_mlp = MLPN(reg_dim_list[0], 
                             reg_dim_list[1:-1], reg_dim_list[-1], 
                             act=nn.ReLU(), use_bn=has_bn, has_l2norm=has_l2norm, 
                             dropout=dropout).to(device)
    
    def forward(self, blocks):
        feat_dict = {}
        for ntype in blocks.ntypes:
            feat_dict[ntype] = \
                self.feat_proj[ntype](blocks.node[ntype].data['x'])
        # ntypes = blocks[0].srcdata['_TYPE']
        # proj_h = self.feat_proj(feats, ntypes)
        dst_h = self.gnn(blocks, feat_dict)
        ## concatenate gnn embeddings and original feats
        # n_feat = blocks[-1].dstdata['x']
        # n_feat = n_feat[:, 0:self.feat_proj.input_feat_dims['net']]
        # net_h = torch.cat([dst_h, n_feat], dim=1)
        ## feeding to MLP classifer
        h = self.class_mlp(dst_h['net'])
        # prob_t, indices = l.max(dim=1, keepdim=True)
        return h, dst_h
    

class ParaGraphEnsemble(nn.Module):
    def __init__(self, num_classes, proj_dim_dict, gnn_dim_list=[32, 32, 32],
                 reg_dim_list=[32, 32, 32, 32, 1], gnn='sage-mean',  
                 has_l2norm=False, has_bn=False, dropout=0.0, 
                 device=torch.device('cuda:0')):
        super(ParaGraphEnsemble, self).__init__()
        self.regressors = nn.ModuleList()
        self.num_classes = num_classes
        self.name = "regressorensemle_"+str(num_classes)+"gnnclassifiers"
        for i in range(num_classes):
            self.regressors.append(
                ParaGraph(proj_dim_dict=proj_dim_dict, 
                          gnn=gnn, has_l2norm=has_l2norm, 
                          has_bn=has_bn, dropout=dropout, 
                          device=device))

    def forward(self, blocks, regIdx):
        model = self.regressors[regIdx]
        h = model(blocks)
        return h