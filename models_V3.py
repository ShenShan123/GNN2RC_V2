import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from HeteroLinear import HeteroMLPLayer

""" This is a feature projection layer (basic structure is MLP) """
class Hete2HomoLayer(nn.Module):
    def __init__(self, linear_dict, act=None, has_l2norm=False, has_bn=True, dropout=0.0):
        super(Hete2HomoLayer, self).__init__()
        self.layers = HeteroMLPLayer(linear_dict, has_l2norm=has_l2norm, has_bn=has_bn,
                                     act=act, dropout=dropout, final_act=False)
        self.proj_dim = linear_dict["net"][-1]
        self.input_feat_dims = {key: dims[0] for key, dims in linear_dict.items()}
        
    '''
    def forward(self, shg, h_dict: torch.Tensor) -> torch.Tensor:
        h_dict = self.layers(h_dict)
        # set the message function and reduce function in update_all()
        funcs = {}
        dsttype = ""
        dst_h0 = None
        for c_etype in shg.canonical_etypes:
            srctype, etype, dsttype = c_etype
            # Save it in graph for message passing
            shg.nodes[srctype].data['h'] = h_dict[srctype]
            if dsttype in h_dict:
                dst_h0 = h_dict[dsttype]
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        # Trigger message passing of multiple types.
        shg.multi_update_all(funcs, 'sum')
        trans_h_dict = shg.ndata['h']
        if dst_h0 is not None:
            trans_h_dict[dsttype] = trans_h_dict[dsttype] + dst_h0
        # return the updated node features in the feature dictionary
        trans_h = torch.cat([value for _, value in trans_h_dict.items()], dim=0)
        return trans_h
    '''
    '''
    def forward(self, h_dict: torch.Tensor) -> torch.Tensor:
        # print("h_dict after:", [(key, h.shape) for key, h in h_dict.items()])
        trans_h_dict = self.layers(h_dict)
        #print("trans_h_dict:", trans_h_dict)#[(key, h.shape) for key, h in trans_h_dict.items()])
        trans_h = torch.cat([value for _, value in trans_h_dict.items()], dim=0)
        return trans_h
    '''

    def forward(self, feats: torch.Tensor, ntypes: torch.Tensor) -> torch.Tensor:
        ## restore the feats to h_dict with dim_list defining the dim number for each feat type
        h_dict = {"device": feats[ntypes == 0][:, 0:self.input_feat_dims['device']], 
                  "inst":feats[ntypes == 1][:, 0:self.input_feat_dims['inst']], 
                  "net": feats[ntypes == 2][:, 0:self.input_feat_dims['net']]}
        trans_h_dict = self.layers(h_dict)
        h = torch.zeros(feats.shape[0], self.proj_dim, device=feats.device)
        # the node order of h is the same as feats
        h[ntypes == 0] = trans_h_dict["device"]
        h[ntypes == 1] = trans_h_dict["inst"]
        h[ntypes == 2] = trans_h_dict["net"]
        return h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # 3-layer GCN
        self.layers.append(dgl.nn.GraphConv(in_size, hid_size, activation=F.relu))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size, activation=F.relu))
        self.layers.append(dgl.nn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h
    
    # def forward(self, g, features):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         if i != len(self.layers) - 1:
    #             h = self.dropout(h)
    #         h = layer(g, h)
    #     return h

# for regression and classification GAT 3-layer is better than 2-layer version
class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads, feat_drop=0.6, attn_drop=0.6):
        super().__init__()
        self.layers = nn.ModuleList()
        # 3-layer GAT
        self.layers.append(dgl.nn.GATConv(in_size, hid_size, heads[0], 
                                          feat_drop=feat_drop, attn_drop=attn_drop, 
                                          activation=F.elu))
        self.layers.append(dgl.nn.GATConv(hid_size*heads[0], hid_size, heads[1], 
                                          feat_drop=feat_drop, attn_drop=attn_drop, 
                                          activation=F.elu))
        self.layers.append(dgl.nn.GATConv(hid_size*heads[0], hid_size, heads[1], 
                                          feat_drop=feat_drop, attn_drop=attn_drop, 
                                          activation=F.elu))
        self.layers.append(dgl.nn.GATConv(hid_size*heads[1], out_size, heads[2], 
                                          feat_drop=feat_drop, attn_drop=attn_drop, 
                                          activation=None))
        
    def forward(self, blocks, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            # print(i, "size of h:", h.shape)
            if i == len(self.layers) - 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
            # print(i, "size of h:", h.shape)
        return h
    
    # def forward(self, g, inputs):
    #     h = inputs
    #     for i, layer in enumerate(self.layers):
    #         h = layer(g, h)
    #         # print(i, "size of h:", h.shape)
    #         if i == len(self.layers) - 1:  # last layer 
    #             h = h.mean(1)
    #         else:       # other layer(s)
    #             h = h.flatten(1)
    #         # print(i, "size of h:", h.shape)
    #     return h

class GATv2(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(dgl.nn.GATv2Conv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=False, share_weights=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(dgl.nn.GATv2Conv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=False, share_weights=True))
        # output projection
        self.gatv2_layers.append(dgl.nn.GATv2Conv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=False, share_weights=True))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](g, h).mean(1)
        return logits

# 2-layer SAGE is good for classification 
# 3-layer SAGE is good for Cap regression
class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(dgl.nn.SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(dgl.nn.SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    # def forward(self, g, x):
    #     h = x
    #     for l, layer in enumerate(self.layers):
    #         h = layer(g, h)
    #         if l != len(self.layers) - 1:
    #             h = F.relu(h)
    #             h = self.dropout(h)
    #     return h

""" initialize the weights in MLP3 """
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act, use_bn, has_l2norm, dropout):
        super(MLPLayer, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.has_l2norm = has_l2norm
        layer_wrapper = []
        if use_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_dim))
        if dropout != 0.0:
            layer_wrapper.append(nn.Dropout(dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.layer.apply(weights_init)
        self.post_layer.apply(weights_init)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply Linear, BatchNorm1d, Dropout and normalize(if need).
        """
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h

# we define a simple MLP module here
class MLPN(nn.Module):
    def __init__(self, mlp_in_dim=32, hid_dims=[64, 128, 64], 
                 mlp_out_dim=1, act=nn.ReLU(), use_bn=False, 
                 has_l2norm=False, dropout=0.0):
        super(MLPN, self).__init__()
        layer_wrapper = []
        if len(hid_dims):
            layer_wrapper.append(MLPLayer(mlp_in_dim, hid_dims[0], act, use_bn, has_l2norm, dropout))
            for i in range(1,len(hid_dims)):
                layer_wrapper.append(MLPLayer(hid_dims[i-1], hid_dims[i], act, use_bn, has_l2norm, dropout))
            layer_wrapper.append(MLPLayer(hid_dims[-1], mlp_out_dim, None, False, False, 0.0))
        else:
            layer_wrapper.append(MLPLayer(mlp_in_dim, mlp_out_dim, None, False, False, 0.0))
        self.backbone = torch.nn.Sequential(*layer_wrapper)

    def forward(self, h):
        h = self.backbone(h)
        return h


class NetCapClassifier(nn.Module):
    def __init__(self, num_classes, proj_dim_dict, gnn_dim_list=[64, 64, 64], 
                 cmlp_dim_list=[64, 64, 64], gnn='sage-mean',  
                 has_l2norm=False, has_bn=False, dropout=0.1, 
                 device=torch.device('cuda:0')):
        super(NetCapClassifier, self).__init__()
        self.device = device
        self.num_classes = num_classes
        # self.layers = nn.ModuleList()
        self.name = "netcapclassifier"+str(len(proj_dim_dict['net'])-1)+"_"+gnn+"_mlpclass"+ \
                    str(len(cmlp_dim_list))
        """ feature projection """ 
        self.feat_proj = Hete2HomoLayer(proj_dim_dict, act=nn.ReLU(), 
                                          has_bn=has_bn, has_l2norm=has_l2norm, 
                                          dropout=dropout).to(device)
        self.proj_dim = gnn_dim_list[0]
        """ a linear layer for sentence embedding """
        self.emb = MLPN(mlp_in_dim=384, hid_dims=[16], mlp_out_dim=8, 
                        act=nn.ReLU(), use_bn=has_bn, has_l2norm=True, 
                        dropout=dropout).to(device)
        
        """ create GNN model """ 
        if gnn == "gcn":
            # a 3-layer GCN
            self.gnn = GCN(gnn_dim_list[0], gnn_dim_list[1], 
                           gnn_dim_list[2], dropout=0.1).to(device)
        elif gnn == "gat":
            self.gnn = GAT(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 
                           heads=[1, 1, 1], feat_drop=dropout, 
                           attn_drop=dropout).to(device)
        # 2-layer graphSAGE for old version
        elif gnn == "sage-mean":
            self.gnn = GraphSAGE(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 1, # change layers
                                 activation=nn.ReLU(), dropout=dropout, 
                                 aggregator_type="mean").to(device)
        elif gnn == "sage-pool":
            self.gnn = GraphSAGE(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 1, 
                                 activation=nn.ReLU(), dropout=dropout, 
                                 aggregator_type="pool").to(device)
        """ MLP classifier at the last layer """
        self.class_mlp = MLPN(cmlp_dim_list[0]+self.feat_proj.input_feat_dims['net'], 
                              cmlp_dim_list[1:], self.num_classes, 
                              act=nn.ReLU(), use_bn=has_bn, has_l2norm=has_l2norm, 
                              dropout=dropout).to(device)
        
    def forward(self, blocks, h_embd):
        ## heterogeneous feature projection to common space
        feats = blocks[0].srcdata['x']
        ntypes = blocks[0].srcdata['_TYPE']
        proj_h = self.feat_proj(feats, ntypes)
        h_embd = self.emb(h_embd)
        ## concatenate name embeddings
        proj_h = torch.cat((proj_h, h_embd), dim=1)
        # proj_h += h_embd
        dst_h = self.gnn(blocks, proj_h)
        ## concatenate gnn embeddings and original feats
        n_feat = blocks[-1].dstdata['x']
        n_feat = n_feat[:, 0:self.feat_proj.input_feat_dims['net']]
        net_h = torch.cat([dst_h, n_feat], dim=1)
        ## feeding to MLP classifer
        h = self.class_mlp(net_h)
        # prob_t, indices = l.max(dim=1, keepdim=True)
        return h, dst_h


class NetCapRegressor(nn.Module):
    def __init__(self, proj_dim_dict, gnn_dim_list=[64, 64, 64],
                 reg_dim_list=[64, 128, 128, 64, 1],
                 gnn='sage-mean',  has_l2norm=False, has_bn=False, dropout=0.1, 
                 device=torch.device('cuda:0')):
        super(NetCapRegressor, self).__init__()
        self.device = device
        # self.num_classes = num_classes
        self.name = "netcapregressor"+str(len(proj_dim_dict['net'])-1)+"_"+gnn+"_mlp"+ \
                    "_regmlp"+str(len(reg_dim_list)-1)
        """ feature projection """ 
        self.feat_proj = Hete2HomoLayer(proj_dim_dict, act=nn.ReLU(), 
                                        has_bn=has_bn, has_l2norm=has_l2norm, 
                                        dropout=dropout).to(device)
        # self.proj_dim = gnn_dim_list[0]
        """ a linear layer for sentence embedding """
        self.emb = MLPN(mlp_in_dim=384, hid_dims=[16], mlp_out_dim=8, 
                        act=nn.ReLU(), use_bn=has_bn, has_l2norm=True, 
                        dropout=dropout).to(device)
        
        """ create GNN model """ 
        if gnn == "gcn":
            # a 3-layer GCN
            self.gnn = GCN(gnn_dim_list[0], gnn_dim_list[1], 
                           gnn_dim_list[2], dropout=0.1).to(device)
        elif gnn == "gat":
            self.gnn = GAT(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 
                           heads=[1, 1, 1], feat_drop=dropout, 
                           attn_drop=dropout).to(device)
        # 2-layer graphSAGE for old version
        elif gnn == "sage-mean":
            self.gnn = GraphSAGE(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 1, # change layers
                                 activation=nn.ReLU(), dropout=dropout, 
                                 aggregator_type="mean").to(device)
        elif gnn == "sage-pool":
            self.gnn = GraphSAGE(gnn_dim_list[0], gnn_dim_list[1], gnn_dim_list[2], 1, 
                                 activation=nn.ReLU(), dropout=dropout, 
                                 aggregator_type="pool").to(device)

        """ MLP regressor """
        self.regr_mlp = MLPN(reg_dim_list[0]+self.feat_proj.input_feat_dims['net'], 
                             reg_dim_list[1:-1], reg_dim_list[-1], 
                             act=nn.ReLU(), use_bn=has_bn, has_l2norm=False, 
                             dropout=dropout).to(device)
    
    def forward(self, blocks, h_embd):
        ## heterogeneous feature projection to common space
        feats = blocks[0].srcdata['x']
        ntypes = blocks[0].srcdata['_TYPE']
        proj_h = self.feat_proj(feats, ntypes)
        h_embd = self.emb(h_embd)
        ## concatenate name embeddings
        proj_h = torch.cat((proj_h, h_embd), dim=1)
        # proj_h += h_embd
        dst_h = self.gnn(blocks, proj_h)
        ## concatenate gnn embeddings and original feats
        n_feat = blocks[-1].dstdata['x']
        n_feat = n_feat[:, 0:self.feat_proj.input_feat_dims['net']]
        net_h = torch.cat([dst_h, n_feat], dim=1)
        ## feeding to MLP classifer
        h = self.regr_mlp(net_h)
        return h
    
class NetCapRegressorEnsemble(nn.Module):
    def __init__(self, num_classes, proj_dim_dict, gnn_dim_list=[64, 64, 64],
                 reg_dim_list=[64, 128, 128, 64, 1], gnn='sage-mean',  
                 has_l2norm=False, has_bn=False, dropout=0.1, 
                 device=torch.device('cuda:0')):
        super(NetCapRegressorEnsemble, self).__init__()
        self.regressors = nn.ModuleList()
        self.num_classes = num_classes
        self.name = "regressorensemle_"+str(num_classes)+"gnnclassifiers"
        for i in range(num_classes):
            self.regressors.append(
                NetCapRegressor(proj_dim_dict=proj_dim_dict, 
                                gnn=gnn, has_l2norm=has_l2norm, 
                                has_bn=has_bn, dropout=dropout, 
                                device=device))

    def forward(self, blocks, regIdx, h_embd):
        model = self.regressors[regIdx]
        h = model(blocks, h_embd)
        return h
    
class ParaGraphGNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(dgl.nn.SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(dgl.nn.SAGEConv(n_hidden, out_feats, aggregator_type)) # activation None

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    

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
                NetCapRegressor(proj_dim_dict=proj_dim_dict, 
                                gnn=gnn, has_l2norm=has_l2norm, 
                                has_bn=has_bn, dropout=dropout, 
                                device=device))

    def forward(self, blocks, regIdx):
        model = self.regressors[regIdx]
        h = model(blocks)
        return h
    """
class SentenceEmbedding(nn.Module):
    def __init__(self, mlp_in_dim=178, hid_dims=None, 
                 mlp_out_dim=32, act=nn.ReLU(), use_bn=False, 
                 has_l2norm=False, dropout=0.0, device=torch.device('cpu')):
        super(SentenceEmbedding, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MLPN(mlp_in_dim=mlp_in_dim, 
                                hid_dims=[128], mlp_out_dim=mlp_out_dim, 
                                act=nn.ReLU(), use_bn=use_bn, has_l2norm=has_l2norm, 
                                dropout=dropout).to(device))

    def forward(self, h):
        # mean_pooling(model_output)
        h = self.layers[0](h)
        return h
        """
