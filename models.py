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
    def forward(self, h_dict: torch.Tensor) -> torch.Tensor:
        # print("h_dict after:", [(key, h.shape) for key, h in h_dict.items()])
        trans_h_dict = self.layers(h_dict)
        # print("trans_h_dict:", [(key, h.shape) for key, h in trans_h_dict.items()])
        trans_h = torch.cat([value for _, value in trans_h_dict.items()], dim=0)
        return trans_h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dgl.nn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dgl.nn.GraphConv(hid_size, hid_size, activation=F.relu))
        self.layers.append(dgl.nn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h

class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads, feat_drop=0.6, attn_drop=0.6):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # 3-layer GAT
        self.gat_layers.append(dgl.nn.GATConv(in_size, hid_size, heads[0], feat_drop=feat_drop, attn_drop=attn_drop, activation=F.elu))
        self.gat_layers.append(dgl.nn.GATConv(hid_size*heads[0], hid_size, heads[1], feat_drop=feat_drop, attn_drop=attn_drop, activation=F.elu))
        self.gat_layers.append(dgl.nn.GATConv(hid_size*heads[1], out_size, heads[2], feat_drop=feat_drop, attn_drop=attn_drop, activation=None))
        
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            # print(i, "size of h:", h.shape)
            if i == len(self.gat_layers) - 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
            # print(i, "size of h:", h.shape)
        return h

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

""" initialize the weights in MLP3 """
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def mlp_block(in_dim, out_dim, nonlinear, use_bn, dropout):
    layers = []
    layers.append(nn.Linear(in_dim, out_dim))
    nonlinear_dict = {
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'relu': nn.ReLU()
    }
    if use_bn:
        layers.append(nn.BatchNorm1d(out_dim))
    if nonlinear is not None:
        layers.append(nonlinear_dict[nonlinear])
    if dropout != 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

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
                 mlp_out_dim=1, act=nn.ReLU(), use_bn=False, has_l2norm=False, dropout=0.0):
        super(MLPN, self).__init__()
        layer_wrapper = []
        layer_wrapper.append(MLPLayer(mlp_in_dim, hid_dims[0], act, use_bn, has_l2norm, dropout))
        for i in range(1,len(hid_dims)):
            layer_wrapper.append(MLPLayer(hid_dims[i-1], hid_dims[i], act, use_bn, has_l2norm, dropout))
        layer_wrapper.append(MLPLayer(hid_dims[-1], mlp_out_dim, None, False, False, 0.0))
        self.backbone = torch.nn.Sequential(*layer_wrapper)

    def forward(self, h):
        h = self.backbone(h)
        return h

# we define a simple MLP module here
class MLP3(nn.Module):
    def __init__(self, mlp_in_dim=32, hid_dims=[64, 128, 64], mlp_out_dim=1, nonlinear='relu', use_bn=False, dropout=0.0):
        super(MLP3, self).__init__()
        assert nonlinear in ['tanh', 'relu', 'sigmoid']
        layer_size = hid_dims # [64, 128, 64]
        layers = []
        layers.append(mlp_block(mlp_in_dim, layer_size[0], nonlinear, use_bn, dropout))
        for i in range(1,len(layer_size)):
            layers.append(mlp_block(layer_size[i-1], layer_size[i], nonlinear, use_bn, dropout))
        self.backbone = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(layer_size[-1], mlp_out_dim)
        self.backbone.apply(weights_init)
        self.out.apply(weights_init)

    def forward(self, h):
        h = self.backbone(h)
        h = self.out(h)
        return h
