import dgl
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class SRAMDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 name=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=True):
        super(SRAMDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        # self._labels = None
        # self._bg = None

    # no need to download
    def download(self):
        # download raw data to local disk
        pass

    # must be implemented
    def process(self):
        het_g, _ = load_graphs(self.raw_path + '.bi_graph.bin')
        shg = het_g[0].long()
        """ we use the directed subgraph and filter out the features of dst nodes """ 
        # remove the power pins
        if self.name == "sandwich":
            # VDD VSS TTVDD
            shg.remove_nodes(torch.tensor([0, 1, 1422]), ntype='net')
        elif self.name == "ultra_8T":
            # VDD VSS SRMVDD
            shg.remove_nodes(torch.tensor([0, 1, 377]), ntype='net')
        elif self.name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            shg.remove_nodes(torch.tensor([0, 1, 2]), ntype='net')
        elif self.name == "ssram":
            # VDD VSS VVDD
            shg.remove_nodes(torch.tensor([0, 1, 352]), ntype='net')
        elif self.name == "array_128_32_8t":
            shg.remove_nodes(torch.tensor([0, 1]), ntype='net')
        elif self.name == "8T_digitized_timing_top_fast":
            shg.remove_nodes(torch.tensor([0, 1]), ntype='net')
        else:
            raise Exception("No dataset named %s" % self.name)

        # mask out the 0.0 and nan targets
        # print("y:", shg.nodes['net'].data['y'].squeeze())
        zero_nids = (shg.nodes['net'].data['y'].squeeze() <= 0.0).bool()
        zero_nids |= (shg.nodes['net'].data['x'].sum(dim=1) <= 0.0).bool()
        # print("sum x:", shg.nodes['net'].data['x'].sum(dim=1))
        shg.remove_nodes(zero_nids.nonzero().squeeze(), ntype='net')
        
        self._num_n = shg.num_nodes('net')
        self._num_d = shg.num_nodes('device')
        self._num_i = shg.num_nodes('inst')
        print('load graph', self.name, 'from', self.raw_path)
        print('num of zero cap indeces:', len(zero_nids.nonzero()))
        print('num_n %d, num_d %d, num_i %i' % (self._num_n, self._num_d, self._num_i))
        print('total node num:', shg.num_nodes())
        print('shg.ntypes:', shg.ntypes)
        print('shg.etypes:', shg.etypes)
        # for comparison 
        print('shg.num_edges:', shg.num_edges('device-net'))
        # shg = dgl.edge_subgraph(het_g, {('device', 'device-net', 'net'): torch.tensor(range(het_g.num_edges('device-net')))})

        """ to bidirected homogeneous graph """
        g = dgl.to_homogeneous(shg)
        print('original NID in g:', g.ndata[dgl.NTYPE])
        # print('g.num_edges:', g.num_edges())
        # print('g.edges:', g.edges(form='all'))
        g = g.int()
        self._bg = dgl.to_bidirected(g, copy_ndata=False)
        self._hg = shg
        # self._bg = dgl.reorder_graph(bg)
        
        def feat_norm(x):
                x_max, _ = x.max(dim=0)
                x_max[x_max == 0.0] = torch.inf
                x_norm = x / x_max.expand(x.shape[0], -1)
                return x_norm.float()
        
        ### prepare the feature dict ###
        # delete the dummy features in device nodes
        if shg.nodes['device'].data['x'].shape[1] > 15:
            # print("self._d_feat shape", self._d_feat.shape)
            self._d_feat = torch.cat((shg.nodes['device'].data['x'][:, 0].view(-1, 1), 
                                      shg.nodes['device'].data['x'][:, 4:-1]), dim=1)
            self._d_feat = feat_norm(self._d_feat)
        else:
            self._d_feat = feat_norm(shg.nodes['device'].data['x'][:, :-1])
        self._d_feat_dim = self._d_feat.shape[1]
        assert self._d_feat_dim == 14

        self._n_feat = feat_norm(shg.nodes['net'].data['x'][:, :-1])
        self._n_feat_dim = self._n_feat.shape[1]
        self._i_feat = feat_norm(shg.nodes['inst'].data['x'].float())
        self._i_feat_dim = self._i_feat.shape[1]

        ### prepare the labels ###
        self._targets = shg.nodes['net'].data['y'].float() * 1000
        self.plot_targets()
        max_l, max_i = torch.max(self._targets, dim=0)
        min_l, min_i = torch.min(self._targets, dim=0)
        print('targets size:', self._targets.shape) 
        print('max labels:{:.2f} with index:{:d}; min labels:{:.2f} with index:{:d}'
              .format(max_l.item(), max_i.item(), min_l.item(),min_i.item()))
        # classificating according to the magnitude of the net capacitance
        self._num_classes = 5
        self._labels = None
        self.get_labels()
        
        ### set the training and testing masks ###
        nids = [i for i in range(self._num_n)]
        if (self.name == "array_128_32_8t" or self.name == "8T_digitized_timing_top_fast"):
            pass
        else:
            labels = self._labels
            for i, class_num in enumerate(self._class_num):
                if class_num < 2:
                    labels[labels == i] = i-1

            self._train_nids, self._val_nids = train_test_split(nids, test_size=0.2, 
                                                random_state=42, stratify=labels)
            # self._train_nids, self._val_nids = train_test_split(nids, test_size=0.25, 
            #                                     random_state=22, stratify=self._labels[train_nids])
            
            # self._test_mask = torch.zeros(self._num_n, dtype=torch.bool)
            self._train_mask = torch.zeros(self._num_n, dtype=torch.bool)
            self._val_mask = torch.zeros(self._num_n, dtype=torch.bool)
            # self._test_mask[self._test_nids] = True
            self._train_mask[self._train_nids] = True
            self._val_mask[self._val_nids] = True
            print('# of train/val samples:{:d}/{:d}'
                  .format(len(self._train_nids), len(self._val_nids)))#, len(self._test_nids)))
        return
    
    def get_test_mask(self):
        return self._test_mask
    
    def get_masks(self):
        return self._train_mask, self._val_mask #, self._test_mask
    
    def get_nids(self):
        # return torch.tensor(self._train_nids, dtype=torch.int32), torch.tensor(self._val_nids, dtype=torch.int32), torch.tensor(self._test_nids, dtype=torch.int32)
        return torch.tensor(self._train_nids, dtype=torch.int64), torch.tensor(self._val_nids, dtype=torch.int64)#, torch.tensor(self._test_nids, dtype=torch.int64)
    
    def get_feat_dict(self):
        return {'device': self._d_feat, 'inst': self._i_feat, 'net': self._n_feat}

    def get_targets(self):
        return self._targets
    
    def get_labels(self, mask=None):
        if self._labels is not None:
            return self._labels
        
        targets = self._targets
        if mask is not None:
            targets = self._targets[mask]
        
        labels = torch.zeros(targets.shape)
        binedges = [0.01, 0.1, 1.0, 10.0, 100.0, torch.inf]
        # binedges = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        class_num = []
        for i in range(len(binedges)-1):
            index = (targets >= binedges[i]) & (targets < binedges[i+1])
            labels[index] = i
            class_num.append(torch.sum(index.squeeze()).item())
            # avoid zero denominator
            if class_num[-1] == 0:
                class_num[-1] = 1

        
        self._labels = labels.long()
        self._class_num = torch.tensor(class_num)
        print('class distribution:', class_num)
        # assert 0
        class_distr = torch.tensor([self._num_n / item for item in class_num])
        self.alpha = class_distr
        return self._labels

    def get_large_cap_mask(self):
        return self._targets > 0.1

    # must be implemented
    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._hg
    
    # must be implemented
    def __len__(self):
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

    def plot_targets(self):
        lb_idx = (self._targets != 0.0).squeeze()
        x = self._targets[lb_idx].squeeze().numpy()
        # print('x:', x)
        sns.set_style('darkgrid')
        bins = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        fig, ax = plt.subplots()
        sns.histplot(x, kde=False, bins=bins, edgecolor='w', linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(bins)
        ax.set_xlabel('Cap (fF)')
        ax.set_title(self.name + ' SRAM')
        # plt.xlim(x.min(), x.max())
        # plt.savefig('./data/plots/'+ self.name + '_label_hist', dpi=400)