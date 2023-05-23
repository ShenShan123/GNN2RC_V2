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

    # no need to download
    def download(self):
        # download raw data to local disk
        pass

    # must be implemented
    def process(self):
        het_g, _ = dgl.load_graphs(self.raw_path + '.bi_graph.bin')
        het_g = het_g[0].long()
        """ we use the directed subgraph and filter out the features of dst nodes """
        # shg = dgl.edge_type_subgraph(hg, [('device', 'device-net', 'net'),])
        shg = het_g
        (xd_max, xn_max, xi_max) = (None, None, None)
        # remove the power pins
        if self.name == "sandwich":
            shg.remove_nodes(torch.tensor([0, 1, 1425]), ntype='net')
        elif self.name == "ultra_8T":
            shg.remove_nodes(torch.tensor([0, 1]), ntype='net')
        elif self.name == "sram_sp_8192w":
            shg.remove_nodes(torch.tensor([0, 1, 2]), ntype='net')
        elif self.name == "array_128_32_8t" or self.name == "8T_digitized_timing_top_fast":
            # These are collected from ultra8T features
            # xd_max= torch.tensor([1.2800e+02, 1.2800e+02, 1.2800e+02, 1.2800e+02, 2.4000e-06, 2.0000e-06,
            #                 2.0000e+00, 2.0000e-06, 1.7400e-04, 4.0000e+01, 2.8000e+02, 1.0500e-05,
            #                 torch.inf, torch.inf,  torch.inf, 4.0000e+00, 4.0000e+00], dtype=torch.float64)
            # xn_max = torch.tensor([1.0658e+05, 3.4760e+03, 9.6860e+04, 1.0644e+05, 3.7819e-02, 6.9600e-04,
            #                         2.0000e+00, 4.0000e-06, 1.7400e-04, 4.1000e+01, 8.1600e+03, 3.3000e-04,
            #                         torch.inf, torch.inf, torch.inf, 4.5845e+04], dtype=torch.float64)
            # xi_max = torch.tensor([7.0588e+02, 4.4988e+02, 1.1849e+03, 8.7500e-01], dtype=torch.float64)
            # These are collected from sandwich features
            xd_max= torch.tensor([5.0000e+00, 5.0000e+00, 5.0000e+00, 5.0000e+00, 3.0000e-06, 1.0000e-06,
                            torch.inf, torch.inf, torch.inf, torch.inf, torch.inf, torch.inf,
                            torch.inf, torch.inf, torch.inf, 4.0000e+00, 1.0000e+00], dtype=torch.float64)
            xn_max= torch.tensor([6.2740e+04, 5.0180e+04, 1.2560e+04, torch.inf, 1.2630e-03, 1.5061e-03,
                            torch.inf, torch.inf, torch.inf, torch.inf, torch.inf,torch.inf,
                            torch.inf, torch.inf, torch.inf, 5.4140e+04], dtype=torch.float64)
            xi_max= torch.tensor([2.4279e+03, 2.9988e+02, 2.9269e+03, 8.7500e-01], dtype=torch.float64)

        self._num_n = shg.num_nodes('net')
        self._num_d = shg.num_nodes('device')
        self._num_i = shg.num_nodes('inst')
        print('load graph', self.name, 'from', self.raw_path)
        print(shg)
        print('shg.num_nodes:', shg.num_nodes())
        print('shg.ntypes:', shg.ntypes)
        print('shg.etypes:', shg.etypes)
        # for comparison 
        print('shg.num_edges:', shg.num_edges('device-net'))
        # shg = dgl.edge_subgraph(het_g, {('device', 'device-net', 'net'): torch.tensor(range(het_g.num_edges('device-net')))})
        # print(shg)
        
        """ to bidirected homogeneous graph """
        g = dgl.to_homogeneous(shg)
        print('original NID in g:', g.ndata[dgl.NTYPE])
        # print('g.num_edges:', g.num_edges())
        # print('g.edges:', g.edges(form='all'))
        g = g.int()
        bg = dgl.to_bidirected(g, copy_ndata=False)
        
        self._bg = dgl.reorder_graph(bg)
        # self._shg = shg

        # normalize the feature with the maximum values
        def feat_norm(x, x_max=None, x_min=None):
            if len(x) < 1:
                return x
        
            if x_min is None:
                # bug to be fixed
                x = x - x.min()
                print("x_min=", x.min())
            else:
                x = x - x_min

            if x_max is None:
                x_max, max_idx = torch.max(x, 0)
                x_max[(x_max == 0.0)] = torch.inf
                print("x_max=", x_max)
                x_norm = x / x_max.expand(x.shape[0], -1)
            else:
                x_norm = x / x_max.expand(x.shape[0], -1)
            
            return x_norm
            
        ### prepare the feature dict ###
        self._d_feat = feat_norm(shg.nodes['device'].data['x'][:, :-1],xd_max).float()
        self._d_feat_dim = self._d_feat.shape[1]
        self._n_feat = feat_norm(shg.nodes['net'].data['x'][:, :-1], xn_max).float()
        self._n_feat_dim = self._n_feat.shape[1]
        self._i_feat = feat_norm(shg.nodes['inst'].data['x'], xi_max).float()
        self._i_feat_dim = self._i_feat.shape[1]
        
        ### prepare the labels ###
        self._targets = shg.nodes['net'].data['y'].float() * 1000
        self.plot_targets()
        # mask out the 0.0 and nan targets
        zero_mask = (self._targets == 0.0) | self._targets.isnan()
        max_l, max_i = torch.max(self._targets[~zero_mask], dim=0)
        min_l, min_i = torch.min(self._targets[~zero_mask], dim=0)
        print('targets size:', self._targets.shape) 
        print('max labels:{:.2f} with index:{:d}; min labels:{:.2f} with index:{:d}'
              .format(max_l.item(), max_i.item(), min_l.item(),min_i.item()))
        print('num of zero cap indeces:', zero_mask.squeeze().sum().item())
        # classificating according to the magnitude of the net capacitance
        self._num_classes = 5
        self._labels = None
        # here we remove the net with 0.0 cap (invalid data samples)
        nids = torch.nonzero(~zero_mask.squeeze()).squeeze().tolist()
        if self.name == "array_128_32_8t" or self.name == "8T_digitized_timing_top_fast":
            pass
        else:
            self.get_labels()
            ### set the training and testing masks ###
            train_nids, self._test_nids = train_test_split(nids, test_size=0.2, 
                                                random_state=42, stratify=self._labels[nids])
            self._train_nids, self._val_nids = train_test_split(train_nids, test_size=0.25, 
                                                random_state=22, stratify=self._labels[train_nids])
            
            self._test_mask = torch.zeros(self._num_n, dtype=torch.bool)
            self._train_mask = torch.zeros(self._num_n, dtype=torch.bool)
            self._val_mask = torch.zeros(self._num_n, dtype=torch.bool)
            self._test_mask[self._test_nids] = True
            self._train_mask[self._train_nids] = True
            self._val_mask[self._val_nids] = True
            print('# of train/val/test samples:{:d}/{:d}/{:d}'
                .format(len(self._train_nids), len(self._val_nids), len(self._test_nids)))
        return

    def get_val_mask(self, test_mask):
        perm = torch.randperm(test_mask.shape[0])
        num_train = int(test_mask.shape[0] * 0.8)
        # print('test_mask shape:', test_mask.shape, 'num_train:', num_train)
        # 20% target nodes remain to be validated
        train_mask = torch.zeros(test_mask.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(test_mask.shape[0], dtype=torch.bool)
        train_mask |= ~test_mask
        val_mask |= train_mask
        train_mask[perm[num_train:]] = False
        val_mask[perm[:num_train]] = False
        # print('train_mask:', train_mask)
        return val_mask, train_mask
    
    def get_test_mask(self):
        return self._test_mask
    
    def get_masks(self):
        return self._train_mask, self._val_mask, self._test_mask
    
    def get_nids(self):
        # return torch.tensor(self._train_nids, dtype=torch.int32), torch.tensor(self._val_nids, dtype=torch.int32), torch.tensor(self._test_nids, dtype=torch.int32)
        return torch.tensor(self._train_nids, dtype=torch.int64), torch.tensor(self._val_nids, dtype=torch.int64), torch.tensor(self._test_nids, dtype=torch.int64)
    
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
        
        self._labels = labels.long()
        
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
        plt.savefig('./data/plots/'+ self.name + '_label_hist', dpi=400)
