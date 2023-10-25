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
                 verbose=True,
                 test=False):
        self.testDS = test
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
        # print('g.num_edges:', g.num_edges())
        # print('g.edges:', g.edges(form='all'))
        g = g.int()
        self._bg = dgl.to_bidirected(g, copy_ndata=True)
        print("_bg original node type:", self._bg.ndata)
        # self._hg = shg
        # self._bg = dgl.reorder_graph(bg)
        
        """ prepare the feature dict """
        # delete the dummy features in device nodes
        if shg.nodes['device'].data['x'].shape[1] > 15:
            # print("self._d_feat shape", self._d_feat.shape)
            d_feat = torch.cat((shg.nodes['device'].data['x'][:, 0].view(-1, 1), 
                                      shg.nodes['device'].data['x'][:, 4:-1]), dim=1).float()
        else:
            d_feat = shg.nodes['device'].data['x'][:, :-1]
        self._d_feat_dim = d_feat.shape[1]
        assert self._d_feat_dim == 14

        n_feat = shg.nodes['net'].data['x'][:, :-1].float().float()
        self._n_feat_dim = n_feat.shape[1]
        i_feat = shg.nodes['inst'].data['x'].float()
        self._i_feat_dim = i_feat.shape[1]

        max_feat_dim = max(self._n_feat_dim, self._i_feat_dim, self._d_feat_dim)
        d_feat = torch.cat((d_feat, torch.zeros(self._num_d, max_feat_dim-self._d_feat_dim)), dim=1)
        n_feat = torch.cat((n_feat, torch.zeros(self._num_n, max_feat_dim-self._n_feat_dim)), dim=1)
        i_feat = torch.cat((i_feat, torch.zeros(self._num_i, max_feat_dim-self._i_feat_dim)), dim=1)
        # print("d_feat size:", d_feat.shape)
        # print("i_feat size:", i_feat.shape)
        ## ntype order is matter!
        self._bg.ndata['x'] = torch.cat((d_feat, i_feat, n_feat), dim=0).float()
        # self._hg.nodes['net'].data['x'] = self._n_feat
        # self._hg.nodes['inst'].data['x'] = self._i_feat
        # self._hg.nodes['device'].data['x'] = self._d_feat

        """ prepare the labels """
        self._targets = shg.nodes['net'].data['y'].float() * 1000
        # self._hg.nodes['net'].data['y'] = self._targets
        self._bg.ndata['y'] = torch.cat((torch.zeros(self._num_d+self._num_i, 1), 
                                         self._targets), dim=0)
        self.plot_targets()
        max_l, max_i = torch.max(self._targets, dim=0)
        min_l, min_i = torch.min(self._targets, dim=0)
        print('targets size:', self._targets.shape) 
        print('max labels:{:.2f} with index:{:d}; min labels:{:.2f} with index:{:d}'
              .format(max_l.item(), max_i.item(), min_l.item(),min_i.item()))
        # classificating according to the magnitude of the net capacitance
        self._labels = None
        # self.get_labels()
        self._bg.ndata['label'] =  torch.cat((torch.ones((self._num_d+self._num_i, 1), dtype=torch.int32)*-1, 
                                              self.get_labels()), dim=0)

        """ prepare train/val masks """
        self._bg.ndata['train_mask'] = torch.zeros((self._bg.num_nodes()), dtype=torch.bool)
        self._bg.ndata['val_mask'] = torch.zeros((self._bg.num_nodes()), dtype=torch.bool)
        nids = [i for i in range(self._num_d+ self._num_i, self._bg.num_nodes())]

        if self.testDS:
            self._bg.ndata['train_mask'][torch.tensor(nids)] = True
            self._bg.ndata['val_mask'][torch.tensor(nids)] = True
            # print("targets:", self._bg.ndata['y'][self._bg.ndata['train_mask']])
        else:
            train_nids, val_nids = train_test_split(nids, test_size=0.2, 
                                                    random_state=42, stratify=self._labels)
            self._bg.ndata['train_mask'][train_nids] = True
            self._bg.ndata['val_mask'][val_nids] = True
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
        
        labels = torch.zeros(targets.shape, dtype=torch.int32)
        # binedges = [0.01, 0.1, 1.0, 10.0, 100.0, torch.inf]
        binedges = [0.01, 10.0, torch.inf] # change 5 classes into 2 classes
        # binedges = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
        class_num = []
        for i in range(len(binedges)-1):
            index = (targets >= binedges[i]) & (targets < binedges[i+1])
            labels[index] = i
            class_num.append(torch.sum(index.squeeze()).item())
            # avoid zero denominator
            if class_num[-1] == 0:
                class_num[-1] = 1
        
        self._labels = labels
        self._class_num = torch.tensor(class_num)
        self._num_classes = len(class_num)
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


class SRAMDatasetList():
    def __init__(self, nameList, device=torch.device('cuda:0'), test_ds="False", featMax={}):
        super(SRAMDatasetList).__init__()
        raw_dir = '/data1/shenshan/SPF_examples_cdlspf/Python_data/'
        datasets = []
        for name in nameList:
            datasets.append(SRAMDataset(name=name, raw_dir=raw_dir, test=test_ds))
                        # SRAMDataset(name='sandwich', raw_dir=raw_dir),
                        # # SRAMDataset(name='sram_sp_8192w', raw_dir=raw_dir),
                        # SRAMDataset(name='ssram', raw_dir=raw_dir),
                        # ]
        # else:
        #     datasets = [SRAMDataset(name='sram_sp_8192w', raw_dir=raw_dir, test=True)]
            # datasets = [SRAMDataset(name='ssram', raw_dir=raw_dir),
            #             SRAMDataset(name='array_128_32_8t', raw_dir=raw_dir),
            #             SRAMDataset(name='8T_digitized_timing_top_fast', raw_dir=raw_dir)]
        assert len(datasets)
        self.name = ""
        self.bgList = []
        self.featMax = featMax
        self._num_n = []
        self._num_d = []
        self._num_i = []
        self._tot_num_n = 0
        self.num_nodes = 0
        self._num_classes = datasets[0]._num_classes
        self.class_distr = torch.zeros((self._num_classes))
        # tot_nodes = 0
        for i, ds in enumerate(datasets):
            self.name += ds.name + "_"
            ds._bg.ndata['_GID'] = torch.ones(ds._bg.num_nodes(), dtype=torch.int) * i
            self.bgList.append(ds._bg)
            self._num_i.append(ds._num_i)
            self._num_d.append(ds._num_d)
            self._num_n.append(ds._num_n)
            self.class_distr += ds._class_num
            self.num_nodes += ds._num_i + ds._num_d + ds._num_n
            self._tot_num_n += ds._num_n

        self.bgs = dgl.batch(self.bgList)
        self._d_feat_dim = datasets[0]._d_feat_dim
        self._n_feat_dim = datasets[0]._n_feat_dim
        self._i_feat_dim = datasets[0]._i_feat_dim
        self.feat_max_norm(test_ds=test_ds)
        # assert 0
        # the alpha used in focal loss, which is the recipcal of the class fraction
        self.alpha =  self.class_distr.sum() / self.class_distr
        self.bgs = self.bgs.to(device)
        if test_ds:
            print("Test datasets are loaded.")
        else:
            print("Training datasets are loaded.")

    def num_nodes(self):
        return self.num_nodes
    '''
    def feat_max_norm(self, test_ds = False, featMax={}):     
        # normalize the feature with the maximum values
        def feat_norm(x, x_max):
            if len(x) < 1:
                return x, x
            # print("max_idx=", max_idx)
            # avoid zero denominators
            x_max[x_max == 0.0] = torch.inf
            x_norm = x / x_max.expand(x.shape[0], -1)
            return x_norm.float(), x_max
        
        # find self.featMax among the training dataset
        if test_ds == False:
            feat_max = {'device': torch.empty(0, self._d_feat_dim), 
                        'inst': torch.empty(0, self._i_feat_dim), 
                        'net': torch.empty(0, self._n_feat_dim)}

            for key in feat_max.keys():
                for fd in self.featDictList:
                    tmp_max, _ = torch.max(fd[key], dim=0)
                    feat_max[key] = torch.cat((tmp_max.view(1, -1), feat_max[key]), dim=0)
                self.featMax[key], _ = feat_max[key].max(dim=0)
                print("feat:", key, "max:", self.featMax[key])
        else:
            if len(featMax) == 0:
                raise Exception("No feature max vector found for the test dataset!")
            self.set_featMax(featMax)
            
        ### prepare the feature dict ###
        for fd in self.featDictList:
            for key in self.featMax.keys():
                fd[key], _ = feat_norm(fd[key], self.featMax[key])

        for fd in self.featDictList:
            for key in self.featMax.keys():
                print("ntype:", key, "feat:", fd[key])

        # if not test_ds:
        #     self.featDict['device'], self.d_feat_max = feat_norm(self.featDict['device'][:, :-1])
        #     self.featDict['net'], self.n_feat_max = feat_norm(self.featDict['net'][:, :-1])
        #     self.featDict['inst'], self.i_feat_max = feat_norm(self.featDict['inst'])
        # else:
        #     self.featDict['device'], _ = feat_norm(self.featDict['device'][:, :-1], self.d_feat_max)
        #     self.featDict['net'], _ = feat_norm(self.featDict['net'][:, :-1], self.n_feat_max )
        #     self.featDict['inst'], _ = feat_norm(self.featDict['inst'], self.i_feat_max)
    '''
    """
    def feat_max_norm(self, test_ds=False):     
        # normalize the feature with the maximum values
        def feat_norm(g_batch, feat_max):
            for ntype in g_batch.ntypes:
                x_max = feat_max[ntype]
                x = g_batch.nodes[ntype].data['x'] 
                x_max[x_max == 0.0] = torch.inf
                x_norm = x / x_max.expand(x.shape[0], -1)
                g_batch.nodes[ntype].data['x'] = x_norm.float()
            print("norm feat:", g_batch.ndata['x'])
        
        def find_max(g_batch):
            feat_max = {}
            for ntype in g_batch.ntypes:
                feat_max[ntype], _ = g_batch.nodes[ntype].data['x'].max(dim=0)
            print("feat_max:", feat_max)
            return feat_max
        
        if test_ds:
            assert len(self.featMax)
            feat_norm(self.hgs, self.featMax)
        else:
            assert len(self.featMax) == 0
            self.featMax = find_max(self.hgs)
            feat_norm(self.hgs, self.featMax)
    """
    
    def feat_max_norm(self, test_ds=False):     
        # normalize the feature with the maximum values
        def feat_norm(g_batch, feat_max):
            print("ndata['x'] size:", g_batch.ndata['x'].shape)
            print("ndata['_TYPE'] size:", g_batch.ndata['_TYPE'].shape)
            for i, ntype in enumerate(["device", "inst", "net"]):
                x_max = feat_max[ntype]
                x_max[x_max == 0.0] = torch.inf
                x = g_batch.ndata['x'][g_batch.ndata['_TYPE'] == i]
                x_norm = x / x_max.expand(x.shape[0], -1)
                g_batch.ndata['x'][g_batch.ndata['_TYPE'] == i] = x_norm.float()
            print("norm feat:", g_batch.ndata['x'])
        
        def find_max(g_batch):
            feat_max = {}
            # keep the node type order
            for i, ntype in enumerate(["device", "inst", "net"]):
                tmp_feat = g_batch.ndata['x'][g_batch.ndata['_TYPE'] == i]
                feat_max[ntype], _ = tmp_feat.max(dim=0)
            print("feat_max:", feat_max)
            return feat_max
        
        if test_ds:
            assert len(self.featMax)
            feat_norm(self.bgs, self.featMax)
        else:
            assert len(self.featMax) == 0
            self.featMax = find_max(self.bgs)
            feat_norm(self.bgs, self.featMax)

    def set_featMax(self, x_max):
        self.featMax = x_max