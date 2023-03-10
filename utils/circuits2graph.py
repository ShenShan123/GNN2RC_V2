import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import numpy as np
import warnings
import datetime

def constructGraph(structList, struIdx, ### struct list and the index of the target subckt
                   id_dict, ### the global indeces (IDs) of nets and devices and subckt (also a instance)
                   portIDs, ### the instance' ports of this subckt connect the other instances
                            ### in outer level, so pass the global indeces of port nets to this subckt
                   edge_dict,
                   data_dict, ### feature data (n for 'net' d for 'device')
                   hier_lvl  ### the hierarchy level of this subckt
                   ):
    curInstID = id_dict["inst"]
    subckt = structList[0, struIdx - 1] ### struIdx starts from 1 in MATLAB
    print('In subckt:', subckt['name'][0], 'struIdx:', struIdx, 'curInstID:', curInstID)
    netList = subckt['nodeList']
    # print('netList shape:', netList.shape)
    instList = subckt['instList']
    # print('instList shape:', instList.shape)
    globalNtList = [-1] * netList.shape[1]
    globalDvList = [-1] * instList.shape[1]
    hier_lvl = hier_lvl + 1
    ix = extractSdata(subckt, hier_lvl)
    data_dict["inst"] = torch.cat((data_dict["inst"], ix), dim=0)
    #assert(hier_lvl != 3)
    portNum = subckt['portNum'][0, 0]
    ### assign an index for each net in this subckt
    for j in range(netList.shape[1]):
        net = netList[0, j]
        # If this net is a port of this subckt, it should have been assigned 
        # an ID in the outer level subckt when the current subckt is instantiated
        if j < portNum and len(portIDs) > 0:
            globalNtList[j] = portIDs[j]
        else:
            globalNtList[j] = id_dict["net"]
            # concatenate the old feature with the new feature hier_lvl
            nx = extractNdata(net, hier_lvl)
            ny = torch.tensor(np.array([feat_Y[id_dict["net"], 0]])).reshape([1, -1])
            #print('nx shape: ', nx.shape)
            data_dict["net_x"] = torch.cat((data_dict["net_x"], nx), dim=0)
            data_dict["net_y"] = torch.cat((data_dict["net_y"], ny), dim=0)
            
            id_dict["net"] = id_dict["net"] + 1

        ### record the net to subckt edges
        n2i_src, n2i_dst = edge_dict['n2i']
        n2i_src.append(globalNtList[j])
        n2i_dst.append(curInstID)
        # print('net:', net['name'][0], 'id:', globalNtList[j], \
        #       'ndata_x[:]:', ndata_x[globalNtList[j], [0, 1, -2, -1]],  \
        #       'ndata_y: {:.6f}'.format(ndata_y[globalNtList[j]].item()))

    ### Here we handle the instance that could be a device or an instance defined by a subckt
    for l in range(instList.shape[1]):
        inst = instList[0, l]
        subStruIdx = inst['subPtr'][0, 0]  ### subStruIdx starts from 1 in MATLAB
        ntPtr = inst['ndPtr']
        ### this is a terminal device
        if subStruIdx == 0:
            assert(inst['type'] != 'Subckt')
            globalDvList[l] = id_dict["device"]
            ### concatenate the old feature with the new feature hier_lvl
            dx = extractDdata(inst, hier_lvl)
            data_dict["device"] = torch.cat((data_dict["device"], dx), dim=0)
            # print('device:', inst['name'][0], 'id:', globalDvList[l], \
            #       'ddata_x[:]:', ddata_x[globalDvList[l], [0, -2, -1]])

            # Store the edges from device to net
            d2n_src, d2n_dst = edge_dict['d2n']
            for i in range(ntPtr.shape[1]):
                d2n_src.append(globalDvList[l])
                d2n_dst.append(globalNtList[ntPtr[0, i] - 1]) ### ntPtr starts from 1 in MATLAB
            
            # Store the edges between device and instance
            d2i_src, d2i_dst = edge_dict['d2i']
            d2i_src.append(globalDvList[l])
            d2i_dst.append(curInstID)

            id_dict["device"] = id_dict["device"] + 1
        ### This is an instance of subckt
        else:
            subPortIDs = [-1] * ntPtr.shape[1]
            # Take the global net index to the internal net of this instance
            for i in range(ntPtr.shape[1]):
                subPortIDs[i] = globalNtList[ntPtr[0, i] - 1] # ntPtr starts from 1 in MATLAB
            # Store the instance edge that comes from the current subckt to its instance component 
            i2i_src, i2i_dst = edge_dict['i2i']
            i2i_dst.append(curInstID)
            id_dict["inst"] = id_dict["inst"] + 1
            i2i_src.append(id_dict["inst"])

            print('enter inst:', inst['name'][0],' defined by the subckt:', \
                  structList[0, subStruIdx - 1]['name'][0], 'with instID:', \
                  id_dict["inst"], 'and port indx', subPortIDs)
            
            sbInstID = constructGraph(structList, subStruIdx, 
                                      id_dict,
                                      subPortIDs, 
                                      edge_dict,
                                      data_dict,
                                      hier_lvl)

            print('return from subckt: ', 
                  structList[0, subStruIdx - 1]['name'][0],
                  'with inst name:', inst['name'][0], 'and sbInstID:', sbInstID)
            
    return curInstID

def extractSdata(sbckt, lvl):
    #print('extractNdata sbckt:', sbckt['name'][0])
    # print('portNum:', sbckt['portNum'])
    x = torch.tensor(np.array([
        sbckt['portNum'][0, 0],
        sbckt['instList'].shape[1],
        sbckt['nodeList'].shape[1],
        1.0 / lvl
    ]))

    return x.reshape((1, -1))

def extractNdata(net, lvl):
    #print('extractNdata net:', net['name'][0])
    x = torch.tensor(np.array([
        # for MOS and MOScap
        net['num_mos'][0, 0], #0
        net['num_mos_g'][0, 0],
        net['num_mos_sd'][0, 0],
        net['num_mos_b'][0, 0],
        net['tot_w'][0, 0],
        net['tot_l'][0, 0], #6
        # for rupolym
        net['num_r'][0, 0], #7
        net['tot_r_w'][0, 0],
        net['tot_r_l'][0, 0], #9
        # for cfmom
        net['num_c'][0, 0],
        net['tot_nr'][0, 0],
        net['tot_lr'][0, 0],
        # for dio
        net['num_d'][0, 0],
        net['tot_area'][0, 0],
        net['tot_pj'][0, 0],
        # for other info
        net['port'][0, 0], # bool flag for port net
        1.0 / lvl # NOTE: should be deleted in the graph training
        #net['cell'][0, 0] # bool flag for a SRAM cell net, new feature added 2022-8-16
    ]))

    #y = torch.tensor([[net['cap'][0, 0]],], dtype=torch.float64) # for cap training and prediction
    #print('y cap=', net['cap'][0, 0])
    return x.reshape((1, -1))

def extractDdata(dvc, lvl):
    #print('extractDdata dvc:', dvc['name'][0])
    type_code = 0
    num_mos = 0
    num_r = 0
    num_c = 0
    num_d = 0
    w = 0.0 
    l = 0.0

    if dvc['type'] == 'MOS':
        num_mos = dvc['m'][0, 0]
        type_code = 1
        w = dvc['w'][0, 0]
        l = dvc['l'][0, 0]
    elif dvc['type'] == 'cell_MOS':
        num_mos = dvc['m'][0, 0]
        type_code = 2
        w = dvc['w'][0, 0]
        l = dvc['l'][0, 0]
    elif dvc['type'] == 'Rupolym':
        num_r = dvc['m'][0, 0]
        type_code = 3
    elif dvc['type'] == 'Cfmom':
        num_c = dvc['m'][0, 0]
        type_code = 4
    elif dvc['type'] == 'Moscap':
        num_mos = dvc['m'][0, 0]
        type_code = 1
        w = dvc['wr'][0, 0]
        l = dvc['m_lr'][0, 0]
    elif dvc['type'] == 'Ndio':
        num_d = dvc['m'][0, 0]
        type_code = 5

    x = torch.tensor(np.array([
        # for MOS and MOScap
        num_mos,
        num_mos * dvc['nf'][0, 0], # corresponding to num_mos_g
        num_mos * dvc['nf'][0, 0], # corresponding to num_mos_sd
        num_mos, # corresponding to num_mos_d
        w,
        l,
        # for rupoly
        num_r,
        dvc['r_w'][0, 0],
        dvc['r_l'][0, 0],
        # for cfmom
        num_c,
        dvc['nr'][0, 0],#6
        dvc['lr'][0, 0],
        # for dio
        num_d,
        dvc['area'][0, 0],
        dvc['pj'][0, 0],
        # 
        len(dvc['ndPtr'][0,:]), # num of ports of this dvc
        type_code,
        1.0 / lvl # NOTE: should be deleted in the graph training
    ]))
    #print('dvc ntPtr=', dvc['ndPtr'][0,:])
    return x.reshape((1, -1))

if __name__ == '__main__':
    start = datetime.datetime.now()
    print('start circuit2graph_V4 at', start)
    """ load subcircuits and transfer to a dgl-graph """
    name = "sram_sp_8192w" #'ultra_8T' # "ultra_8T" #
    dataPath = '/data1/shenshan/RCpred/' + name + '.sbckt.dat'
    ### This file stores the orignal feature and the capacitance, which is Y.
    dataPath2 = '/data1/shenshan/RCpred/' + name + '_XY.mat'
    data = scipy.io.loadmat(dataPath)
    data2 = scipy.io.loadmat(dataPath2)
    print('type of structList:', type(data['structList']))
    print('shape of structList:', data['structList'].shape)
    structList = data['structList']
    feat_X = data2['X']
    print('shape of X:', feat_X.shape)
    feat_Y = data2['Y']
    print('shape of Y:', feat_Y.shape)
    # assert 0
    subcktName = 'sram_sp_8192w' #"ultra_8T_macro" # 'pe_macro2' # 'RVBx3' #
    # subcktName = "DELAY4_PULSE" # "sa_write" #"SA" #
    edge_dict = {'d2n': ([], []), 'd2i':([], []), 'n2i':([], []), 'i2i':([], [])}
    data_dict = {}
    data_dict["net_x"] = torch.zeros((0, 17), dtype=torch.float64)
    data_dict["net_y"] = torch.zeros((0, 1), dtype=torch.float64)
    data_dict["device"] = torch.zeros((0, 18), dtype=torch.float64)
    data_dict["inst"] = torch.zeros((0, 4), dtype=torch.float64)
    for i in range(structList.shape[1]):
        if structList[0, i]['name'][0] == subcktName: #'ana_buff': #
            struIdx = i + 1
            print('find subckt', i) #BUFF_PULSE
            constructGraph(structList, struIdx, 
                        {"inst":0, "net":0, "device":0},
                        [], edge_dict, data_dict, 0)

            for key in edge_dict:
                src, dst = edge_dict[key]
                print(key + "_src:", src[:100])
                print(key + "_dst:", dst[:100])
            print('ndata_x shape: ', data_dict["net_x"].shape)
            print('ndata_y shape: ', data_dict["net_y"].shape)
            print('ddata_x shape: ', data_dict["device"].shape)
            print('idata_x shape: ', data_dict["inst"].shape)
            if feat_Y.shape[0] != data_dict["net_x"].shape[0]:
                warnings.warn("Number of caps doesn't match the number of nets in sub-circuits struct, leading to wrong labels of nodes in the graph !! ",
                            RuntimeWarning)
            # assert 0
            print('constructing dgl graph ...')
            d2n = torch.tensor(edge_dict['d2n'])
            d2i = torch.tensor(edge_dict['d2i'])
            n2i = torch.tensor(edge_dict['n2i'])
            i2i = torch.tensor(edge_dict['i2i'])
            graph_data = {
                ('device', 'device-net', 'net'): (d2n[0], d2n[1]),
                ('device', 'device-inst', 'inst'): (d2i[0], d2i[1]),
                ('net', 'net-inst', 'inst'): (n2i[0], n2i[1]),
                ('inst', 'inst-inst', 'inst'): (i2i[0], i2i[1]),
            }
            hg = dgl.heterograph(graph_data)
            assert(hg.num_nodes('net') == data_dict["net_x"].shape[0])
            assert(hg.num_nodes('net') == data_dict["net_y"].shape[0])
            assert(hg.num_nodes('device') == data_dict["device"].shape[0])
            assert(hg.num_nodes('inst') == data_dict["inst"].shape[0])
            hg.nodes['net'].data['x'] = data_dict["net_x"]
            hg.nodes['net'].data['y'] = data_dict["net_y"]
            hg.nodes['device'].data['x'] = data_dict["device"]
            hg.nodes['inst'].data['x'] = data_dict["inst"]
            print('hg: ', hg)
            gFilePath = "/data1/shenshan/RCpred/" + name + ".bi_graph.bin"
            dgl.data.utils.save_graphs(gFilePath, [hg])
            print('saved graph hg to', gFilePath)
            break
    end = datetime.datetime.now()
    print('all finished at', end, ' with runtime', (end-start).seconds / 3600, 'h')