import dgl
import torch
import numpy as np
import warnings
import datetime

def constructGraph(structList, 
                   struIdx, ### struct list and the index of the target subckt
                   id_dict, ### the global indeces (IDs) of nets and devices and subckt (also a instance)
                   portIDs, ### the instance' ports of this subckt connect the other instances
                            ### in outer level, so pass the global indeces of port nets to this subckt
                   edge_dict,
                   data_dict, ### feature data (n for 'net' d for 'device')
                   hier_lvl,  ### the hierarchy level of this subckt
                #    instName=None
                   ):
    curInstID = id_dict["inst"]
    subckt = structList[struIdx]
    print('In subckt:', subckt.name, 'struIdx:', struIdx, 'curInstID:', curInstID)
    # print('netList shape:', netList.shape)
    # print('deviceList shape:', deviceList.shape)
    globalNtList = [-1] * len(subckt.netList)
    globalDvList = [-1] * len(subckt.deviceList)
    hier_lvl = hier_lvl + 1
    ix = extractSdata(subckt, hier_lvl)
    data_dict["inst"] = torch.cat((data_dict["inst"], ix), dim=0)
    #assert(hier_lvl != 3)
    if len(portIDs) and (subckt.portNum != len(portIDs)):
        raise Exception('port num %d does not match the len of portIDs %d'
                        %(subckt.portNum, len(portIDs)))
    ### assign an index for each net in this subckt
    for j, net in enumerate(subckt.netList):
        # If this net is a port of this subckt, it should have been assigned 
        # an ID in the outer level subckt when the current subckt is instantiated
        if j < subckt.portNum and len(portIDs) > 0:
            globalNtList[j] = portIDs[j]
        else:
            globalNtList[j] = id_dict["net"]
            # concatenate the old feature with the new feature hier_lvl
            nx = extractNdata(net, hier_lvl)
            # the target net cap extracted from spf file
            if len(net.capList):
                ny = torch.tensor(net.capList.pop(0)).view(-1, 1)
                # nt = np.array(targets[id_dict["net"]])
                # if len(net.capList):
                #     print('cap after pop:', net.capList)
            else :
                ny = torch.tensor([[-1.0],])
                raise warnings.warn('No matching net cap!! net: %s subckt:%s'%
                                    (net.name, subckt.name))
                # nt = np.array(targets[id_dict["net"]])
            #print('nx shape: ', nx.shape)
            data_dict["net_x"] = torch.cat((data_dict["net_x"], nx), dim=0)
            data_dict["net_y"] = torch.cat((data_dict["net_y"], ny), dim=0)
            
            # if instName is not None:
            #     netName = '.'.join([instName, net.name])
            # else:
            #     netName = net.name
            
            # # write a pre-layout netlist with pred/target net capacitance
            # if targets is not None:
            #     with open('pred_c.cdl', 'a') as c_file:
            #         cap = "C" + str(id_dict["net"])
            #         line = ' '.join([cap, netName, "0", str(ny.item())+"f", "\n"])
            #         c_file.write(line)
            # print("netName:", netName)
            id_dict["net"] = id_dict["net"] + 1

        ### record the net to subckt edges
        n2i_src, n2i_dst = edge_dict['n2i']
        n2i_src.append(globalNtList[j])
        n2i_dst.append(curInstID)
        # print('net:', net['name'][0], 'id:', globalNtList[j], \
        #       'ndata_x[:]:', ndata_x[globalNtList[j], [0, 1, -2, -1]],  \
        #       'ndata_y: {:.6f}'.format(ndata_y[globalNtList[j]].item()))

    ### Here we handle the instance that could be a device or an instance defined by a subckt
    for l, device in enumerate(subckt.deviceList):
        subStruIdx = device.subPtr 
        ntPtr = device.ntPtr
        ### this is a terminal device
        if subStruIdx == -1:
            assert(device.type != 'Subckt')
            globalDvList[l] = id_dict["device"]
            ### concatenate the old feature with the new feature hier_lvl
            dx = extractDdata(device, hier_lvl)
            data_dict["device"] = torch.cat((data_dict["device"], dx), dim=0)
            # print('device:', inst['name'][0], 'id:', globalDvList[l], \
            #       'ddata_x[:]:', ddata_x[globalDvList[l], [0, -2, -1]])

            # Store the edges from device to net
            d2n_src, d2n_dst = edge_dict['d2n']
            for pointer in ntPtr:
                d2n_src.append(globalDvList[l])
                d2n_dst.append(globalNtList[pointer])
            
            # Store the edges between device and instance
            d2i_src, d2i_dst = edge_dict['d2i']
            d2i_src.append(globalDvList[l])
            d2i_dst.append(curInstID)

            id_dict["device"] = id_dict["device"] + 1
        ### This is an instance of subckt
        else:
            assert(device.type == 'Subckt')
            # # record the name of this subcircuit instance
            # # this is for naming the internal nets in this subcircuit instance
            # if instName is not None:
            #     subInstName = '.'.join([instName, device['name'][0]])
            # else:
            #     subInstName = device['name'][0]
            subPortIDs = [-1] * len(ntPtr)
            # Take the global net index to the internal net of this instance
            for i, pointer in enumerate(ntPtr):
                subPortIDs[i] = globalNtList[pointer]
            # Store the instance edge that comes from the current subckt to its instance component 
            i2i_src, i2i_dst = edge_dict['i2i']
            i2i_dst.append(curInstID)
            id_dict["inst"] = id_dict["inst"] + 1
            i2i_src.append(id_dict["inst"])

            print('enter inst:', device.name,' defined by the subckt:', \
                  structList[subStruIdx].name, 'with instID:', \
                  id_dict["inst"], 'and port indx', subPortIDs)

            sbInstID = constructGraph(structList,
                                      subStruIdx, 
                                      id_dict,
                                      subPortIDs, 
                                      edge_dict,
                                      data_dict,
                                      hier_lvl)

            # print('return from subckt: ', 
            #       structList[subStruIdx].name,
            #       'with inst name:', device.name, 'and sbInstID:', sbInstID)
            
    return curInstID

def extractSdata(sbckt, lvl):
    #print('extractNdata sbckt:', sbckt['name'][0])
    # print('portNum:', sbckt['portNum'])
    x = torch.tensor(np.array([
        sbckt.portNum,
        len(sbckt.deviceList),
        len(sbckt.netList),
        1.0 / lvl
    ]))

    return x.reshape((1, -1))

def extractNdata(net, lvl):
    #print('extractNdata net:', net.name'][0])
    x = torch.tensor(np.array([
        # for MOS and MOScap
        net.num_mos, #0
        net.num_mos_g,
        net.num_mos_sd,
        net.num_mos_b,
        net.tot_w,
        net.tot_l, #6
        # for rupolym
        net.num_r, #7
        net.tot_r_w,
        net.tot_r_l, #9
        # for cfmom
        net.num_c,
        net.tot_nr,
        net.tot_lr,
        # for dio
        net.num_d,
        net.tot_area,
        net.tot_pj,
        # for other info
        net.port, # bool flag for port net
        1.0 / lvl # NOTE: should be deleted in the graph training
        #net.cell # bool flag for a SRAM cell net, new feature added 2022-8-16
    ]))

    #y = torch.tensor([[net.cap],], dtype=torch.float32) # for cap training and prediction
    #print('y cap=', net.cap)
    return x.reshape((1, -1))

def extractDdata(dvc, lvl):
    #print('extractDdata dvc:', dvc.name'][0])
    type_code = 0
    num_mos = 0
    num_r = 0
    num_c = 0
    num_d = 0
    w = 0.0 
    l = 0.0

    if dvc.type=='MOS':
        num_mos = dvc.m
        type_code = 1
        w = dvc.w
        l = dvc.l
    elif dvc.type=='cell_MOS':
        num_mos = dvc.m
        type_code = 2
        w = dvc.w
        l = dvc.l
    elif dvc.type=='Rupolym':
        num_r = dvc.m
        type_code = 3
    elif dvc.type=='Cfmom':
        num_c = dvc.m
        type_code = 4
    elif dvc.type=='Moscap':
        num_mos = dvc.m
        type_code = 1
        w = dvc.wr
        l = dvc.m_lr
    elif dvc.type=='Ndio':
        num_d = dvc.m
        type_code = 5

    x = torch.tensor(np.array([
        # for MOS and MOScap
        num_mos,
        num_mos * dvc.nf, # corresponding to num_mos_g
        num_mos * dvc.nf, # corresponding to num_mos_sd
        num_mos, # corresponding to num_mos_d
        w,
        l,
        # for rupoly
        num_r,
        dvc.r_w,
        dvc.r_l,
        # for cfmom
        num_c,
        dvc.nr,#6
        dvc.lr,
        # for dio
        num_d,
        dvc.area,
        dvc.pj,
        # 
        len(dvc.ntPtr), # num of ports of this dvc
        type_code,
        1.0 / lvl # NOTE: should be deleted in the graph training
    ]))
    #print('dvc ntPtr=', dvc.ndPtr'][0,:])
    return x.reshape((1, -1))

# if __name__ == '__main__':
def run_cir2g(subcktList, subIdx):
    start = datetime.datetime.now()
    print('start circuit2graph_V4 at', start)
    edge_dict = {'d2n': ([], []), 'd2i':([], []), 'n2i':([], []), 'i2i':([], [])}
    data_dict = {}
    data_dict["net_x"] = torch.zeros((0, 17), dtype=torch.float32)
    data_dict["net_y"] = torch.zeros((0, 1), dtype=torch.float32)
    data_dict["device"] = torch.zeros((0, 18), dtype=torch.float32)
    data_dict["inst"] = torch.zeros((0, 4), dtype=torch.float32)
    
    constructGraph(subcktList, subIdx, 
                {"inst":0, "net":0, "device":0},
                [], edge_dict, data_dict, 0)
    
    for key in edge_dict.keys():
        src, dst = edge_dict[key]
        print(key + "_src:", src[:100])
        print(key + "_dst:", dst[:100])
    print('ndata_x shape: ', data_dict["net_x"].shape)
    print('ndata_y shape: ', data_dict["net_y"].shape)
    print('ddata_x shape: ', data_dict["device"].shape)
    print('idata_x shape: ', data_dict["inst"].shape)

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
    gFilePath = "/data1/shenshan/SPF_examples_cdlspf/Python_data/" + \
                 subcktList[subIdx].name + ".bi_graph.bin"
    dgl.data.utils.save_graphs(gFilePath, [hg])
    print('saved graph hg to', gFilePath)
    end = datetime.datetime.now()
    print('all finished at', end, ' with runtime', str(end-start), 'h')