from read_cdl import read_cdl
from build_graph import build_graph
import os
import warnings

def findInterNetName(netCapDict: dict(), subList:list(), subIdx: int, 
                      prefix : str): 
    # This function search each node and sub-node in this subckt
    # netCapDict - dict contains net name: net cap
    # subList - the list of subckt in the cdl file
    # subIdx - the subckt that needs to be extracted node info
    # prefix -  the instance name and the parent name
    
    subckt = subList[subIdx]
    if len(prefix) == 0:
        ntStartIdx = 0
    else:
        ntStartIdx = subckt.portNum 
        prefix = prefix+'/'
    
    # record the nets' info in this subckt
    for net in subckt.netList[ntStartIdx:]:
        # record the name and cap for different insts of this subckt
        net.postNameList.append((prefix+net.name).upper())
        if net.postNameList[-1] in netCapDict.keys():
            if net.isempty():
                print("encounter a empty net:", net)
                raise Exception('Net: %s postname: %s in subckt: %s is empty!!'
                                %(net.name, net.postNameList[-1], subckt.name))
            net.capList.append(netCapDict[net.postNameList[-1]])
            # y.append(node.cap)
            # print('node post name: %s cap:%f' % (node.postName, netDict[node.postName]))
        else:
            warnings.warn('cannot find postname: %s name:%s' %(net.postNameList[-1], net.name))
            net.capList.append(-1.0)
        assert len(net.capList) == len(net.postNameList)
        # if len(net.postNameList[-1])>0:
        #     print(net)
    
    # record the internal nodes' info of each instance in this subckt
    for device in subckt.deviceList:
        assert(str(device.type) == str('Cfmom') or str(device.type) == str('Rupolym') 
               or str(device.type) == str('MOS') or str(device.type) == str('Subckt') 
               or str(device.type) == str('Ndio') or str(device.type) == str('Moscap'))
        if not str(device.type) == str('Subckt') :
            continue
        # find the definition of this instance
        cSubIdx = device.subPtr
        assert cSubIdx >= 0
        findInterNetName(netCapDict, subList, cSubIdx, prefix+device.name)

def read_spf(filename):
    # Read the spf file and find the parasitic capacitance for each net
    # filename - spf file path
    # return - the dict of {name: cap} pairs
    spf_net_dict = dict()
    os.system("grep '|NET .*PF' "+ filename + " > nets.log")
    try:
        with open('./nets.log', 'r') as f:
            lines = f.readlines()
    except:
        raise Exception('Cannot open file %s'%filename)
    
    for line in lines:
        tokens = line.strip('*|NET').strip()
        # if re.search('EN\[', line):
        #     print(tokens)
        tokens = tokens.split(' ')
        spf_net_dict[tokens[0].upper()] = float(tokens[1].strip('PF\n'))
        
        # print(tokens[0] + " " + str(spf_net_dict[tokens[0]]))
    print('spf read finished!')
    return spf_net_dict
    
    # for i, item in enumerate(spfnets):
    #     

if __name__ == '__main__':
    moduleName = 'ssram' #'pe_macro2' #'ultra_8T_macro' #'sram_sp_8192w' #'array_128_32_8t' #'8T_digitized_timing_top_fast' # ''ultra_8T_macro' # 
    subcktList, topSubIdx = read_cdl('/data1/shenshan/SPF_examples_cdlspf/CDL_files/SSRAM/'+moduleName+'.cdl', moduleName)
    # subIdx = -1
    # # moduleName = 'front_8T_digitized_timing_top_256_fast'
    # for subIdx, subckt in enumerate(subcktList):
    #     if subckt.name == moduleName:
    #         # print('find subckt name: %s %d' % ('vref', ))
    #         break
    
    assert topSubIdx != -1
    # targets = list()
    # assert 0
    print('Reading SPF file...')
    # moduleName = '8T_digitized_timing_top_fast'
    net_cap_dict = read_spf('/data1/shenshan/SPF_examples_cdlspf/SPF_files/SSRAM/'+moduleName+'.spf')
    assert len(net_cap_dict)
    
    print('Calling findInterNetName...')
    findInterNetName(net_cap_dict, subcktList, topSubIdx, '')
    # assert 0
    tokenizer = AutoTokenizer.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2")
    
    print('Converting to a graph...')
    build_graph(subcktList, topSubIdx)
