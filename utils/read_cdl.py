import numpy as np
from subckt import Subckt, Net, Device
import re
import pickle
import warnings
import copy

def handle_wrapper_lines(filename):
    try:
        fpn = open(filename,'rt')
    except:
        raise Exception('Can not read file %s'%filename)
    
    print('filename:', filename)
    tokens = filename.split('/')
    dirpath = filename.split(tokens[-1])
    dirpath = dirpath[0]
    
    lines = fpn.readlines()
    fpn.close()
    print('# lines:', len(lines))
    newLines = []
    
    for i, line in enumerate(lines):
        ## ==================== text handling =============================
        # content after a '*' is a comment in SPICE
        line = line.split('*')[0].strip()
        line = line.split('$')[0].strip()
        if len(line)==0:
            continue
        if re.match(r'\.param', line, re.I):
            continue

        # handle the line wrapper
        if re.match('\+', line):
            assert len(newLines)
            line = newLines[-1]+' '+line[1:].strip()
            newLines[-1] = line.strip()
            # print('newLines:', newLines[-1])
            continue

        # find the include file
        if re.match('.inc', line, re.I):
            assert (i < len(lines)-1) and (not re.match('\+', lines[i+1].strip()))
            tokens = line.split(' ')[1].strip("'")
            tokens = tokens.split('/')
            # default dirpath
            incLines = handle_wrapper_lines(dirpath+tokens[-1])
            newLines = newLines + incLines
            continue
        
        newLines.append(line)
    return newLines

def read_cdl(filename, topname): 
    # This function convert a cdl file to a list of python class
    # filename: the cdl file name
    # subcktList: a list of Subckt , can be None
    subcktList = list()
    lines = handle_wrapper_lines(filename)
    topSubcktIdx = -1
    # assert 0
    for i, line in enumerate(lines):
        # print(line)
        # continue
        ## =================== construct a subckt =========================
        if re.match('^.subckt',line, re.I):
            tokens = line.split()
            subckt = Subckt()
            subckt.name = tokens[1]
            subckt.portNum = len(tokens) - 2
            for i in range(2, len(tokens)):
                net = Net()
                net.name = tokens[i]
                net.port = 1
                subckt.netList.append(net)
            continue
        ## =============== save the subckt to the list ====================
        if re.match('.ends', line, re.I):
            # copy params of each instance to the param list of subckt
            for device in subckt.deviceList:
                for paramk, paramv in device.params.items():
                    if paramk:
                        subckt.params[paramk] = paramv
            # record the top module index
            if subckt.name == topname:
                topSubcktIdx = len(subcktList) 
            # push this subckt to subcktList
            subcktList.append(subckt)
            print('push subckt:%s with %d instances and %d nets into subcktList with id %d\n' 
                  % (subckt.name,len(subckt.deviceList),len(subckt.netList), 
                     len(subcktList)-1))
            continue
        ## we split the line text
        newtokens = []
        tokens = re.split(r'[ /\t]', line)
        for token in tokens:
            if token != '':
                newtokens.append(token)

        tokens = newtokens
        # print('tokens:', tokens)
        modelNameIdx = 0
        subIdx = []
        
        paramKeyVals = dict()
        ## distinguish the instance types according to a equal sign
        if re.search('=', line):
            # we find a equal sign, so this is a device instance
            paramFlag = list()
            for j, token in enumerate(tokens):
                if '=' in token:
                    paramFlag.append(j)
                    [key, value] = token.split('=')
                    # we transform the parameter unit here
                    if value[-1] in ['u', 'U'] and re.match('[0-9]', value):
                        value = float(value[:-1]+'E-06')
                    elif value[-1] in ['n', 'N'] and re.match('[0-9]', value):
                        value = float(value[:-1]+'E-09')
                    elif value[-1] in ['p', 'P'] and re.match('[0-9]', value):
                        value = float(value[:-1]+'E-12')
                    elif re.match('[0-9]+', value):
                        value = float(value)
                    paramKeyVals[key.lower()] = value
            modelNameIdx = paramFlag[0] - 1
            assert modelNameIdx > 0
            modelName = tokens[modelNameIdx]
            # print('paramKeyVals:', paramKeyVals)
        else:
            # otherwise it is a instance of a pre-defined subckt
            modelName = tokens[-1]
            modelNameIdx = len(tokens) - 1
        ## === now we handle the instance in the subckt, all its info is in this line ===
        device = Device()
        device.name = tokens[0]
        device.subName = modelName
        value = 0
        # handle the device model parameters
        # this section is technology dependent, differnt techs have different
        # model names
        if 'm' in paramKeyVals.keys():
            if isinstance(paramKeyVals['m'], float):
                # this is a digital number
                device.m = paramKeyVals['m']
            else:
                device.params[paramKeyVals['m']] = 'm'
                # if the 'm' param is waiting to be defined, we set its value to 0 temporarily
                device.m = 0.0
        # handle the device model parameters and save them into the inst
        if modelName.lower() in ['pch_mac','nch_mac','pch_lvt_mac','nch_lvt_mac',
                                 'pch_ulvt_mac','nch_ulvt_mac','pch_hvt_mac',
                                 'nch_hvt_mac','pchpu_sr_mac','nchpd_sr_mac',
                                 'nchpg_sr_mac','nch_rpsr_mac','pchpu_sr',
                                 'nchpd_sr','nchpg_sr','nch_rpsr','nch_na_mac']:
            device.type = 'MOS'
            # find parameters
            if 'w' in paramKeyVals.keys():
                if isinstance(paramKeyVals['w'], float):
                # this is a digital number
                    device.w = paramKeyVals['w']
                else:
                    device.params[paramKeyVals['w']] = 'w'
            if 'l' in paramKeyVals.keys():
                if isinstance(paramKeyVals['l'], float):
                # this is a digital number
                    device.l = paramKeyVals['l']
                else:
                    device.params[paramKeyVals['l']] = 'l'
            if 'nf' in paramKeyVals.keys():
                if isinstance(paramKeyVals['nf'], float):
                # this is a digital number
                    device.nf = paramKeyVals['nf']
                else:
                    device.params[paramKeyVals['nf']] = 'nf'
        elif modelName.lower() == 'rupolym':
            device.type = 'Rupolym'
            # find parameters
            if 'w' in paramKeyVals.keys():
                if isinstance(paramKeyVals['w'], float):
                # this is a digital number
                    device.r_w = paramKeyVals['w']
                else:
                    device.params[paramKeyVals['w']] = 'r_w'
            if 'l' in paramKeyVals.keys():
                if isinstance(paramKeyVals['l'], float):
                # this is a digital number
                    device.r_l = paramKeyVals['l']
                else:
                    device.params[paramKeyVals['l']] = 'r_l'

        elif modelName.lower() in ['cfmom_wo','cfmom_2t']:
            device.type = 'Cfmom'
            # find parameters
            if 'nr' in paramKeyVals.keys():
                if isinstance(paramKeyVals['nr'], float):
                # this is a digital number
                    device.nr = paramKeyVals['nr']
                else:
                    device.params[paramKeyVals['nr']] = 'nr'
            if 'lr' in paramKeyVals.keys():
                if isinstance(paramKeyVals['lr'], float):
                # this is a digital number
                    device.lr = paramKeyVals['lr']
                else:
                    device.params[paramKeyVals['lr']] = 'lr'
                    
        elif modelName.lower() in ['nmoscap','pmoscap']:
            device.type = 'Moscap'
            # find parameters
            if 'wr' in paramKeyVals.keys():
                if isinstance(paramKeyVals['wr'], float):
                # this is a digital number
                    device.wr = paramKeyVals['wr']
                else:
                    device.params[paramKeyVals['wr']] = 'wr'
                    
            if 'lr' in paramKeyVals.keys():
                if isinstance(paramKeyVals['lr'], float):
                # this is a digital number
                    device.m_lr = paramKeyVals['lr']
                else:
                    device.params[paramKeyVals['lr']] = 'm_lr'
        
        elif modelName.lower() in ['ndio_ll', 'ndio_lvt']:
            device.type = 'Ndio'
            # find parameters
            if 'area' in paramKeyVals.keys():
                if isinstance(paramKeyVals['area'], float):
                # this is a digital number
                    device.area = paramKeyVals['area']
                else:
                    device.params[paramKeyVals['area']] = 'area'
            if 'pj' in paramKeyVals.keys():
                if isinstance(paramKeyVals['pj'], float):
                # this is a digital number
                    device.pj = paramKeyVals['pj']
                else:
                    device.params[paramKeyVals['pj']] = 'pj'
        # this case handle the inst that is a instantiate from a subckt
        else:
            device.type = 'Subckt'
            subIdx = -1
            # find which subckt the instance belongs to
            for k, sub in enumerate(subcktList):
                if device.subName == sub.name:
                    subIdx = k
            # record the reference of the current subckt, which is the
            # element to be push in subckList
            device.subPtr = subIdx

            if subIdx != -1:
                # check the port number, if it doesn't match, throw an error
                if subcktList[subIdx].portNum != modelNameIdx - 1:
                    print(line)
                    raise Exception('instance: {}, subckt: {}, port number mismatch:{:d} and {:d}'
                                    .format(filename, device.name,subcktList[subIdx].name,
                                            subcktList[subIdx].portNum, modelNameIdx - 1))
                # record the paramters that will be used in updateSubckt
                if len(paramKeyVals)!=0 :
                    assert (len(subcktList[subIdx].params) == len(paramKeyVals))
                    device.paramKeyVals = copy.deepcopy(paramKeyVals)
                    #subcktList[subIdx].update(paramKeyVals)
            else:
                warnings.warn('No definition! inst:{} with name:{} subckt:{}'
                              .format(device.name, device.subName, subckt.name))
                
                # to do :  handle the mis-definition inst with params
                if len(paramKeyVals) != 0:
                    raise Exception('instance:{} with model:{} misses the definition!'
                                    .format(device.name,modelName))
        
        devIdx = []
        for k, dev_in_sub in enumerate(subckt.deviceList):
            if dev_in_sub.name == device.name:
                devIdx.append(k)

        # handle the same instance. It happends in 2 situations: (a)
        # instance re-define, which is illegel, (b) instance with a
        # line wrapper '+' followering the device parameters
        assert len(devIdx)==0
        
        subckt.deviceList.append(device)
        # instance handling is finished at here
        # to do

        ## ========== now we handle the nets' info of this instance ==========
        # do statistics for the nets
        for ni in range(1, modelNameIdx):
            # find the net in the existing netList
            ntIdx = -1
            for k, net_in_sub in enumerate(subckt.netList):
                if net_in_sub.name == tokens[ni]:
                    net = net_in_sub
                    ntIdx = k
            if ntIdx == -1:
                net = Net()
                net.name = tokens[ni]

            # if multiple nets connect to the same instance, we set a flag
            # to suggest this is the first time we met the net
            firstPos = -1
            for t, token in enumerate(tokens):
                if tokens[ni] == token:
                    firstPos = t
                    break
            firstNd = (firstPos == ni)
            if 'MOS' == device.type:
                net.num_mos += device.m * firstNd
                net.num_mos_g += device.m * device.nf * (ni == 2)
                net.num_mos_sd += device.m * device.nf * (ni == 1 or ni == 3)
                net.num_mos_b += device.m * (ni == 4)
                net.tot_w += device.w * device.m * (ni == 1 or ni == 3)
                net.tot_l += device.l * device.m * device.nf * (ni == 2)
            elif 'Rupolym' == device.type:
                net.num_r += device.m * firstNd
                net.tot_r_l += device.r_l * device.m
                net.tot_r_w += device.r_w * device.m
            elif 'Cfmom' == device.type:
                net.num_c += device.m * firstNd
                net.tot_nr += device.m * device.nr
                net.tot_lr += device.m * device.lr
            # moscap is also a mos device, we do not distinguish
            # them. For moscap, the D and S are shorted together.
            elif 'Moscap' == device.type:
                net.num_mos += device.m * firstNd
                net.num_mos_g += device.m * device.nf * (ni == 1)
                net.num_mos_sd += device.m * device.nf * (ni == 2) * 2
                net.tot_w += device.m * device.wr * 2
                net.tot_l += device.m * device.m_lr
            elif 'Ndio' == device.type:
                net.num_d += device.m * firstNd
                net.tot_area += device.m * device.area
                net.tot_pj += device.m * device.pj
            elif 'Subckt' == device.type:
                # we move this part to updateSubckt()
                if subIdx != -1:
                    pass
                    # merge the info from the port of the matched subckt to
                    # the current net of the current subckt
                    # if len(subcktList[subIdx].stashNtList):
                    #     print('net before merge:', net)
                    #     net.mergeNets(subcktList[subIdx].stashNtList[ni - 1])
                    #     print('net after merge:', net)
                    # else:
                    #     net.mergeNets(subcktList[subIdx].netList[ni - 1])
            
            # if net.isempty():
            #     print(net) 
                
            # save the instance pointer (index number) into the net, the
            # new inst that we just construted is at the tail of the deviceList
            net.devPtr.append(len(subckt.deviceList)-1)
            if ntIdx == -1:
                subckt.netList.append(net)
            # save the net pointer (index number) into the inst in the deviceList
            subckt.deviceList[-1].ntPtr.append(ntIdx if ntIdx != -1 else len(subckt.netList)-1)
        # clean up the nets connecting to parameterized instances in this subckt
        # if  device.type == 'Subckt' and subIdx!=-1 and len(subcktList[subIdx].stashNtList):
        #     subcktList[subIdx].stashNtList = list()
    
    assert len(subcktList)

    # handle the missed subckt definition in the subcktList
    updateSubckt(subcktList, topSubcktIdx)
    
    ## save the subckList
    if len(subcktList):
        mName = re.split(r'[/.]', filename.strip())
        sbcktFileName = '/data1/shenshan/SPF_examples_cdlspf/Python_data/' + mName[-2] +'.sbckt.bin'
        try:
            with open(sbcktFileName, "wb") as f:
                pickle.dump(subcktList, f)
                print('write bin file {} successfully'.format(sbcktFileName))
        except:
            raise Exception('failed to save subcktList into {}'.format(sbcktFileName))
    
    return subcktList, topSubcktIdx
    
def updateSubckt(subcktList, subIdx): 
    # this subckt doesn't contain a missed definition instance
    subckt = subcktList[subIdx]
    if subckt.scan:
        assert (len(subckt.params) == 0)
        return

    print("updating subckt: %s, id: %d" % (subckt.name, subIdx))
    
    # iterativelly check its deviceList
    for i, device in enumerate(subckt.deviceList):
        if device.type != 'Subckt':
            continue
        
        # if inst's definition is missing, we seach the subcktList again
        # to update the net's infomation
        cSubIdx = -1
        for s, sub in enumerate(subcktList):
            if sub.name == device.subName:
                cSubIdx = s
                break
        
        if cSubIdx != -1: 
            # update the net info connecting to this inst
            assert(subcktList[cSubIdx].portNum == len(device.ntPtr))
            # call this function recursively till there is no inst that
            # hasn't been defined yet
            updateSubckt(subcktList, cSubIdx)
            # update the referenced pointer list of the child subckt
            subcktList[cSubIdx].refPtr.append(subIdx)
            device.subPtr = cSubIdx
            # fixed the error when we cannot find its definition 
            device.type = 'Subckt'
            if len(device.paramKeyVals):
                assert (len(subcktList[cSubIdx].params))
                assert (len(subcktList[cSubIdx].params) == len(device.paramKeyVals))
                print("with paramKeyVals:", device.paramKeyVals)
                subcktList[cSubIdx].update(device.paramKeyVals)

            # update nets in this subckt indexed by device.ntPtr
            for l, ntIdx in enumerate(device.ntPtr):
                curNet = subckt.netList[ntIdx]
                # Note: the pin order of the current inst is same as that 
                # in its definition subckt
                if len(device.paramKeyVals):
                    print('before merge:', curNet)
                    curNet.mergeNets(subcktList[cSubIdx].stashNtList[l])
                    print('after merge:', curNet)
                else:
                    curNet.mergeNets(subcktList[cSubIdx].netList[l])
        
            # clean the stashNtList for other device instance
            subcktList[cSubIdx].stashNtList = list()
        elif cSubIdx == -1:
            raise Exception('Definition not found! inst: {}, subckt: {}'
                            .format(subckt.deviceList[i].name,
                                    subckt.deviceList[i].subName))
    
    for i, net in enumerate(subckt.netList):
        if net.isempty():
           warnings.warn("Empty net: %s, id: %d, subckt: %s, id: %d" 
                            % (net.name, i, subckt.name, subIdx))
        # if re.search("^BL0", net.name):
        #     print("net:", net, "in subckt:", subckt.name)

    subckt.scan = False if len(subckt.params) else True
    
    return