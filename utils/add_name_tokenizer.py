from read_cdl import read_cdl
from build_graph import build_graph
# from cdl2graph_main import findInterNetName
import os
import warnings
from transformers import AutoTokenizer, AutoModel
from dgl.data.utils import load_graphs, save_graphs
import subckt
import torch
import pickle
import re
import torch.nn.functional as F

def circuitNamePreprocess(name:str, type:str):
    name = name.lower()
    if type == 'inst' or type == 'device':
        name = re.sub('/', '_', name)
        name = re.sub('_zha', '', name)
        name = re.sub('_shen', '', name)
        stdcell = False
        if re.search('bwp7t[34]0p140',  name):
            stdcell = True
            name = re.sub('bwp7t[34]0p140', '',  name) 
        # name = re.sub('bl', '_bitline_', name)
        # if re.search('clkb', name):
        ## gate name
        name = re.sub('drv', '_driver_', name)
        name = re.sub('buf+[er]*', '_buffer_', name)
        name = re.sub('b[68]t','_bitcell_', name)
        name = re.sub('inv','_inverter_gate_', name)
        if re.search('nand', name):
            name = re.sub('nand','_not_and_gate_', name)
        elif re.search('trans', name):
            name = re.sub('trans', '_transmission_gate_', name)
        elif re.search('^and', name):
            name = re.sub('^and','_and_gate_', name)
            
        if stdcell:
            name = re.sub('^nd','_not_and_gate_', name)
            name = re.sub('^an','_and_gate_', name)
            name = re.sub('ao','_and_or_gate_', name)
            name = re.sub('del','_delay_cell_', name)
            name = re.sub('fa','_1_bit_full_adder_', name)
            name = re.sub('nr','_not_or_gate_', name)
            name = re.sub('xnr','_exclusive_not_or_gate_', name)
            name = re.sub('hdr','_header_', name)
            name = re.sub('lh','_high_enable_latch_', name)
            name = re.sub('ln','_low_enable_latch_', name)
        name = re.sub('lvl', '_level_shifter_', name)
        name = re.sub('lch','_latch_', name)
        name = re.sub('dec','_decoder_', name)
        name = re.sub('df+','_d_flip_flop_', name)
        name = re.sub('xor','_exclusive_or_gate_', name)
        name = re.sub('[clo]*ckb','_clock_buffer_', name)
        name = re.sub('[clo]*ck','_clock_', name)
        name = re.sub('tri','_triple_', name)
        name = re.sub('mux','_multiplexer_', name)
        name = re.sub('ctrl','_controller_', name)
        name = re.sub('reg','_register_', name)
        name = re.sub('amp','_amplifier_', name)
        name = re.sub('sa', '_sense_amplifier_', name)
        name = re.sub('bot', '_bottom_', name)
        name = re.sub('chn', '_channel_', name)
        name = re.sub('mem', '_memory_', name)
        name = re.sub('col', '_column_', name)
        name = re.sub('pre', '_precharge_', name)

        if re.search('[lhs]vt', name):
            name = re.sub('ulvt', '_with_ultra_low_threshold_voltage_', name)
            name = re.sub('lvt', '_with_low_threshold_voltage_', name)
            name = re.sub('svt', '_with_standard_threshold_voltage_', name)
            name = re.sub('uhvt', '_with_ultra_high_threshold_voltage_', name)
            name = re.sub('hvt', '_with_high_threshold_voltage_', name)
        name = re.sub('inst', '_instance_', name)
        # if type== "device":
        #     name = re.sub('^[x]*m+', '_transistor_instance_', name)
        if re.search('[np]ch', name):
            name = re.sub('_mac', '', name)
            name = re.sub('nch', '_nmos_', name)
            name = re.sub('pch', '_pmos_', name)
            name = re.sub('pu', '_pull_up_', name)
            name = re.sub('pd', '_pull_down_', name)
            name = re.sub('rp', '_replica_port_', name)
            name = re.sub('pg', '_pass_gate_', name)
            name = re.sub('sr', '_sram_transistor_', name)
    ## signals
    elif type == 'net':
        ## in sram cell
        name = re.sub('rd', '_read_', name)
        name = re.sub('wr', '_write_', name)
        if re.search('rbl', name):
            name = re.sub('rbl[bn]', '_replica_bitline_bar_', name)
            name = re.sub('rbl', '_replica_bitline_', name)
        elif re.search('nbl[bn]*', name):
            name = re.sub('nbl[bn]', '_negtive_bitline_bar_', name)
            name = re.sub('nbl', '_negtive_bitline_', name)
        elif re.search('bl', name):
            name = re.sub('bl[bn]', '_bitline_bar_', name)
            name = re.sub('bl', '_bitline_', name)
        if re.search('[wr]bl', name):
            name = re.sub('wwl[bn]', '_write_wordline_bar_', name)
            name = re.sub('wwl', '_write_wordline_', name)
            name = re.sub('rwl[bn]', '_read_wordline_bar_', name)
            name = re.sub('rwl', '_read_wordline_', name)
        elif re.search('wl', name):
            name = re.sub('wl[bn]', '_wordline_bar_', name)
            name = re.sub('wl', '_wordline_', name)
        name = re.sub('ctrl[nb]','_control_bar_', name)
        name = re.sub('ctrl','_control_', name)
        name = re.sub('amp[bn]','_amplify_bar_', name)
        name = re.sub('amp','_amplify_', name)
        name = re.sub('sel','_select_', name)
        name = re.sub('^w','_write_', name)
        # name = re.sub('^r','_read_', name)
        name = re.sub('lc','_latch_', name)
        name = re.sub('lc','_latch_', name)
        if re.match('cen', name):
            name = re.sub('cen', '_chip_enble_bar_', name)
        elif re.match('ce', name):
            name = re.sub('ce', '_chip_enble_', name)
        if re.match('wen', name):
            name = re.sub('wen', '_write_enble_bar_', name)
        elif re.match('en$', name):
            name = re.sub('en$', '_enble_bar_', name)
        name = re.sub('we', '_write_enble_', name)
        if re.search('pr[e]*ch', name):
            name = re.sub('pr[e]*ch', '_precharge_', name)
        elif re.search('pre', name):
            name = re.sub('pre', '_precharge_', name)
        # elif re.search('e$', name) and re.search('pulse'):
        #     name = re.sub('e$', '_enable_', name)
        # elif re.search('en$', name):
        #     name = re.sub('en$', '_enable_bar_', name)
        # name = re.sub('^sa', '_sense_amplifier_', name)
        name = re.sub('sae[bn]', '_sense_amplifier_enable_bar_', name)
        name = re.sub('sae$', '_sense_amplifier_enable_', name)
        name = re.sub('sa$', '_sense_amplifier_', name)
        name = re.sub('rst[bn]', '_reset_bar_', name)
        name = re.sub('rst', '_reset_', name)
        name = re.sub('dr[iv]', '_drive_', name)
        name = re.sub('clk[bn]', '_clock_bar_', name)
        name = re.sub('clk', '_clock_', name)
        if re.search('[clo]*ck[bn]', name):
            name = re.sub('[clo]*ck[bn]','_clock_bar_', name)
        elif re.search('[clo]*ck', name):
            name = re.sub('[clo]*ck','_clock_', name)
        # if re.match('[abc]n[0-9]*', name):
        #     name = re.sub('in', '_input1_bar_', name)
        #     name = re.sub('an', '_input1_bar_', name)
        #     name = re.sub('bn', '_input2_bar_', name)
        #     name = re.sub('cn', '_input3_bar_', name)
        # elif re.match('[iabc][0-9]*', name):
        #     name = re.sub('i', '_input1_', name)
        #     name = re.sub('a', '_input1_', name)
        #     name = re.sub('b', '_input2_', name)
        #     name = re.sub('c', '_input3_', name)
        if re.match('d[bn][0-9]+', name):
            name = re.sub('d[bn]', '_data_input_bar_', name)
        elif re.match('d[0-9]+', name):
            name = re.sub('d', '_data_input_', name)
        if re.match('q[bn][0-9]+', name):
            name = re.sub('q[bn]', '_data_output_bar_', name)
        elif re.match('q[0-9]+', name):
            name = re.sub('q', '_data_output_', name)
        # if re.match('[yz][bn][0-9]*', name):
        #     name = re.sub('z[bn]', '_output_bar_', name)
        #     name = re.sub('y[bn]', '_output_bar_', name)
        # elif re.match('[yz][0-9]*', name):
        #     name = re.sub('[yz]', '_output_', name)
        name = re.sub('val$', '_value_', name)
        name = re.sub('vdd', '_power_supply_', name)
        name = re.sub('vcc', '_power_supply_', name)
        name = re.sub('vss', '_ground_node_', name)
        name = re.sub('gnd', '_ground_node_', name)

    if re.search('[vi]ref', name):
            name = re.sub('vref', '_voltage_reference_', name)
            name = re.sub('iref', '_current_reference_', name)
    elif re.search('ref', name):
        name = re.sub('ref', '_reference_', name)
    name = re.sub('neg', '_negative_', name)
    name = re.sub('pos', '_positive_', name)
    name =  "_" + name + "_"
    return re.sub('_+', '_', name)

# def mean_pooling(model_output, attention_mask):
        

## todo
def get_st_embedings(input, modelst):
#Compute token embeddings
    with torch.no_grad():
        input.to(modelst.device)
        model_output = modelst(**input)
        # print("model_output[last_hidden_state] shape:", model_output["last_hidden_state"].shape)
        # print("model_output[pooler_output] shape:", model_output["pooler_output"].shape)
        # print("model_output[0] shape:", model_output[0].shape)
        # st_embeddings = mean_pooling(model_output, )
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        return F.normalize(sentence_embeddings, p=2, dim=1)
        # print("st_embeddings shape:", st_embeddings.shape)
        # return st_embeddings

def findInterNetNameAndTokenize(subList: list(), 
                                subIdx: int, prefix: str,
                                tokens_emb: dict(), 
                                tokenizer, stmodel): 
    # This function search each node and sub-node in this subckt
    # netCapDict - dict contains net name: net cap
    # subList - the list of subckt in the cdl file
    # subIdx - the subckt that needs to be extracted node info
    # prefix -  the instance name and the parent name
    
    subckt = subList[subIdx]

    if len(prefix) == 0:
        ntStartIdx = 0
        instname = "The subcircuit is "+circuitNamePreprocess(subckt.name, type='inst')+'.'
    else:
        ntStartIdx = subckt.portNum 
        # instname += " and its instance name is " + prefix + "."
        instname = "The subcircuit is "+circuitNamePreprocess(subckt.name, type='inst')+\
        " and instantiated as " + prefix
        prefix = prefix+'/'

    print("instname:", instname, "original:", subckt.name)
    encoded_input = tokenizer(instname, padding=True, truncation=True, 
                              max_length=64, return_tensors='pt')
    tokens_emb["inst"].append(get_st_embedings(encoded_input, stmodel))
    
    # record the nets' info in this subckt
    for net in subckt.netList[ntStartIdx:]:
        # record the name and cap for different insts of this subckt
        netname = "The node name is "+circuitNamePreprocess(net.name, type='net') + "."
        # print("netname:", netname)
        encoded_input = tokenizer(netname, padding=True, truncation=True, 
                                  max_length=64, return_tensors='pt')
        tokens_emb["net"].append(get_st_embedings(encoded_input, stmodel))
    
    # record the internal nodes' info of each instance in this subckt
    for device in subckt.deviceList:
        assert(str(device.type) == str('Cfmom') or str(device.type) == str('Rupolym') 
               or str(device.type) == str('MOS') or str(device.type) == str('Subckt') 
               or str(device.type) == str('Ndio') or str(device.type) == str('Moscap'))
        if device.type != 'Subckt':
            # if device.type == "MOS":
            #     dvctype = "MOS transistor"
            # elif device.type == "Cfmom":
            #     dvctype = "MOM capacitor"
            # elif device.type == 'Rupolym':
            #     dvctype = "poly resistor"
            # elif device.type == 'Moscap':
            #     dvctype = "MOS capacitor"
            # elif device.type == "Ndio":
            #     dvctype = "diode"

            devicename = "This device is an instance of "+ \
                         circuitNamePreprocess(device.subName, 'device')+"."
            # print("devicename:", devicename)
            encoded_input = tokenizer(devicename, padding=True, truncation=True, 
                                      max_length=64, return_tensors='pt')
            tokens_emb["device"].append(get_st_embedings(encoded_input, stmodel))
        else:
            # find the definition of this instance
            cSubIdx = device.subPtr
            assert cSubIdx >= 0
            findInterNetNameAndTokenize(subList, cSubIdx, prefix+device.name, 
                                        tokens_emb, tokenizer, stmodel)

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.bi_graph.bin'):
                fullname = os.path.join(root, f)
                yield fullname

    

if __name__ == '__main__':   
    raw_dir = '/data1/shenshan/SPF_examples_cdlspf/Python_data/'
    # for i in findAllFile(raw_dir):
    #     print(i)
    file_name = "sram_sp_8192w"#"sandwich"#'ssram'#'ultra_8T'#'array_128_32_8t' #
    py_file_name = raw_dir+file_name+".sbckt.bin"
    moduleName = 'sram_sp_8192w'#'pe_macro2'#'ssram'#'ultra_8T_macro'#'array_128_32_8t' #'8T_digitized_timing_top_fast' #
    topSubIdx = -1
    subcktList = []
    shg = None

    if not os.path.isfile(py_file_name):
        pass
        # subcktList, topSubIdx = read_cdl('/data1/shenshan/SPF_examples_cdlspf/CDL_files/SSRAM/'+moduleName+'.cdl', moduleName)
        # assert topSubIdx != -1
    else:
        try:
            with open(py_file_name, "rb") as f:
                subcktList = pickle.load(f)
                print('read sbckt.bin file {} successfully'.format(py_file_name))
                subIdx = -1
                # moduleName = 'front_8T_digitized_timing_top_256_fast'
                for subIdx, subckt in enumerate(subcktList):
                    if subckt.name == moduleName:
                        topSubIdx = subIdx
                        # print('find subckt name: %s %d' % ('vref', ))
                        break
        except:
            raise Exception('failed to load subcktList into {}'.format(py_file_name))
    
    assert topSubIdx != -1
    assert len(subcktList)

    tokens_dict = {"net": [], "device": [], "inst": []}
    print('Calling findInterNetNameAndTokenize...')
    tokenizer = AutoTokenizer.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2")
    stmodel= AutoModel.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2").to(torch.device('cuda:0') )
    findInterNetNameAndTokenize(subcktList, topSubIdx, '', tokens_dict, tokenizer, stmodel)

    # print("net encode len", len(tokens_dict["net"]))
    # net_tid_lens = []
    # net_tid_data = []
    # max_length = 32
    print("tokens_dict[net] shape:", tokens_dict["net"][0].shape)
    # for ids in tokens_dict["net"]:
    #     # print("ids:", ids)
    #     net_tid_lens.append(ids.shape[1])
    #     net_tid_data.append(torch.cat((ids, torch.zeros(ids.shape[0], max_length - ids.shape[1])), dim=1))

    # print("device encode len", len(tokens_dict["device"]))
    # device_tid_lens = []
    # device_tid_data = []
    # for ids in tokens_dict["device"]:
    #     device_tid_lens.append(ids.shape[1])
    #     device_tid_data.append(torch.cat((ids, torch.zeros(1, max_length - ids.shape[1])), dim=1))

    # print("subckt inst encode len", len(tokens_dict["inst"]))
    # inst_tid_lens = []
    # inst_tid_data = []
    # for ids in tokens_dict["inst"]:
    #     inst_tid_lens.append(ids.shape[1])
    #     inst_tid_data.append(torch.cat((ids, torch.zeros(1, max_length - ids.shape[1])), dim=1))

    try:
        het_g, _ = load_graphs(raw_dir + file_name+ '.bi_graph.bin')
        shg = het_g[0].long()
    except:
        raise Exception('failed to load bi_graph {}'.format(py_file_name))
    
    assert shg is not None
    assert shg.nodes['net'].data['x'].shape[0] == len(tokens_dict['net'])
    assert shg.nodes['device'].data['x'].shape[0] == len(tokens_dict['device'])
    assert shg.nodes['inst'].data['x'].shape[0] == len(tokens_dict['inst'])

    shg.nodes['net'].data['emb'] = torch.cat(tokens_dict['net'], dim=0).to(torch.float16).to(torch.device('cpu'))
    # shg.nodes['net'].data['tid_len'] = torch.tensor(net_tid_lens).view(-1, 1)
    shg.nodes['device'].data['emb'] = torch.cat(tokens_dict['device'], dim=0).to(torch.float16).to(torch.device('cpu'))
    # shg.nodes['device'].data['tid_len'] = torch.tensor(device_tid_lens).view(-1, 1)
    shg.nodes['inst'].data['emb'] = torch.cat(tokens_dict['inst'], dim=0).to(torch.float16).to(torch.device('cpu'))
    # shg.nodes['inst'].data['tid_len'] = torch.tensor(inst_tid_lens).view(-1, 1)
    # assert 0
    gFilePath = "/data1/shenshan/SPF_examples_cdlspf/Python_data/" + \
                 file_name + ".bi_graph_nametokens.bin"
    save_graphs(gFilePath, [shg])
    print('saved graph hg to', gFilePath)