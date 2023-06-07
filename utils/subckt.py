import numpy as np
import copy

class Subckt():
    def __init__(self = None): 
        self.netList = list()
        self.stashNtList = list()
        self.deviceList = list()
        self.name = ''
        self.portNum = 0
        self.misInstDf = False
        self.refPtr = []
        self.params = dict()
        return

    def copySub(self): 
        nSub = Subckt()
        nSub.netList = copy.deepcopy(self.netList)
        nSub.stashNtList = copy.deepcopy(self.stashNtList)
        nSub.deviceList = copy.deepcopy(self.deviceList)
        nSub.name = self.name
        nSub.portNum = self.portNum
        nSub.misInstDf = self.misInstDf
        nSub.refPtr = self.refPtr
        nSub.params = copy.deepcopy(self.params)
        return nSub
        
        
    def update(self, paramKeyVals : dict = None): 
        #nSub = copySub(obj);
        # init a new nodeList for the instance of this subckt
        self.stashNtList = copy.deepcopy(self.netList)
        # update instances in the subckt using the given parameters
        for device in self.deviceList:
            for paramKey in paramKeyVals.keys():
                if paramKey not in device.params.keys():
                    continue
                print('inst name:{} type:{} param:{} value:{:.2f} instparam:{}'
                      .format(device.name, device.type, paramKey, 
                              paramKeyVals[paramKey], device.params[paramKey]))
                if device.params[paramKey] == 'm':
                    device.m = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'w':
                    device.w = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'l':
                    device.l = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'nf':
                    device.nf = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'r_w':
                    device.r_w = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'r_l':
                    device.r_l = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'nr':
                    device.nr = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'lr':
                    device.nf = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'wr':
                    device.wr = paramKeyVals[paramKey]
                elif device.params[paramKey] == 'm_lr':
                    device.m_lr = paramKeyVals[paramKey]

            lastPtr = - 1
            # update the node's info with the parameters in the given inst
            for j in range(len(device.ntPtr)):
                net = self.stashNtList[device.ntPtr[j]]
                firstNd = (lastPtr != device.ntPtr[j])
                if 'MOS' == device.type:
                    net.num_mos += device.m * firstNd
                    # print('node.num_mos:', node.num_mos)
                    net.num_mos_g += device.m * device.nf * (j == 1)
                    net.num_mos_sd += device.m * device.nf * (j == 0 or j == 2)
                    net.num_mos_b += device.m * (j == 3)
                    net.tot_w += device.w * device.m * (j == 0 or j == 2)
                    net.tot_l += device.l * device.m * device.nf * (j == 1)
                elif 'Rupolym' == device.type:
                    net.num_r += device.m
                    net.tot_r_l += device.r_l * device.m
                    net.tot_r_w += device.r_w * device.m
                elif 'Cfmom' == device.type:
                    net.num_c += device.m * firstNd
                    net.tot_nr += device.m * device.nr
                    net.tot_lr += device.m * device.lr
                elif 'Moscap' == device.type:
                    net.num_mos += device.m * firstNd
                    net.num_mos_g += device.m * device.nf * (j == 0)
                    net.num_mos_sd += device.m * device.nf * (j == 1) * 2
                    net.tot_w += device.m * device.wr
                    net.tot_l += device.m * device.m_lr
                elif 'Ndio' == device.type:
                    net.num_d += device.m * firstNd
                    net.tot_area += device.m * device.area
                    net.tot_pj += device.m * device.pj
                elif 'Subckt' == device.type:
                    raise Exception('self inst:%s is a subckt:%s !!',device.name,device.subName)
                lastPtr = device.ntPtr[j]
        # for i, node in enumerate(self.nodeList):
        #     print('node in nodeList', node)
        #     print('node in stashNodeList',self.stashNdList[i])
        
    # def clean(self): 
    #     if len(self.params)==0:
    #         return self
        
    #     # to do
    #     for i in np.arange(1,len(self.instList)+1).reshape(-1):
    #         self.instList[i].clean()
        
    #     for i in np.arange(1,len(self.nodeList)+1).reshape(-1):
    #         self.nodeList[i].clean()
        
    #     return self

    
class Device():
    def __init__(self): 
        self.w = 0.0
        self.l = 0.0
        self.m = 1
        self.nf = 1
        self.r_w = 0.0
        self.r_l = 0.0
        self.nr = 0.0
        self.lr = 0.0
        self.stm = 0
        self.spm = 0
        self.wr = 0.0
        self.m_lr = 0.0
        self.area = 0.0
        self.pj = 0.0
        self.name = ''
        self.subPtr = -1
        # the indeces pointing to netList in subckt
        self.ntPtr = list()
        self.type = ''
        self.subName = ''
        self.params = dict()
        return
        
    def hasParams(self): 
        return len(self.params)
        
    def clean(self): 
        self.w = 0.0
        self.l = 0.0
        self.m = 1
        self.nf = 1
        self.r_w = 0.0
        self.r_l = 0.0
        self.nr = 0.0
        self.lr = 0.0
        self.stm = 0
        self.spm = 0
        self.wr = 0.0
        self.m_lr = 0.0
        self.area = 0.0
        self.pj = 0.0
        return self


class Net():
    def __init__(self): 
        self.num_mos = 0
        self.num_mos_g = 0
        self.num_mos_sd = 0
        self.num_mos_b = 0
        self.tot_w = 0.0
        self.tot_l = 0.0
        self.num_r = 0
        self.tot_r_w = 0.0
        self.tot_r_l = 0.0
        self.num_c = 0
        self.tot_nr = 0.0
        self.tot_lr = 0.0
        self.num_d = 0
        self.tot_area = 0.0
        self.tot_pj = 0.0
        self.name = ''
        self.postNameList = list()
        self.port = 0
        
        self.devPtr = list()
        self.capList = list()
        return
        
        
    def mergeNets(self, otherNet): 
        self.num_mos += otherNet.num_mos
        self.num_mos_g += otherNet.num_mos_g
        self.num_mos_sd += otherNet.num_mos_sd
        self.num_mos_b += otherNet.num_mos_b
        self.tot_w += otherNet.tot_w
        self.tot_l += otherNet.tot_l
        self.num_r += otherNet.num_r
        self.tot_r_w += otherNet.tot_r_w
        self.tot_r_l += otherNet.tot_r_l
        self.num_c += otherNet.num_c
        self.tot_nr += otherNet.tot_nr
        self.tot_lr += otherNet.tot_lr
        self.num_d += otherNet.num_d
        self.tot_area += otherNet.tot_area
        self.tot_pj += otherNet.tot_pj
        self.port += otherNet.port
        
        # self.cap += otherNode.cap
        return
        
    def copyNet(self): 
        nNet = Net()
        nNet.num_mos = self.num_mos
        nNet.num_mos_g = self.num_mos_g
        nNet.num_mos_sd = self.num_mos_sd
        nNet.num_mos_b = self.num_mos_b
        nNet.tot_w = self.tot_w
        nNet.tot_l = self.tot_l
        nNet.num_r = self.num_r
        nNet.tot_r_w = self.tot_r_w
        nNet.tot_r_l = self.tot_r_l
        nNet.num_c = self.num_c
        nNet.tot_nr = self.tot_nr
        nNet.tot_lr = self.tot_lr
        nNet.num_d = self.num_d
        nNet.tot_area = self.tot_area
        nNet.tot_pj = self.tot_pj
        nNet.name = self.name
        nNet.port = self.port
        nNet.devPtr = self.devPtr
        nNet.postNameList = copy.deepcopy(self.postNameList)
        nNet.capList = copy.deepcopy(self.capList)
        return nNet
        
    def clean(self): 
        self.num_mos = 0
        self.num_mos_g = 0
        self.num_mos_sd = 0
        self.num_mos_b = 0
        self.tot_w = 0.0
        self.tot_l = 0.0
        self.num_r = 0
        self.tot_r_w = 0.0
        self.tot_r_l = 0.0
        self.num_c = 0
        self.tot_nr = 0.0
        self.tot_lr = 0.0
        self.num_d = 0
        self.tot_area = 0.0
        self.tot_pj = 0.0
        self.capList = []
        self.postNameList = []
        return self
    
    def nonempty(self):
        return bool(self.num_mos + self.num_r + self.num_c + self.num_d)
    
    def __repr__(self):
        pass
        # return "Test()"
    
    def __str__(self):
        info = self.name+" ["
        if self.num_mos:
            info += " ".join(["#mos=%d"%self.num_mos, "#g=%d"%self.num_mos_g, 
                              "#sd=%d"%self.num_mos_sd, "#b=%d"%self.num_mos_b,
                              "totw=%.2E"%self.tot_w, "totl=%.2E"%self.tot_l])
        if self.num_r:
            info += " " + " ".join(["#res=%d"%self.num_r, "totw=%.2E"%self.tot_r_w,
                             "totl=%.2E"%self.tot_r_l])
        if self.num_c:
            info += " " + " ".join(["#c=%d"%self.num_c, "totnr=%d"%self.tot_nr,
                             "totlr=%.2E"%self.tot_lr])
        if self.num_d:
            info += " " + " ".join(["#d=%d"%self.num_d, "area=%.2E"%self.tot_area,
                             "totpj=%.2E"%self.tot_pj])
        if len(self.capList) != 0:
            assert len(self.postNameList) == len(self.capList)
            for (i, cap) in enumerate(self.capList):
                info += " %s=%f"%(self.postNameList[i], cap)
        return info+"]"