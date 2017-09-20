import math
import numpy as np
from cut import CutData
''' 
To caculate the codebook matrix for the second layer:

1.caculate the codebook of 1st layer(codebook1);
2.duplicate codebook1, make a larger codebook (fuc cal_codebook);
3.prepare receptive fields for each neuron in codebook1 (with original CutData fuc and reshape);
4.multiply every neuron in codebook1 with its receptive field, get the active table;
5.train those active tables with SOM algorithm;

'''

class SecondL(object):
    def __init__(self,
                diameter,
                digi_size,
                readmn,
                codebook1,
                mapsize,
                repeat_times):
        self.readmn = readmn
        self.diameter = diameter
        self.neuron_num = mapsize[0]*repeat_times
        self.digi_size = digi_size
        self.codebook1 = codebook1
        self.mapsize = mapsize
        self.repeat_times = repeat_times

        
    def cal_codebook(self):
        sq_codebook = self.codebook1.reshape(self.mapsize[0],self.mapsize[1],self.diameter**2)
        D = np.tile(sq_codebook,[self.repeat_times,self.repeat_times,1])
        codebook2_pre = D.reshape(self.neuron_num*self.neuron_num,self.diameter**2)
        codebook2 = np.asarray(codebook2_pre)
        return codebook2
    
    def active_m(self, limit):
        self.limit = limit
        cutt2 = CutData(self.diameter,self.neuron_num,self.digi_size,self.readmn) 
        cutt2.caculate_cut_P()
        active = [[],[]]
        n = 1
        for i in self.readmn():
            H = np.asarray(cutt2.do_cut(i[1]))
            J = H.reshape(self.diameter**2,self.neuron_num**2)
            active_temp = np.dot(self.cal_codebook(),J)
            each_active = np.diag(active_temp)
            active[0].append(i[0])
            active[1].append(each_active)
            n=n+1
            if n>self.limit:
                break
        return active