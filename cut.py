import math
import numpy as np
"To get receptive fields of neurons in a matrix. Cut data matrix to smaller pieces"

class CutData(object):
    def __init__(self,
                diameter,
                neuron_num,
                digi_size,
                readmn):
        self.readmn = readmn
        self.diameter = diameter
        self.neuron_num = neuron_num
        self.digi_size = digi_size
        self.fields = []
                 
    
    def caculate_cut_P(self):
        #diameter: size of cut tiles
        #neuron_num: number of neurons in one column and row
        #digi_size: size of orignal digit
        if self.neuron_num <=1:
            self.fields = [[0,self.digi_size,0,self.digi_size]]
            
        else:
            neuron_dis = (self.digi_size-self.diameter)/(self.neuron_num-1)
            row_end = False
            xleft,xright,ytop,ybottom = 0,self.diameter,0,self.diameter
            for i in range(self.neuron_num**2):
                self.fields.append([math.floor(xleft),math.floor(xright),math.floor(ytop),math.floor(ybottom)])
                if row_end:
                    xleft = 0
                    xright = self.diameter
                    ytop = self.digi_size-self.diameter if ybottom+neuron_dis > self.digi_size else ytop+neuron_dis
                    ybottom = self.digi_size if ybottom+neuron_dis > self.digi_size else ybottom+neuron_dis
                    row_end = False
                else:
                    ytop = ytop
                    ybottom = ybottom
                    if xright+neuron_dis > self.digi_size:
                        xleft = self.digi_size-self.diameter
                        xright = self.digi_size
                        row_end = True
                        #if ybottom == self.digi_size:
                        #   self.fields.append([xleft,xright,ytop,ybottom])
                        #   break
                    else:
                        xright = xright+neuron_dis
                        xleft = xleft+neuron_dis
        return self.fields
    
    def do_cut(self,data):
        tiles = []
        for i in range(len(self.fields)):
            tiles.append(data[self.fields[i][0]:self.fields[i][1], self.fields[i][2]:self.fields[i][3]])
        return tiles

    def all_cut(self,nvlimit):
        self.nvlimit = nvlimit
        all_cut = []
        for i in self.readmn():
            C = self.do_cut(i[1])
            for j in C:
                j = np.reshape(j,self.diameter**2)
                if sum(j) > self.nvlimit:
                    all_cut.append(j)
                    #nvlimit, to avoid empty cuts
        return all_cut
    