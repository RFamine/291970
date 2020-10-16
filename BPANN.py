# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:46:55 2020

@author: XRchen
"""
import math
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

T=0.001

def fun_1(x0): 
    y=0.5*math.sin(6*math.pi*x0*T)
    return y


def fun_2(x1):
    if x1>0:
        a=fun_2(x1-1)
        y=(fun_1(x1-1)-0.9*a)/(1+math.pow(a,2))
    else:
        y=0        
    return y

class BPnetwork():
    

    def __init__(self,hiddenlayer):
        self.hiddenLayer=hiddenlayer#number of hiddenlayer nodes
        self.Wi=np.random.rand(2,hiddenlayer)#weight of hiddenlayer input
        self.Bi=np.random.rand(1,hiddenlayer)
        self.Wo=np.random.rand(hiddenlayer,1)#weight of hiddenlayer output
        self.oWo=self.Wo
        self.oWi=self.Wi
        self.Bo=np.random.rand(1)
        self.lr=0.5#learning rate
        self.a=0.05#factor momentum
   
    def sigmoid(self,x):
        y=1/(1+np.exp(-x))
        return y

    def dsigmoid(self,x):
        y=math.exp(-x)/np.pow(1+math.exp(-x))
        return y
    
    def linear(self,x):
        return x
    
    def train(self,ofun,ifun,x):
        Xi=np.hstack((ifun(x),ofun(x)))
        
        hin=np.dot(Xi,self.Wi)-self.Bi
        hout=self.sigmoid(hin)#output of hiddenlayer
        
        yin=np.dot(hout,self.Wo)-self.Bo
        yout=yin
        
        e=ofun(x+1)-yout
        print("e")
        print(e)#show the residual
        
        self.Wo=self.Wo.reshape(1,self.hiddenLayer)#change the shape of wo for matrix multiply
        Xi=Xi.reshape(2,1)
        
        b=self.lr*e*self.Wo*hout*(1-hout)
        dWi=np.dot(Xi,b)
        
        self.Wo=self.Wo.reshape(self.hiddenLayer,1)
        Xi=Xi.reshape(1,2)
        
        dWo=self.lr*e*hout
        dWo=dWo.reshape(self.hiddenLayer,1)
        
        two=self.Wo
        twi=self.Wi
        
        self.Wi=self.Wi+dWi+self.a*(self.Wi-self.oWi)
        self.Wo=self.Wo+dWo+self.a*(self.Wo-self.oWo)
        
        self.oWo=two
        self.oWi=twi

    
    def approach(self,ifun,ofun,x):
        Xi=np.hstack((ifun(x),ofun(x)))
        hin=np.dot(Xi,self.Wi)-self.Bi
        hout=self.sigmoid(hin)
        
        yin=np.dot(hout,self.Wo)-self.Bo
        yout=yin
        return yout
        
x=np.linspace(0,999,1000)
y=np.zeros(1000)
y1=np.zeros(1000)
j=0;

a=BPnetwork(13)
j=0
for i in x:
    y[j]=fun_2(i)
    j=j+1

for i in x:
    a.train(fun_2,fun_1,i)

j=0   
for i in x:
    y1[j]=a.approach(fun_1,fun_2,i)
    j=j+1
    

plt.xlim(0,999)
plt.plot(x,y)
plt.plot(x,y1)

     



        
        
        
    


        
        



    

    






    