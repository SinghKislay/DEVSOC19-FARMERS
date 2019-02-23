import numpy as np
import csv
import os
import tensorflow as tf
from tqdm import tqdm

DATA_DIR="./data_train"
LABEL_DIR="./label_train"

def exchange(path):
    with open(path,'r') as csvfile:
        exch_data=csv.reader(csvfile)
        exch_data=list(exch_data)
        exch_data.pop(0)
        buffer=exch_data[0][0].split('-')[0]
        temp=0
        list_exch=[]
        r=0
        for i in exch_data:
            if(i[0].split('-')[0]==buffer):
                if(len(i[17])>0):
                    r=r+1
                    temp=temp+float(i[17])
            if(i[0].split('-')[0]!=buffer):
                temp=temp/r
                list_exch.append([int(buffer),temp])
                buffer=i[0].split('-')[0]
                r=0
                if(len(i[17])>0):
                    r=r+1
                    temp=temp+float(i[17])
        
        list_exch.pop(-1)
        list_exch.pop(-1)
        return list_exch

def rainfall(path):
    with open(path,'r') as csvfile:
        rain_data=csv.reader(csvfile)
        rain_data=list(rain_data)
        rain_data.pop(0)
        rain_list=[]
        for date in range(1995,2016):
            for _,i in enumerate(rain_data):
                if(i[0].split(' ')[0]=='ASSAM' or i[0].split(' ')[0]=="BIHAR" or i[0].split(' ')[0]=="PUNJAB" or i[0].split(' ')[0]=="WEST" or i[0].split(' ')[0]=="EAST" or i[0].split(' ')[0]=="KERELA"):
                    if(int(i[1])==date):
                        temp=0
                        l=0
                        for x,j in enumerate(i):
                            if (x>1 and len(j)>0 and j!="NA"):
                                l=l+1
                                temp=temp+ float(j)
            rain_list.append([int(date),temp/l])           
        return rain_list
        
def gdp(path):
    with open(path,'r') as csvfile:
        gdp_data=csv.reader(csvfile)
        gdp_data=list(gdp_data)
        gdp_data.pop(0)
        gdp_list=[]
        for i in gdp_data:
            if(int(i[0])>=1995 and int(i[0])<2016):
                gdp_list.append([int(i[0]),int(i[1])])
        
        return gdp_list

def label(path):
    with open(path,'r') as csvfile:
        label=csv.reader(csvfile)
        label=list(label)
        
        label_list=[]
        for i in range(1995,2016):
            for x,j in enumerate(label[0]):
                if (x>0):
                    if int(j)==i:
                        label_list.append([i,int(label[1][x])])
        return label_list

def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#r_list=rainfall('./data_train/rainfall.csv')
#print(r_list)
#l=gdp("./data_train/gdp.csv")
#print(l)
#temp=exchange('./data_train/exchange.csv')
#print(temp)
#label("./label_train/rice_demand.csv")
with tf.python_io.TFRecordWriter("./tfRecords.train") as writer:
    lablel_list=label("./label_train/rice_demand.csv")
    exchange_list=exchange('./data_train/exchange.csv')
    rainfall_list=rainfall('./data_train/rainfall.csv')
    gdp_list=gdp("./data_train/gdp.csv")
    #print([lablel_list,exchange_list,rainfall_list,gdp_list])
    
    for i in tqdm(range(0,20,4)):
        
        data={
            'gdp1':wrap_float(gdp_list[i][1]),
            'exchange1':wrap_float(exchange_list[i][1]),
            'rainfall1':wrap_float(rainfall_list[i][1]),
            'label1':wrap_float(lablel_list[i][1]),
            'gdp2':wrap_float(gdp_list[i+1][1]),
            'exchange2':wrap_float(exchange_list[i+1][1]),
            'rainfall2':wrap_float(rainfall_list[i+1][1]),
            'label2':wrap_float(lablel_list[i+1][1]),
            'gdp3':wrap_float(gdp_list[i+2][1]),
            'exchange3':wrap_float(exchange_list[i+2][1]),
            'rainfall3':wrap_float(rainfall_list[i+2][1]),
            'label3':wrap_float(lablel_list[i+2][1]),
            'gdp4':wrap_float(gdp_list[i+3][1]),
            'exchange4':wrap_float(exchange_list[i+3][1]),
            'rainfall4':wrap_float(rainfall_list[i+3][1]),
            'label4':wrap_float(lablel_list[i+3][1]),

        }

        feature=tf.train.Features(feature=data)
        example=tf.train.Example(features=feature)
        serialized = example.SerializeToString()
        writer.write(serialized)

