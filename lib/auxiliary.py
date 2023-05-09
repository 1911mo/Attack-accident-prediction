import numpy as np
import pandas as pd
import torch
import seaborn as sb
import matplotlib.pyplot as plt
import random


# 输入标签矩阵或预测矩阵，输出loss,与标签值无关
def ten_loss(a,  k, device):  # a网络输出，
    h = np.load('weight_nyc.npy',allow_pickle=True)
    h = torch.from_numpy(h).to(device)
    y15 = top_k(k, device)
    y_35 = last_k(k, device)
    return torch.sum(a*y_35)-torch.sum(a*h*y15)

def top_k(K, device):  # a为5、10、20、30、40
    aaa = pd.read_pickle('accident_value.pkl')
    aaa = aaa.reshape(-1, 400)
    b = np.sum(aaa, axis=0)
    sort_array = np.argsort(-b)
    sort_array = sort_array[:K]
    d = np.zeros(400)
    d[sort_array] = 1
    d = d.reshape(20, 20)

    risk_file = torch.from_numpy(d)
    risk_file = risk_file.to(device=device)
    return risk_file


def last_k(K, device):  # a为5、10、20、30、40
    aaa = pd.read_pickle('accident_value.pkl')
    aaa = aaa.reshape(-1, 400)
    b = np.sum(aaa, axis=0)
    sort_array = np.argsort(-b)
    sort_array = sort_array[K:2*K]
    d = np.zeros(400)
    d[sort_array] = 1
    d = d.reshape(20, 20)

    risk_file = torch.from_numpy(d)
    risk_file = risk_file.to(device=device)
    return risk_file

def item_saliency_map_a(input_grads, k, map_243):
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20
    
    input_grads = input_grads[:,:,0,:,:].mean(dim=1)  # 先样本平均32*1*1*20*20
    #input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []
    view_node = []
    # 243个道路区域有多少被包含在里边
    d1 = input_grads.reshape(input_grads.shape[0],400)
    for i in range(input_grads.shape[0]):
        
        d2  = d1[i,:].reshape(20,20)
        dk =d2.clone().reshape(400)
        dk_index = torch.argsort(-dk)
        ds = torch.where(d2>0,1.,0.)
        dn = ds*map_243
        
        if torch.sum(torch.sum(dn))>k:
            d2 = torch.where(d2>dk[dk_index[k]],1.,0.)
            #print('走过路过')
        else:
            d2 = torch.where(d2>0,1.,0.)
        d3 = d2*map_243
        d3 = d3.cpu().numpy()
        node_map.append(d3)
    #hoo = node_saliency_map
    return node_map

def vers_attack001_k(map,feature,device,l_50):
    #输入：干净样本，32*7*48*20*20
    #输出：对抗样本：32*7*48*20*20
    #map 32*20*20
    #第一步攻击效果计算
    #第二步扰动最小化
    #7个时间片仅仅攻击最后一个
    #torch编程，输入已经在device上
    #print(zinb_sample.shape)
    a,b,c,d,e =feature.shape
    feature_use = torch.squeeze(feature[:,:,0,:,:]) #32*7*20*20
    
    for i in range(a):#32
        a_num = torch.sum(map[i,:,:]>0.5)#寻找位置
        print('a_sum攻击点数为{}'.format(a_num))
        if a_num==400:
            a_num = 50
        if a_num > 50:
            a_num = 50
        map_use = torch.unsqueeze(map,1).repeat(1,7,1,1)#32*7*20*20
        feature_attack = feature_use*map_use#只攻击这些地方
        feature_attack = feature_attack.reshape(a,7,-1)#32*7*400
        feature_new = feature_attack.to(device)#32*7*400
        for j  in range(7):
            _,zzz = torch.sort(feature_new[i,j,:],descending=True)
            lrand = random.randint(1,999)
            for k in range(a_num-1):
                #print(lrand)
                #print(feature_new.shape)
                print(zzz[a_num-k-1])
                feature_new[i,j,zzz[a_num-k-1]] = l_50[lrand,k]
            for k in range(a_num, 400):
                feature_new[i, j, zzz[k]] = 0.
    feature_new = feature_new.reshape(a,7,20,20)
    feature_return = feature.clone().detach()
    feature_return[:,:,0,:,:]= feature_new 
    #print((feature_return[1,1,0,:,:]==feature[1,1,0,:,:]).min())
    return feature_return

def vers_attack001_map(map,feature,device,l_50):
    #输入：干净样本，32*7*48*20*20
    #输出：对抗样本：32*7*48*20*20
    #map 32*20*20
    #第一步攻击效果计算
    #第二步扰动最小化
    #7个时间片仅仅攻击最后一个
    #torch编程，输入已经在device上
    #print(zinb_sample.shape)
    a =feature.shape[0]
    map_a = map.clone()
    feature_new = feature.clone()
    for i in range(a):#32
        a_sample = []
        map_a = map[i,:,:].clone()
        map_b = map[i,:,:].clone()
        #print(map_a)
        #print('a_sum攻击点数为{}'.format(a_num))
        for j  in range(7):
            lrand = random.randint(1,999)
            lo = 0
            for m in range(20):
                for n in range(20):
                    if map_b[m,n]>0 and lo<30:
                        lo = lo+1
                        map_a[m,n] = l_50[lrand,lo]
            a_sample.append(map_a)
            #print(map_a*46.0)
        #7*20*20
        feature_risk = torch.stack(a_sample)
        #print(feature_risk.shape)
        feature_new[i,:,0,:,:] = feature_risk
    return feature_new