from lib.metrics import mask_evaluation_np, risk_change_rate_torch,atc_round
import numpy as np
import pandas as pd
import torch
import sys
import os
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from myfunction import kl_div, js_div, log_test_results, logger_info, view_tensor
import time
import random
from torch.utils.data import DataLoader


def none_admm_attack(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = []
    batch_idx = 0
    batch_num_x = 0
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)
    #print(risk_mask_use.shape)
    #载入
    zinb_sample = np.load('D:\\paper\\code\\GSNet-master\\GSNet-master\\sanmpling_nor.npy')#1000*400
    #需要进行修改，1000可能太大，采样后没有进行归一化,不要放到函数里多次载入。
    zinb_sample = zinb_sample/46.0#归一化
    zinb_sample= np.expand_dims(zinb_sample, 0).repeat(32, axis=0)##32*1000*400
    zinb_sample= np.expand_dims(zinb_sample, 1).repeat(7, axis=1)##32**7*1000*400
    zinb_sample = torch.from_numpy(zinb_sample)
    zinb_sample = zinb_sample.to(device)

    for map, test_loader_ues in ziped:#迭代取值
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        #X_pgd001 = feature#32*7*48*20*20
        # 生成对抗样本
        #把数据送人GPU
        feature = feature.to(device) 
        label=label.to(device)
        net=net.to(device)
        target_time= target_time.to(device)
        graph_feature= graph_feature.to(device)
        X_pgd001 = admm_attack(zinb_sample,feature,label,net,target_time, graph_feature,#这一行都是tensor
                      road_adj, risk_adj, poi_adj, grid_node_map,device)

        # 每个batch的三个量
        batch_adv = net(feature, target_time, graph_feature,
                        road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_x = net(X_pgd001, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        batch_label = label.detach().cpu().numpy()
        # 保存入数组和打印
        clean_prediction_list.append(batch_x)
        acc_prediction_list.append(batch_adv)
        label_list.append(batch_label)
        inves_batch_adv = scaler.inverse_transform(batch_adv)
        inves_batch_x = scaler.inverse_transform(batch_x)
        inves_batch_label = scaler.inverse_transform(batch_label)
        clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_x, risk_mask, 0)
        adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
            inves_batch_label, inves_batch_adv, risk_mask, 0)
        # 打印日志输出
        batch_num_x += len(feature)
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    clean_prediction = np.concatenate(clean_prediction_list, 0)
    prediction = np.concatenate(acc_prediction_list, 0)
    label = np.concatenate(label_list, 0)
    # 将标准化后的数据转为原始数据
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    inverse_trans_clean_pre = scaler.inverse_transform(clean_prediction)
    # 打印一个epoch的信息
    clean_RMSE, clean_Recall, clean_MAP, clean_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_clean_pre, risk_mask, 0)
    adv_RMSE, adv_Recall, adv_MAP, adv_RCR = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    logger_info(logger, False, 'Info:  [{}/{} ({:.0f}%)]\t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f} time:{:.3f}'.format(
                batch_num_x, 1080,
                100. * batch_idx / len(dataloader),
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,
                time.time() - start))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre



def Sparse_y(origin_matrix,a):
    #a，卷积核心大小
    #输入大小：20*20
    measurement,_ = origin_matrix.shape
    k = int(measurement/a)#卷积矩阵的大小
    print(k)
    c_di_norm2 = np.ones((k,k))
    for p in range(k):
        for q in range(k):
            c_di_norm2[p][q] = np.linalg.norm(origin_matrix[p*a:p*a+a, q*a:q*a+a])
    print(c_di_norm2)
    core = np.ones((a,a)).astype(float)

    #矩阵扩充
    sss = np.ones((measurement,measurement))
    for p in range(k):
        for q in range(k):
            sss[p*a:p*a+a, q*a:q*a+a] = core*c_di_norm2[p,q]
    print(sss)
    return sss

def Sparse_y_pile(origin_matrix,a):
    #a，卷积核心大小
    #输入大小：7*20*20
    #pile_num输入的估计数，如输入7个时间片的事故记录去估计下一个时间片
    #c_di_norm2卷积后的矩阵
    pile_num,measurement,_ = origin_matrix.shape
    k = int(measurement/a)#卷积矩阵的大小
    print(k)
    c_di_norm2 = np.ones((pile_num,k,k))
    for pil in range(pile_num):
        for p in range(k):
            for q in range(k):
                c_di_norm2[pil][p][q] = np.linalg.norm(origin_matrix[pil,p*a:p*a+a, q*a:q*a+a])
    print(c_di_norm2)
    core = np.ones((a,a)).astype(float)
     #矩阵扩充
    sss = np.ones((pile_num,measurement,measurement))
    for pil in range(pile_num):
        for p in range(k):
            for q in range(k):
                sss[pil,p*a:p*a+a, q*a:q*a+a] = core*c_di_norm2[pil,p,q]
    print(sss)
    return sss

def nontorch_Sparse_y_pile_batch(origin_matrix,a):
    #a，卷积核心大小
    #输入大小：32*7*20*20
    #pile_num输入的估计数，如输入7个时间片的事故记录去估计下一个时间片
    #32是batch数量
    #c_di_norm2卷积后的矩阵
    #输出大小：32*7*20*20
    batch_fund,pile_num,measurement,_ = origin_matrix.shape
    k = int(measurement/a)#卷积矩阵的大小
    print(k)
    c_di_norm2 = np.ones((batch_fund,pile_num,k,k))
    for bf in range(batch_fund):
        for pil in range(pile_num):
            for p in range(k):
                for q in range(k):
                    c_di_norm2[bf][pil][p][q] = np.linalg.norm(origin_matrix[bf,pil,p*a:p*a+a, q*a:q*a+a])
    print(c_di_norm2)
    core = np.ones((a,a)).astype(float)
     #矩阵扩充
    sss = np.ones((batch_fund,pile_num,measurement,measurement))
    for bf in range(batch_fund):
        for pil in range(pile_num):
            for p in range(k):
                for q in range(k):
                    sss[bf,pil,p*a:p*a+a, q*a:q*a+a] = core*c_di_norm2[bf,pil,p,q]
    print(sss)
    return sss

def Sparse_y_pile_batch(origin_matrix,a):
    #a，卷积核心大小
    #输入大小：32*7*20*20
    #pile_num输入的估计数，如输入7个时间片的事故记录去估计下一个时间片
    #32是batch数量
    #c_di_norm2卷积后的矩阵
    #输出大小：32*7*20*20
    batch_fund,pile_num,measurement,_ = origin_matrix.shape
    k = int(measurement/a)#卷积矩阵的大小
    print(k)
    device_in = torch.device('cuda')#cpu训练此处做更改
    c_di_norm2 = torch.ones((batch_fund,pile_num,k,k)).to(device_in)
    for bf in range(batch_fund):
        for pil in range(pile_num):
            for p in range(k):
                for q in range(k):
                    c_di_norm2[bf][pil][p][q] = torch.linalg.norm(origin_matrix[bf,pil,p*a:p*a+a, q*a:q*a+a])
    print(c_di_norm2)
    core = torch.ones((a,a)).to(device_in)
     #矩阵扩充
    sss = torch.ones((batch_fund,pile_num,measurement,measurement)).to(device_in)
    for bf in range(batch_fund):
        for pil in range(pile_num):
            for p in range(k):
                for q in range(k):
                    sss[bf,pil,p*a:p*a+a, q*a:q*a+a] = core*c_di_norm2[bf,pil,p,q]
    print(sss)
    return sss

def admm_attack_one(x_origin,label,model,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map):
    #输入：特征和训练好的模型
    #x_0大小：32*7*20*20
    #x_origin原始输入样本
    #攻击发生在归一化之后
    #未实现批攻击，仅仅是1*7*20*20，输入单样本输出单样本
    det = np.zeros((7, 20,20))
    w = np.zeros((7, 20,20))
    y = np.zeros((7, 20,20))
    z = np.zeros((7, 20,20))
    u = np.zeros((7, 20,20))
    v = np.zeros((7, 20,20))
    s = np.zeros((7, 20,20))

    ro = 0.8
    gama = 0.5
    tao = 0.5
    iteration = 100

    #滑动窗口做卷积
    attack_accident = x_origin[:,:,0,:,:]#32*7*1*20*20
    attack_accident_in = np.squeeze(attack_accident)

    for i in range(iteration):
        a = z - u/ro
        b = z - s/ro
        c = z - v/ro
        #更新det
        det = ro/(ro+2*gama)*a
        #更新 y
        y_di = Sparse_y_pile(c,2)
        y_tem = np.zeros_like(y_di)#中间变量，进行（）+操作
        y_tem1 = np.where((1-tao/(ro*y_di))>=0,1-tao/(ro*y_di),y_tem)
        y = y_tem1*c
        #更新w
        w = w 
        #更新z
        eta = 1/np.sqrt(iteration+1)
        model_xo = model(x_origin, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小20*20
        #修改嵌入
        attack_z = attack_accident_in+z
        attack_input = x_origin
        attack_input[:,:,0,:,:] = attack_z
        model_xo_z = model(attack_input, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小20*20
        div_model_x_z01 = div_model_x_z(attack_input,label, model,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map)
        dev_f = model_xo^3*np.exp(model_xo_z)*div_model_x_z01#还需要升维
        z = (1/(eta+3*ro))*(eta*z+ro*(det+u/ro+w+s/ro+y+v/ro)-dev_f)
        #更新系数
        u = u + ro*(det-z)
        v = v + ro*(y-z)
        s = s + ro*(w-z)
    return attack_input

def notorch_admm_attack(x_origin,label,model,target_time, graph_feature,#这一行都是tensor
                      road_adj, risk_adj, poi_adj, grid_node_map):
    #输入：特征和训练好的模型，32*7*48*20*20
    #输出：32*7*48*20*20
    #x_origin原始输入样本
    #攻击发生在归一化之后
    #批攻击，32*7*1*20*20
    det = np.zeros((32,7, 20,20))
    w = np.zeros((32,7, 20,20))
    y = np.zeros((32,7, 20,20))
    z = np.zeros((32,7, 20,20))
    u = np.zeros((32,7, 20,20))
    v = np.zeros((32,7, 20,20))
    s = np.zeros((32,7, 20,20))

    ro = 0.8
    gama = 0.5
    tao = 0.5
    iteration = 1

    #滑动窗口做卷积
    attack_accident = x_origin[:,:,0,:,:]#32*7*1*20*20
    attack_accident_in = np.squeeze(attack_accident)

    for i in range(iteration):
        a = z - u/ro
        b = z - s/ro
        c = z - v/ro
        #更新det
        det = ro/(ro+2*gama)*a
        #更新 y
        y_di = Sparse_y_pile_batch(c,2)
        y_tem = torch.zeros_like(y_di)#中间变量，进行（）+操作
        y_tem1 = np.where((1-tao/(ro*y_di))>=0,1-tao/(ro*y_di),y_tem)
        y = y_tem1*c
        #更新w
        w = ZINB_distribution(x_origin,b) 
        #更新z,是不是要放到gpu上
        eta = 1/np.sqrt(iteration+1)
        model_xo = model(x_origin, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小32*20*20
        #修改嵌入
        attack_z = attack_accident_in+z
        attack_input = x_origin#attack_input为攻击后的特征*full
        attack_input[:,:,0,:,:] = attack_z
        model_xo_z = model(attack_input, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小32*20*20
        model_xo_z = np.expand_dims(model_xo_z, 1).repeat(7, axis=1)#升维为32*7*20*20 
        div_model_x_z01 = div_model_x_z(attack_input,label, model,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map)
        div_model_x_z01 = np.expand_dims(div_model_x_z01, 1).repeat(7, axis=1)#升维为32*7*20*20 
        dev_f = model_xo^3*np.exp(model_xo_z)*div_model_x_z01
        z = (1/(eta+3*ro))*(eta*z+ro*(det+u/ro+w+s/ro+y+v/ro)-dev_f)
        #更新系数
        u = u + ro*(det-z)
        v = v + ro*(y-z)
        s = s + ro*(w-z)
    return attack_input

def admm_attack(zinb_sample,x_origin,label,model,target_time, graph_feature,#这一行都是tensor
                      road_adj, risk_adj, poi_adj, grid_node_map,device):
    #输入：特征和训练好的模型，32*7*48*20*20
    #输出：32*7*48*20*20
    #x_origin原始输入样本
    #攻击发生在归一化之后
    #批攻击，32*7*1*20*20
    det = torch.zeros((32,7, 20,20)).to(device)
    w = torch.zeros((32,7, 20,20)).to(device)
    y = torch.zeros((32,7, 20,20)).to(device)
    z = torch.zeros((32,7, 20,20)).to(device)
    u = torch.zeros((32,7, 20,20)).to(device)
    v = torch.zeros((32,7, 20,20)).to(device)
    s = torch.zeros((32,7, 20,20)).to(device)

    ro = 0.8
    gama = 0.5
    tao = 0.5
    iteration = 10

    #滑动窗口做卷积
    attack_accident = x_origin[:,:,0,:,:]#32*7*1*20*20
    attack_accident_in = torch.squeeze(attack_accident)

    for i in range(iteration):
        a = z - u/ro
        b = z - s/ro
        c = z - v/ro
        #更新det
        det = ro/(ro+2*gama)*a
        #更新 y
        y_di = Sparse_y_pile_batch(c,2)
        y_tem = torch.zeros_like(y_di).to(device)#中间变量，进行（）+操作
        y_tem1 = torch.where((1-tao/(ro*y_di))>=0,1-tao/(ro*y_di),y_tem)
        y = y_tem1*c
        #更新w
        w = ZINB_distribution(zinb_sample,attack_accident_in,b) 
        #更新z,是不是要放到gpu上
        eta = 1/torch.sqrt(torch.tensor(iteration+1))
        model_xo = model(x_origin, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小32*1*20*20
        #修改嵌入
        attack_z = attack_accident_in+z
        with torch.no_grad():
            attack_input = x_origin#attack_input为攻击后的特征*full
            attack_input[:,:,0,:,:] = attack_z
        model_xo_z = model(attack_input, target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()#大小32*20*20
        model_xo_z = np.expand_dims(model_xo_z, 1).repeat(7, axis=1)#升维为32*7*20*20 
        div_model_x_z01 = div_model_x_z(attack_input,label, model,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map)#torch.Size([32, 7, 48, 20, 20])
        div_model_x_z01 = div_model_x_z01.detach().cpu().numpy()[:,:,0,:,:].squeeze()
        #dev_f = model_xo^3*np.exp(model_xo_z)*div_model_x_z01
        #dev_f = torch.from_numpy(dev_f).to(device)
        dev_f = torch.from_numpy(div_model_x_z01).to(device)
        z = (1/(eta+3*ro))*(eta*z+ro*(det+u/ro+w+s/ro+y+v/ro)-dev_f)
        #更新系数
        u = u + ro*(det-z)
        v = v + ro*(y-z)
        s = s + ro*(w-z)
    return attack_input

def div_model_x_z(x_z,label, net,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map):
    #返回梯度矩阵
    x_z.requires_grad = True  # 先放进去在设置
    x_z.retain_grad()
    with torch.enable_grad():
        loss = nn.MSELoss()(net(x_z, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map),  label)
        loss.backward()
    return x_z.grad.data

def ZINB_distribution_nouse(oringin_xo,b):
    #传入b，干净样本，生成符合分布的迭代样本
    #oringin_xo/b大小32*7*20*20
    #寻找b的前二十位置
    b_sort_f = b.reshape(32,7,-1)
    b_sort = np.sort(b_sort_f,axis=2)
    b_sort1 = b_sort[:,:,379:400]#超过40也没事情
    b_temp = np.ones_like(b)
    b_sort2 = b_temp*b_sort1#32*7*400,0/379
    b_index = np.where(b>b_sort2,1.0,0.0)#需要改变的位置是1.0处，有20个
    deno_origin = oringin_xo*46.0#从deno_origin中推断参数，生成新分布，覆盖位置

    B,N,_,_ = oringin_xo.shape
    for b in range(b):
        for n in range(N):
            pass
    return 0

def nontorch_ZINB_distribution(oringin_xo,b):
    #传入b，从采样出来的w中选择最接近b的
    #oringin_xo:32*7*400
    #b:32*7*400
    zinb_sample = np.load('/home/wyb/mycode/GSNet-master/sanmpling_nor.npy')#1000*400
    #需要进行修改，1000可能太大，采样后没有进行归一化,不要放到函数里多次载入。
    zinb_sample= np.expand_dims(zinb_sample, 0).repeat(32, axis=0)##32*1000*400
    zinb_sample= np.expand_dims(zinb_sample, 1).repeat(7, axis=1)##32**7*1000*400
    original_sample= np.expand_dims(oringin_xo,2).repeat(1000, axis=2)##32*7**1000**400
    w_sample = original_sample-zinb_sample
    min_w_b = w_sample-b#需要最小化的矩阵
    D=np.ones((32,7,1000))
    for i in range(32):
        for j in range(7):
            min_noem = np.linalg.norm(min_w_b[i,j,:,:],axis=1)
            D[i,j,:] = min_noem #32*7*1000 --->32*7
    A=np.ones((32,7))
    for i in range(32):
        d = np.argmin(D[i,:,:])#7*1
        A[i,:] = d#32*7
    #w:32*7*400
    w = np.ones((32,7,400))
    for i in range(32):
        for j in range(7):
            w[i,j,:] = zinb_sample[i,j,A[i,j],:]#从位置找到对应的矩阵
    return w

def ZINB_distribution(zinb_sample,oringin_xo,b):
    #传入b，从采样出来的w中选择最接近b的
    #oringin_xo:32*7*20*20
    #b:32*7*20*20
    #zinb_sample,32*7*1000*400
    #print(oringin_xo.shape)
    original_sample= torch.unsqueeze(oringin_xo,2).repeat(1,1,1000,1,1)##32*7**1000**20*20
    device02 = torch.device('cuda')
    original_sample = original_sample.reshape(32,7,1000,-1).to(device02)#32*7**1000**400
    #400进行操作
    b= torch.unsqueeze(b,2).repeat(1,1,1000,1,1).reshape(32,7,1000,-1)
    w_sample = original_sample-zinb_sample
    min_w_b = w_sample-b#需要最小化的矩阵
    D=torch.ones((32,7,1000)).to(device02)
    for i in range(32):
        for j in range(7):
            min_noem = torch.linalg.norm(min_w_b[i,j,:,:],axis=1)#torch.size([1000])
            D[i,j,:] = min_noem #32*7*1000 --->32*7
    #(D.shape) =torch.Size([32, 7, 1000])
    A=torch.ones((32,7,1)).to(device02)
    for i in range(32):
        for j in range(7):
            d = torch.argmin(D[i,j,:])#7*1
            A[i,j,:] = d#32*7*1
    #w:32*7*400
    A = A.long()#转换数据类型
    w = torch.ones((32,7,400)).to(device02)
    for i in range(32):
        for j in range(7):
            w[i,j,:] = w_sample[i,j,A[i,j,0],:]#从位置找到对应的矩阵
    return w.reshape(32,7,20,20)

def parameter_inference(a,b):
    #参数推断，返回符合参数的分布
    #a提供位置，b提供数字
    a1 = np.argsort(a,axis=2)#a提供位置，b提供数字
    c = np.zeros_like(b)
    a_local = a1[:,:,2:4]#2*3*2
    b_num = np.sort(b,axis=2)[:,:,2:4]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                local = a_local[i,j,k]
                c[i,j,local]=b_num[i,j,k]
    print(c)