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

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

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
    print(risk_mask_use.shape)

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature#32*7*48*20*20
        # 生成对抗样本
        #取出等待攻击的事故特征
        attack_accident = feature[:,:,0,:,:]#32*7*1*20*20
        attack_accident_in = np.squeeze(attack_accident)
        #将修改后的值嵌入，生成新的特征
        X_pgd001 = admm_attack(feature,net,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map)

        #
        X_pgd001 = X_pgd001.to(device)
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

def admm_attack(x_origin,model,target_time, graph_feature,
                      road_adj, risk_adj, poi_adj, grid_node_map):
    #输入：特征和训练好的模型
    #x_0大小：32*7*400
    #x_origin原始输入样本
    #攻击发生在归一化之后
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
                      road_adj, risk_adj, poi_adj, grid_node_map).detach().cpu().numpy()
        dev_f = model_xo^3*np.exp(model(x_0+z))*div_model_x_z
        z = (1/(eta+3*ro))*(eta*z+ro*(det+u/ro+w+s/ro+y+v/ro)-dev_f)
        #更新系数
        u = u + ro*(det-z)
        v = v + ro*(y-z)
        s = s + ro*(w-z)
    return (x_0+z)