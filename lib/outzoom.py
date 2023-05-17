from lib.metrics_chicago import attack_set_by_degree, mask_evaluation_np, risk_change_rate_torch, atc_round, attack_set_by_degree, attack_set_by_pagerank, attack_set_by_betweenness
from lib.metrics_chicago import get_normalized_adj
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
from lib.myfunction import kl_div, js_div, log_test_results, logger_info, view_tensor
import time
import random
from torch.utils.data import DataLoader
import datetime
from lib.aurin import vers_attack001_map

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H) == 8760*48*20*20
        """
        train_temp = np.transpose(
            train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))  # k*48
        self.max = np.max(train_temp, axis=0)  # 48个最大值
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)

        0:risk(numeric,sum)
        1~24:time_period，(one-hot)
        25~31:day_of_week，(one-hot)
        32:holiday，(one-hot)
        33~39:POI (numeric)
        40:temperature (numeric)
        41:Clear,(one-hot)
        42:Cloudy，(one-hot)
        43:Rain，(one-hot)
        44:Snow，(one-hot)
        45:Mist，(one-hot)
        46:inflow(numeric)
        47:outflow(numeric)
        """

        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 32:39] = (data[:, 32:39] - self.min[32:39]) / \
            (self.max[32:39] - self.min[32:39])
        data[:, 40] = (data[:, 40] - self.min[40]) / \
            (self.max[40] - self.min[40])
        data[:, 46] = (data[:, 46] - self.min[46]) / \
            (self.max[46] - self.min[46])
        data[:, 47] = (data[:, 47] - self.min[47]) / \
            (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(
            train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))  # (T*W*H,D)
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33] = (data[:, 33] - self.min[33]) / \
            (self.max[33] - self.min[33])
        data[:, 39] = (data[:, 39] - self.min[39]) / \
            (self.max[39] - self.min[39])
        data[:, 40] = (data[:, 40] - self.min[40]) / \
            (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def mask_loss(predicts, labels, region_mask, data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size, pre_len, _, _ = predicts.shape
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    loss = ((labels-predicts)*region_mask)**2
    if data_type == 'nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type == 'chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 1/17)
        index_3 = (labels > 1/17) & (labels <= 2/17)
        index_4 = labels > 2/17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return torch.mean(loss)


@torch.no_grad()
def compute_loss(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                 grid_node_map, global_step, device,
                 data_type='nyc'):
    """compute val/test loss

    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        device {Device} -- GPU

    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp = []

    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        l = mask_loss(net(feature, target_time, graph_feature, road_adj, risk_adj,
                      poi_adj, grid_node_map), label, risk_mask, data_type)  # l的shape：(1,)
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                         grid_node_map, global_step, scaler, device):
    """predict val/test, return metrics

    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU

    Returns:
        np.float32 -- RMSE，Recall，MAP
        np.array -- label and pre，shape(num_sample,pre_len,W,H)

    """
    # print("数据类型", type(net), type(dataloader), type(risk_mask), type(road_adj), type(grid_node_map),)
    net.eval()
    prediction_list = []
    label_list = []
    for feature, target_time, graph_feature, label in dataloader:
        # print("特征的数据类型", type(feature))
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        # print("the size of feature is".format())
        prediction_list.append(net(feature, target_time, graph_feature,
                               road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    # 1080*1*20*20=34*32*20*20,把34该array拼成一个
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_, risk_change_rate_ = mask_evaluation_np(
        inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, risk_change_rate_, inverse_trans_pre, inverse_trans_label


@torch.no_grad()
def predict_and_lable(net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
                      grid_node_map, global_step, scaler, device):
    """
    return prediction,label

    """
    net.eval()
    prediction_list = []
    label_list = []
    futere_all = []  # 装载future
    for feature, target_time, graph_feature, label in dataloader:
        feature, target_time, graph_feature, label = feature.to(
            device), target_time.to(device), graph_feature.to(device), label.to(device)
        prediction_list.append(net(feature, target_time, graph_feature,
                               road_adj, risk_adj, poi_adj, grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)
    return prediction, label


def k_map(input_map, k):
    # 输入k值和区域矩阵，返回前K个位置
    '''
    0 1 0 0 1
    1 0 1 1 1
    0 0 0 0 1
    1 0 1 1 0
    '''

    d = input.reshape(1, -1)
    d = np.sort(d, axis=1)
    return


def batch_saliency_map(input_grads, k):
    # 输入大小：32*7*48*20*20,数据类型是tensor

    # input_grads = input_grads.mean(dim=0)  # 先样本平均1*7*48*20*20
    # np.save('input_grads.npy', input_grads.cpu().numpy())
    input_grads = torch.where(
        input_grads > 0., input_grads, torch.zeros_like(input_grads))  # 只保留正值
    input_grads = input_grads.mean(dim=0)
    input_grads = input_grads.mean(dim=0)  # 1*1*1*20*20
    # 243个道路区域有多少被包含在里边
    d1 = input_grads.cpu().numpy()  # 寻找k值
    d = d1
    d = d.reshape(1, -1)
    d = np.sort(d, axis=1)  # 变为1维
    kd = d[0][400-k]
    # print(kd)
    node_saliency_map = np.where(d1 < kd, 0, 1)  # 20*20大小的
    return node_saliency_map

# sum_loss的节点选择方法


def item_saliency_map_zz(input_grads, k, batch_size):
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*48*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []
    view_node = []
    # 243个道路区域有多少被包含在里边
    for i in range(batch_size):
        d1 = input_grads.cpu().numpy()  # 寻找k值
        d1 = d1[i, :, :]
        d = d1
        d = d.reshape(1, -1)
        d = np.sort(d, axis=1)  # 变为1维
        kd = d[0][400-k]
        # 查看数据
        view_node.append(kd)
        # print(kd)
        node_saliency_map = np.where(d1 < kd, 0., 1.)  # 20*20大小的
        node_map.append(node_saliency_map)
    np.save('k_node.npy', np.array(view_node))
    return np.array(node_map)


def item_saliency_map_ff(input_grads, k, batch_size):
    # 反向寻找
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*48*20*20
    input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []

    # 243个道路区域有多少被包含在里边
    for i in range(batch_size):
        d1 = input_grads.cpu().numpy()  # 寻找k值
        d1 = d1[i, :, :]
        d = d1
        d = d.reshape(1, -1)
        d = np.sort(d, axis=1)  # 变为1维
        kd = d[0][50]
        # print(kd)
        node_saliency_map = np.where(d1 < kd, 1., 0.)  # 20*20大小的
        node_map.append(node_saliency_map)
    return np.array(node_map)


# 输入标签矩阵或预测矩阵，输出loss,与标签值无关

def item_saliency_map_a(input_grads, k, map_198):
    # 输入大小：32*7*48*20*20,数据类型是tensor
    # 输出：32*20*20

    input_grads = input_grads[:, :, 0, :, :].mean(dim=1)  # 先样本平均32*1*1*20*20
    # input_grads = input_grads.mean(dim=1)  # 先样本平均32*1*1*20*20
    node_map = []
    view_node = []
    # 243个道路区域有多少被包含在里边
    d1 = input_grads.reshape(input_grads.shape[0], 400)
    for i in range(input_grads.shape[0]):

        d2 = d1[i, :].reshape(20, 20)
        dk = d2.clone().reshape(400)
        dk_index = torch.argsort(-dk)
        ds = torch.where(d2 > 0, 1., 0.)
        dn = ds*map_198

        if torch.sum(torch.sum(dn)) > k:
            d2 = torch.where(d2 > dk[dk_index[k]], 1., 0.)
            # print('走过路过')
        else:
            d2 = torch.where(d2 > 0, 1., 0.)
        d3 = d2*map_198
        d333 = d3.reshape(1, 400)
        slect = (k-torch.sum(d3).int()).item()

        # if slect>0 and random.random()>0.2:
        if slect > 0:

            amap = (map_198-d3).reshape(1, 400)
            ss = torch.nonzero(amap)
            a_list = random.sample(range(1, ss.shape[0]), slect)
            for j in a_list:
                d333[ss[j, 0], ss[j, 1]] = 1.0
        d3 = d333.reshape(20, 20)
        d3 = d3.cpu().numpy()
        # 补齐40个点

        node_map.append(d3)
    #hoo = node_saliency_map
    return node_map


def sum_loss(a, risk_mask):
    return torch.sum(a*risk_mask)


def ten_loss(a, risk_mask, k, device):  # a网络输出，
    #h = np.load('weight_cho.npy')
    #h = torch.from_numpy(h).to(device)
    y15 = top_k(k, device)
    y_35 = last_k(50-k, device)
    return torch.sum(a*y_35)-torch.sum(a*y15)


def sum_exp_loss(a, risk_mask):
    return torch.sum(torch.exp(a*risk_mask))


def sum_risk_file(a, pre_num, device):  # a为5、10、20、30、40
    filename = 'data/nyc/risk_mask_{}.npy'.format(pre_num)
    risk_file = np.load(filename)
    risk_file = torch.from_numpy(risk_file)
    risk_file = risk_file.to(device=device)
    return torch.sum(a*risk_file)


def sum_risk_file_get(a, K, device):  # a为5、10、20、30、40
    aaa = np.load('accident_cho_data.npy')
    aaa = aaa.reshape(-1, 400)
    b = np.sum(aaa, axis=0)
    sort_array = np.argsort(-b)
    sort_array = sort_array[:K]
    d = np.zeros(400)
    d[sort_array] = 1
    d = d.reshape(20, 20)

    risk_file = torch.from_numpy(d)
    risk_file = risk_file.to(device=device)
    return torch.sum(a*risk_file)


def ten_loss(a,  k, device):  # a网络输出，
    h = np.load('data/weight_chicago.npy', allow_pickle=True)
    h = torch.from_numpy(h).to(device)
    y15 = top_k(k, device)
    y_35 = last_k(k, device)
    return torch.sum(a*y_35)-torch.sum(a*h*y15)


def top_k(K, device):  # a为5、10、20、30、40
    aaa = np.load('data/accident_cho_data.npy')
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
    aaa = np.load('data/accident_cho_data.npy')
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


def wight_loss(a, b, ro):
    e = torch.where(a > 0, a, ro * torch.ones_like(a))
    d = (a-b)*(a-b)*e
    d = d.sum()
    return d

# 节点选择后进行攻击



def rand_attack(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
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
    # print(risk_mask_use.shape)

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
        X_pgd001 = feature.clone().detach()
        # 生成对抗样本
        for _ in range(num_steps):  # 偷懒写法
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            map, target_time, graph_feature, label = map.to(device), target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            # risk_adj归0测试

            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            # 修改的地方
            eta = torch.clamp(torch.randn(
                size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map*step_size
            # 均匀分布与随机分布
            #eta = torch.clamp(torch.rand(size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        # feature = atc_round(feature,X_pgd001)#攻击后攻击前
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


def log_test_csv(adv_val_predict, val_target, val_predict,  cfgs, select_node_name, select_ack_name, file_name, risk_mask):
    '''
    节点选择方法，攻击方法
    'dataset', 'model','node_select', 'method','K' ,'batch size'
    计算并保存结果'clean_RMSE', 'clean_recaall', 'clean_MAP', 'clean_RCR',
              'adv_RMSE', 'adv_recaall', 'adv_MAP', 'adv_RCR',
              'local_adv_RMSE', 'local_adv_recaall', 'local_adv_MAP', 'local_adv_RCR'

    '''
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(data_set)
    model_name = cfgs.backbone
    metric_list.append(model_name)
    metric_list.append(select_node_name)
    metric_list.append(select_ack_name)
    metric_list.append(cfgs.K)
    metric_list.append(cfgs.batchsize)

    clean_RMSE, clean_recaall, clean_MAP, clean_RCR = mask_evaluation_np(
        val_target, val_predict, risk_mask, 0)
    adv_RMSE, adv_recaall, adv_MAP, adv_RCR = mask_evaluation_np(
        val_target, adv_val_predict, risk_mask, 0)
    local_adv_RMSE, local_adv_recaall, local_adv_MAP, local_adv_RCR = mask_evaluation_np(
        val_predict, adv_val_predict, risk_mask, 0)

    metric_list.append(clean_RMSE)
    metric_list.append(clean_recaall)
    metric_list.append(clean_MAP)
    metric_list.append(clean_RCR)
    metric_list.append(adv_RMSE)
    metric_list.append(adv_recaall)
    metric_list.append(adv_MAP)
    metric_list.append(adv_RCR)
    metric_list.append(local_adv_RMSE)
    metric_list.append(local_adv_recaall)
    metric_list.append(local_adv_MAP)
    metric_list.append(local_adv_RCR)
    # 所有batch跑完，1个epoch保存

    log_test_results(cfgs.model_dir, metric_list, file_name)


def log_test_csv_time_K(epoch, out_K, adv_val_predict, val_target, val_predict,  cfgs, select_node_name, select_ack_name, file_name, risk_mask):
    '''
    节点选择方法，攻击方法
    'dataset', 'model','node_select', 'method','K' ,'batch size'
    计算并保存结果'clean_RMSE', 'clean_recaall', 'clean_MAP', 'clean_RCR',
              'adv_RMSE', 'adv_recaall', 'adv_MAP', 'adv_RCR',
              'local_adv_RMSE', 'local_adv_recaall', 'local_adv_MAP', 'local_adv_RCR'

    '''

    # 获取当前时间
    now = datetime.datetime.now()

    # 将当前时间格式化为字符串
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(time)
    metric_list.append(epoch)
    model_name = cfgs.backbone
    metric_list.append(data_set)
    metric_list.append(model_name)
    metric_list.append(select_node_name)
    metric_list.append(select_ack_name)
    metric_list.append(out_K)
    metric_list.append(cfgs.batchsize)

    clean_RMSE, clean_recaall, clean_MAP, clean_RCR = mask_evaluation_np(
        val_target, val_predict, risk_mask, 0)
    adv_RMSE, adv_recaall, adv_MAP, adv_RCR = mask_evaluation_np(
        val_target, adv_val_predict, risk_mask, 0)
    local_adv_RMSE, local_adv_recaall, local_adv_MAP, local_adv_RCR = mask_evaluation_np(
        val_predict, adv_val_predict, risk_mask, 0)

    metric_list.append(clean_RMSE)
    metric_list.append(clean_recaall)
    metric_list.append(clean_MAP)
    metric_list.append(clean_RCR)
    metric_list.append(adv_RMSE)
    metric_list.append(adv_recaall)
    metric_list.append(adv_MAP)
    metric_list.append(adv_RCR)
    metric_list.append(local_adv_RMSE)
    metric_list.append(local_adv_recaall)
    metric_list.append(local_adv_MAP)
    metric_list.append(local_adv_RCR)
    # 所有batch跑完，1个epoch保存

    log_test_results(cfgs.model_dir, metric_list, file_name)


def log_test_csv_time_K(epoch, out_K, adv_val_predict, val_target, val_predict,  cfgs, select_node_name, select_ack_name, file_name, risk_mask):
    '''
    节点选择方法，攻击方法
    'dataset', 'model','node_select', 'method','K' ,'batch size'
    计算并保存结果'clean_RMSE', 'clean_recaall', 'clean_MAP', 'clean_RCR',
              'adv_RMSE', 'adv_recaall', 'adv_MAP', 'adv_RCR',
              'local_adv_RMSE', 'local_adv_recaall', 'local_adv_MAP', 'local_adv_RCR'

    '''

    # 获取当前时间
    now = datetime.datetime.now()

    # 将当前时间格式化为字符串
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(time)
    metric_list.append(epoch)
    model_name = cfgs.backbone
    metric_list.append(data_set)
    metric_list.append(model_name)
    metric_list.append(select_node_name)
    metric_list.append(select_ack_name)
    metric_list.append(out_K)
    metric_list.append(cfgs.batchsize)

    clean_RMSE, clean_recaall, clean_MAP, clean_RCR = mask_evaluation_np(
        val_target, val_predict, risk_mask, 0)
    adv_RMSE, adv_recaall, adv_MAP, adv_RCR = mask_evaluation_np(
        val_target, adv_val_predict, risk_mask, 0)
    local_adv_RMSE, local_adv_recaall, local_adv_MAP, local_adv_RCR = mask_evaluation_np(
        val_predict, adv_val_predict, risk_mask, 0)

    metric_list.append(clean_RMSE)
    metric_list.append(clean_recaall)
    metric_list.append(clean_MAP)
    metric_list.append(clean_RCR)
    metric_list.append(adv_RMSE)
    metric_list.append(adv_recaall)
    metric_list.append(adv_MAP)
    metric_list.append(adv_RCR)
    metric_list.append(local_adv_RMSE)
    metric_list.append(local_adv_recaall)
    metric_list.append(local_adv_MAP)
    metric_list.append(local_adv_RCR)
    # 所有batch跑完，1个epoch保存

    log_test_results(cfgs.model_dir, metric_list, file_name)


def log_test_csv_time_K_ink(epoch, out_K, in_k, adv_val_predict, val_target, val_predict,  cfgs, select_node_name, select_ack_name, file_name, risk_mask):
    '''
    节点选择方法，攻击方法
    'dataset', 'model','node_select', 'method','K' ,'batch size'
    计算并保存结果'clean_RMSE', 'clean_recaall', 'clean_MAP', 'clean_RCR',
              'adv_RMSE', 'adv_recaall', 'adv_MAP', 'adv_RCR',
              'local_adv_RMSE', 'local_adv_recaall', 'local_adv_MAP', 'local_adv_RCR'

    '''

    # 获取当前时间
    now = datetime.datetime.now()

    # 将当前时间格式化为字符串
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(time)
    metric_list.append(epoch)
    model_name = cfgs.backbone
    metric_list.append(data_set)
    metric_list.append(model_name)
    metric_list.append(select_node_name)
    metric_list.append(select_ack_name)
    metric_list.append(in_k)
    metric_list.append(out_K)
    metric_list.append(cfgs.batchsize)

    clean_RMSE, clean_recaall, clean_MAP, clean_RCR = mask_evaluation_np(
        val_target, val_predict, risk_mask, 0)
    adv_RMSE, adv_recaall, adv_MAP, adv_RCR = mask_evaluation_np(
        val_target, adv_val_predict, risk_mask, 0)
    local_adv_RMSE, local_adv_recaall, local_adv_MAP, local_adv_RCR = mask_evaluation_np(
        val_predict, adv_val_predict, risk_mask, 0)

    metric_list.append(clean_RMSE)
    metric_list.append(clean_recaall)
    metric_list.append(clean_MAP)
    metric_list.append(clean_RCR)
    metric_list.append(adv_RMSE)
    metric_list.append(adv_recaall)
    metric_list.append(adv_MAP)
    metric_list.append(adv_RCR)
    metric_list.append(local_adv_RMSE)
    metric_list.append(local_adv_recaall)
    metric_list.append(local_adv_MAP)
    metric_list.append(local_adv_RCR)
    # 所有batch跑完，1个epoch保存

    log_test_results(cfgs.model_dir, metric_list, file_name)


def vers_attack001(map, feature, device, l_50):
    # 输入：干净样本，32*7*48*20*20
    # 输出：对抗样本：32*7*48*20*20
    # map 32*20*20
    # 第一步攻击效果计算
    # 第二步扰动最小化
    # 7个时间片仅仅攻击最后一个
    # torch编程，输入已经在device上
    # print(zinb_sample.shape)
    a, b, c, d, e = feature.shape
    feature_use = torch.squeeze(feature[:, :, 0, :, :])  # 32*7*20*20

    for i in range(a):  # 32
        a_num = torch.sum(map[i, :, :] > 0.5)  # 寻找位置
        if a_num == 400:
            a_num = 50
        if a_num > 50:
            a_num = 50
        map_use = torch.unsqueeze(map, 1).repeat(1, 7, 1, 1)  # 32*7*20*20
        feature_attack = feature_use*map_use  # 只攻击这些地方
        feature_attack = feature_attack.reshape(a, 7, -1)  # 32*7*400
        feature_new = feature_attack.to(device)  # 32*7*400
        for j in range(7):
            _, zzz = torch.sort(feature_new[i, j, :], descending=True)
            lrand = random.randint(1, 999)
            for k in range(a_num-1):
                # print(lrand)
                # print(feature_new.shape)
                # print(zzz[a_num-k-1])
                feature_new[i, j, zzz[a_num-k-1]] = l_50[lrand, k]
            for k in range(a_num, 400):
                feature_new[i, j, zzz[k]] = 0.
    feature_new = feature_new.reshape(a, 7, 20, 20)
    feature_return = feature.clone().detach()
    feature_return[:, :, 0, :, :] = feature_new
    # print(torch.sum(feature==feature_return))
    # print((feature_return[1,1,0,:,:]==feature[1,1,0,:,:]).min())
    return feature_return


def node_map(select_name, chi_k, out_K, in_k, cfgs, net, dataloader, risk_mask, road_adj, risk_adj, poi_adj,
             grid_node_map, device, data_type='nyc'):
    # 输入特征，模型和标签
    # y是标签
    # select_name:节点选择方法
    # Random:随机处理
    # num_nodes：节点数量
    # K：选择攻击的节点数量
    for_random = cfgs.for_random
    num_nodes = cfgs.num_nodes
    K = out_K
    step_size = cfgs.step_size
    epsilon = cfgs.epsilon
    net.train()
    map = []
    if cfgs.dataset == 'Chicago':
        len_cho = 495
        wise = pd.read_pickle('data/chicago/risk_adj.pkl')
        mak = np.load('data/chicago/197_chicago.npy')
        mak = np.expand_dims(mak, 0).repeat(len_cho, axis=0).reshape(-1, 400)
       # with torch.no_grad():
        if select_name == 'none':
            a = np.ones((len_cho, 20, 20))
            return a
        if select_name == 'pagerank':
            index = attack_set_by_pagerank(wise, K)
            a = np.zeros(400)
            a[index] = 1
            a = a.reshape(20, 20)
            a = np.expand_dims(a, 0).repeat(len_cho, axis=0)
            return a
        if select_name == 'Centrality':
            index = attack_set_by_betweenness(wise, K)
            a = np.zeros(400)
            a[index] = 1
            a = a.reshape(20, 20)
            a = np.expand_dims(a, 0).repeat(len_cho, axis=0)
            return a
        if select_name == 'Degree':
            index = attack_set_by_degree(wise, K)
            a = np.zeros(400)
            a[index] = 1
            a = a.reshape(20, 20)
            a = np.expand_dims(a, 0).repeat(len_cho, axis=0)
            return a
        if select_name == 'random':
            # 返回np矩阵1080*20*20，值为1 的地方为选中的节点(K个)
            a = np.zeros((len_cho, 20, 20),)
            b = a.reshape(len_cho, -1)
            for len_future in range(len_cho):
                list = [i for i in range(num_nodes)]
                index = random.sample(list, K)
                b[len_future][index] = 1
            return (b*mak).reshape(len_cho, 20, 20)

        if select_name == 'top_history':
            # 返回np矩阵1080*20*20，值为1 的地方为选中的节点(K个)
            aaa = np.load('accident_cho_data.npy')
            aaa = aaa.reshape(-1, 400)
            b = np.sum(aaa, axis=0)
            sort_array = np.argsort(-b)
            sort_array = sort_array[:K]
            d = np.zeros(400)
            d[sort_array] = 1
            d = d.reshape(20, 20)
            return np.expand_dims(d, 0).repeat(len_cho, axis=0)

        if select_name == 'saliency_loss_mse':
            for feature, target_time, graph_feature, label in dataloader:
                # 选择敏感节点
                X_saliency = feature
                X = feature
                saliency_steps = 5
                target_time, graph_feature, label = target_time.to(
                    device), graph_feature.to(device), label.to(device)
                risk_mask_use = torch.from_numpy(risk_mask).to(device)
                X = X.to(device)
                X_saliency = X_saliency.to(device)
                X_saliency.requires_grad = True  # 先放进去在设置
                X_saliency.retain_grad()

                net.to(device)
                for _ in range(saliency_steps):
                    if for_random:
                        random_noise = torch.FloatTensor(
                            *X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                        X_saliency = Variable(
                            X_saliency.data + random_noise, requires_grad=True)
                    opt_saliency = optim.SGD([X_saliency], lr=1e-3)
                    opt_saliency.zero_grad()
                    with torch.enable_grad():
                        loss_saliency = nn.MSELoss()(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                         grid_node_map), label)
                    loss_saliency.backward()

                    eta_s = step_size * X_saliency.grad.data.sign()
                    inputs_grad = X_saliency.grad.data
                    # X_saliency随着迭代变化
                    X_saliency = Variable(
                        X_saliency.data + eta_s, requires_grad=True)
                    eta_s = torch.clamp(
                        X_saliency.data - X.data, -epsilon, epsilon)
                    X_saliency = Variable(X.data + eta_s, requires_grad=True)
                    X_saliency = Variable(torch.clamp(
                        X_saliency, 0, 1.0), requires_grad=True)
                saliency_map = item_saliency_map_zz(
                    inputs_grad, K, feature.shape[0])
                map.append(saliency_map)
            select_name = np.concatenate(map, 0)
            print(select_name.shape)
            return select_name
        if select_name == 'HGNS':  
            for feature, target_time, graph_feature, label in dataloader:
                X_saliency = feature
                X = feature
                target_time, graph_feature, label = target_time.to(
                    device), graph_feature.to(device), label.to(device)
                risk_mask_use = torch.from_numpy(risk_mask).to(device)
                X = X.to(device)
                X_saliency = X_saliency.to(device)
                X_saliency.requires_grad = True  
                X_saliency.retain_grad()

                net.to(device)

                if for_random:
                    random_noise = torch.FloatTensor(
                        *X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                    X_saliency = Variable(
                        X_saliency.data + random_noise, requires_grad=True)
                opt_saliency = optim.SGD([X_saliency], lr=1e-3)
                opt_saliency.zero_grad()
                with torch.enable_grad():
                    loss_saliency = ten_loss(net(X_saliency, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                                 grid_node_map), chi_k, device)  # 12是风险最高的前12个区域
                loss_saliency.backward()
                inputs_grad = X_saliency.grad.data
                # 看看梯度信息
                map_view = inputs_grad[5, 0, 0, :, :].clone(
                ).detach().cpu().numpy().reshape(20, 20)
                map_view = np.where(map_view > 0.00005, 1, 0)
                map_198 = np.load('data/chicago/197_chicago.npy')
                map_198 = torch.from_numpy(map_198)
                map_198 = map_198.to(device)
                saliency_map = item_saliency_map_a(
                    inputs_grad, K, map_198)  # 此处K是攻击节点的位置
                map.append(saliency_map)
                # 可视化
                #map_view = saliency_map[5,:,:].copy()
                # sb.heatmap(map_view)
                # map.fu()

            select_name = np.concatenate(map, 0)
            # print(select_name.shape)
            np.save('map_{}.npy'.format(K), select_name)
            return select_name


def vers_attack002(hours, feature):
    # feature:32*7*48*20*20
    # 32维度，每个样本有个时间
    # a:32*7*24*20*20
    if hours == 0:
        return feature

    for i in range(feature.shape[0]):  # 样本
        for j in range(feature.shape[1]):  # 时间片
            a = torch.zeros(24, feature.shape[3], feature.shape[4])
            # print(feature[i,j,1:25,1,1])
            if torch.sum(feature[i, j, 1:25, 1, 1]) != 1.0:
                s = random.randint(0, 4)
                # print('错误出现')
                # print(feature[i,j,1:25,1,1])
            else:
                s = np.nonzero(feature[i, j, 1:25, 1, 1]).item()  # 寻找index
            # print(feature[i,j,1:25,1,1])
            # print(s)
            #s =[s,s[1000+i]]
            if s+hours > 23:
                a[s, :, :] = torch.zeros(20, 20)
                a[s+hours-24, :, :] = torch.ones(20, 20)
            else:
                k = s
                a[k, :, :] = torch.zeros(20, 20)
                a[k+hours, :, :] = torch.ones(20, 20)
            feature[i, j, 1:25, :, :] = a
    return feature


def vers_attack003(days, feature):  # 7days
    # feature:32*7*48*20*20
    # 32维度，每个样本有个时间
    # a:32*7*7*20*20
    if days == 0:
        return feature

    for i in range(feature.shape[0]):  # 样本
        for j in range(feature.shape[1]):  # 时间片
            a = torch.zeros((7, feature.shape[3], feature.shape[4]))
            # print(feature[i,j,25:32,1,1])
            if torch.sum(feature[i, j, 25:32, 1, 1] == 0) != 6:
                s = random.randint(0, 4)
                # print(feature[i,j,25:32,1,1])
            else:
                s = np.nonzero(feature[i, j, 25:32, 1, 1]).item()  # 寻找index
            if s+days > 6:
                a[s, :, :] = torch.zeros(20, 20)
                a[s+days-7, :, :] = torch.ones(20, 20)
            else:
                k = s
                a[k, :, :] = torch.zeros(20, 20)
                a[k+days, :, :] = torch.ones(20, 20)
            feature[i, j, 25:32, :, :] = a
    return feature


def vers_attack004(weathers, feature):
    # feature:32*7*48*20*20
    # 32维度，每个样本有个时间
    # a:32*7*7*20*20
    if weathers == 0:
        return feature

    for i in range(feature.shape[0]):  # 样本
        for j in range(feature.shape[1]):  # 时间片
            a = torch.zeros((5, feature.shape[3], feature.shape[4]))
            if torch.sum(feature[i, j, 41:46, 1, 1] == 0) != 5:
                s = random.randint(0, 4)
            else:
                s = np.nonzero(feature[i, j, 41:46, 1, 1]).item()  # 寻找index
            if s+weathers > 4:
                a[s, :, :] = torch.zeros(20, 20)
                a[s+weathers-5, :, :] = torch.ones(20, 20)
            else:
                k = s
                a[k, :, :] = torch.zeros(20, 20)
                a[k+weathers, :, :] = torch.ones(20, 20)
            feature[i, j, 41:46, :, :] = a
    return feature


def vers_attack005(feature):
    feature[:, :, 1:33, :, :] = torch.round(feature[:, :, 1:33, :, :])
    return feature


def STZINB(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
           grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    不可感知
    '''
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    Random_noise = cfgs.Random_noise
    days = cfgs.days
    hours = cfgs.hours
    weather = cfgs.weather

    net.train()
    acc_prediction_list = []
    label_list = []
    clean_prediction_list = []
    a = 0
    batch_idx = 0
    batch_num_x = 0
    unable = []
    # with torch.no_grad():
    # 载入map数据
    map_loader = DataLoader(dataset=map, batch_size=32, shuffle=False)
    ziped = zip(map_loader, dataloader)  # 封装map的迭代器

    risk_mask_use = np.expand_dims(risk_mask, 0).repeat(32, axis=0)
    risk_mask_use = torch.from_numpy(risk_mask_use).to(device)

    l_50 = np.load('data/30arroy.npy')/46.0
    l_50 = torch.from_numpy(l_50).to(device)
    if cfgs.dataset == 'NYC':
        a_grid_ori = pd.read_pickle('data/nyc/grid_node_map.pkl')  # 400*243
        a_grid_ori = np.expand_dims(
            a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243
        a_hnum = 1080
    else:
        a_grid_ori = pd.read_pickle(
            'data/chicago/grid_node_map.pkl')  # 400*243
        a_grid_ori = np.expand_dims(
            a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243
        a_hnum = 495
    for map, test_loader_ues in ziped:

        batch_idx += 1
        # map 32*20*20,升维
        map_01 = map.clone().detach().to(device)  # risk用
        if cfgs.dataset == 'NYC':
            map = np.expand_dims(map, 1).repeat(48, axis=1)  # 32*48*20*20
        else:
            map = np.expand_dims(map, 1).repeat(41, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        a_grid_ori_a = a_grid_ori.copy()
        a_grid_ori_a = np.expand_dims(a_grid_ori_a, 0).repeat(
            feature.shape[0], axis=0)  # 32*7*400*243
        a_grid_ori_b = torch.from_numpy(
            a_grid_ori_a).to(torch.float).to(device)
        start = time.time()
        X_pgd001 = feature.clone().detach()
        # graph_feature攻击

        # 生成对抗样本

        if Random_noise:
            random_noise = torch.FloatTensor(
                *feature.shape).uniform_(-epsilon/10, epsilon/10)
            # print(feature.shape)
            # print(random_noise.shape)
            # print(map.shape)
            feature = Variable(feature.data + map *
                               random_noise, requires_grad=True)
        map = map.to(device)
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            target_time, graph_feature, label = target_time.to(
                device), graph_feature.to(device), label.to(device)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = kl_div(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                  grid_node_map),  label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..

            eta = step_size * X_pgd.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        # feature = atc_round(feature,X_pgd001)#攻击后攻击前
        X_pgd001 = X_pgd001.to(device)
        # 计算不可感知性
        unable.append(torch.sum(feature[:, :, 1:25, :, :] > 0))
        # 组合攻击方法
        # 1.risk攻击
        feature = vers_attack001_map(map_01, feature, device, l_50)
        #jilo = vers_attack001(map_01, X_pgd001, device, l_50)
        # 2.时间轮转攻击

        # 24Hours
        feature = vers_attack002(hours, feature)
        # 7days
        feature = vers_attack003(days, feature)
        # weather
        feature = vers_attack004(weather, feature)
        # holiday
        feature = vers_attack005(feature)
        # 组合攻击结束
        target_time_adv = target_time.clone().detach()
        graph_feature_adv = graph_feature.clone().detach()

        target_time_adv = feature[:, 6, 1:33, 5, 5]  # 任选一个地方，时间片6
        # noda_map
        # a_grid:32*7*400*243
        # 32*7*1*400  32*7*400*243
        hi = feature.shape[0]
        ddd = torch.matmul(feature[:, :, 0, :, :].reshape(
            hi, 7, 400).unsqueeze(2), a_grid_ori_b).squeeze()  # 32*7*1*400

        graph_feature_adv[:, :, 1, :] = ddd
        if cfgs.dataset == 'NYC':
            graph_feature_adv[:, :, 1:3, :] = torch.matmul(
                feature[:, :, 46:48, :, :].reshape(hi, 7, 2, 400), a_grid_ori_b)
        else:
            graph_feature_adv[:, :, 1:3, :] = torch.matmul(
                feature[:, :, 39:41, :, :].reshape(hi, 7, 2, 400), a_grid_ori_b)
        # 每个batch的三个量
        # 测试
        # ck2 = jilo[5, 6, 0, :, :]*46.0  # 32*7*48*20*20
        #ck2 = ck2.detach().cpu().numpy()
        #np.save('ck3.npy', ck2)
        # fea.dull()

        ###
        batch_adv = net(feature, target_time_adv, graph_feature_adv,  # 32*7*3*243//32*32
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
                batch_num_x, a_hnum,
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
    #logger_info(logger, False, 'Info:  不可感知性质：{}'.format(torch.sum(torch.tensor(unable).detach().cpu()/1080./7/20/20).item()))
    logger_info(logger, False, '攻击时间：Info:time:{:.3f} \t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f}'.format(
                time.time() - start,
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre



def min_impr(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
             grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    min攻击方法
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

    if cfgs.dataset == 'NYC':
        a_grid_ori = pd.read_pickle('data/nyc/grid_node_map.pkl')  # 400*243
        a_grid_ori = np.expand_dims(
            a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243
        a_hnum = 1080
    else:
        a_grid_ori = pd.read_pickle(
            'data/chicago/grid_node_map.pkl')  # 400*243
        a_grid_ori = np.expand_dims(
            a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243
        a_hnum = 495
    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(41, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature.clone().detach()
        X_pgd001 = X_pgd001.to(device)
        # 添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)
        a_grid_ori_a = a_grid_ori.copy()
        a_grid_ori_a = np.expand_dims(a_grid_ori_a, 0).repeat(
            feature.shape[0], axis=0)  # 32*7*400*243
        a_grid_ori_b = torch.from_numpy(
            a_grid_ori_a).to(torch.float).to(device)

        previous_grad = torch.zeros_like(feature.data)
        previous_grad = previous_grad.to(device)
        # 生成对抗样本
        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)
            feature = feature.to(device)
            net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..
            grad = X_pgd.grad.data / \
                torch.mean(torch.abs(X_pgd.grad.data),
                           [1, 2, 3, 4], keepdim=True)
            previous_grad = torch.tensor(
                1.0) * previous_grad + grad  # 1.0可以是一个参数，写入cfgs文件配置中
            X_pgd = Variable(X_pgd.data + step_size *
                             previous_grad.sign(), requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            feature = X_pgd
        eta = torch.clamp(X_pgd.data - X_pgd001.data, -epsilon, epsilon) * map
        X_pgd = Variable(X_pgd001.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd_round = atc_round(X_pgd, X_pgd001)  # 攻击后攻击前
        # print(torch.sum(X_pgd[:,:,1:33,:,:]!=(vote_hours(X_pgd,device)[:,:,1:33,:,:])))
        X_pgd_round = X_pgd_round.to(device)

        # 每个batch的三个量
        X_pgd_h = vote_hours(X_pgd_round, device)
        X_pgd_d = vote_days(X_pgd_h, device)
        X_pgd_ho = vote_holiday(X_pgd_d, device)
        X_pgd_w = vote_weather_cho(X_pgd_ho, device)
        # print(torch.sum(X_pgd_w[:,:,41:46,:,:]!=X_pgd_ho[:,:,41:46,:,:]))
        # print(X_pgd_h.diss())
        graph_feature_adv = graph_feature.clone().detach()
        hi = X_pgd_w.shape[0]
        ddd = torch.matmul(X_pgd_w[:, :, 0, :, :].reshape(
            hi, 7, 400).unsqueeze(2), a_grid_ori_b).squeeze()  # 32*7*1*400
        graph_feature_adv[:, :, 1, :] = ddd
        graph_feature_adv[:, :, 1:3, :] = torch.matmul(
            X_pgd_w[:, :, 39:41, :, :].reshape(hi, 7, 2, 400), a_grid_ori_b)
        # 每个batch的三个量
        batch_adv = net(X_pgd_w, target_time, graph_feature_adv,
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
                batch_num_x, a_hnum,
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
    logger_info(logger, False, '攻击时间：Info:time:{:.3f} \t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f}'.format(
                time.time() - start,
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def pgd_impr(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
             grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    fgsm攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    # fgsm只迭代一次
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    Random_noise = cfgs.Random_noise
    att_round = cfgs.att_round
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

    a_grid_ori = pd.read_pickle('data/chicago/grid_node_map.pkl')  # 400*243
    a_grid_ori = np.expand_dims(a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(41, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature.clone().detach()
        # 不添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)
        feature = feature.to(device)
        a_grid_ori_a = a_grid_ori.copy()
        a_grid_ori_a = np.expand_dims(a_grid_ori_a, 0).repeat(
            feature.shape[0], axis=0)  # 32*7*400*243
        a_grid_ori_b = torch.from_numpy(
            a_grid_ori_a).to(torch.float).to(device)
        # 生成对抗样本
        if Random_noise:
            random_noise = torch.FloatTensor(
                *feature.shape).uniform_(-epsilon/10, epsilon/10).cuda()
            feature = Variable(feature.data + map *
                               random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)

            net = net.to(device)
            feature.retain_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(net(feature, target_time, graph_feature, road_adj, risk_adj, poi_adj,
                                        grid_node_map), label)
            loss.backward()

            X_pgd = feature  # 脱裤子放屁..一个bug,1670行查看会出错X_pgd.grad.data
            eta = step_size * feature.grad.data.sign()*map
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - feature.data, -epsilon, epsilon)
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        if att_round:
            X_pgd_round = atc_round(feature, X_pgd001)  # 攻击后攻击前
        X_pgd001 = X_pgd001.to(device)
        # 每个batch的三个量
        X_pgd_h = vote_hours(X_pgd_round, device)
        X_pgd_d = vote_days(X_pgd_h, device)
        X_pgd_ho = vote_holiday(X_pgd_d, device)
        X_pgd_w = vote_weather_cho(X_pgd_ho, device)

        feature_h = vote_hours(feature, device)

        # ----------##########取整就行
        graph_feature_adv = graph_feature.clone().detach()

        hi = feature.shape[0]
        ddd = torch.matmul(X_pgd_w[:, :, 0, :, :].reshape(
            hi, 7, 400).unsqueeze(2), a_grid_ori_b).squeeze()  # 32*7*1*400

        graph_feature_adv[:, :, 1, :] = ddd
        graph_feature_adv[:, :, 1:3, :] = torch.matmul(
            X_pgd_w[:, :, 39:41, :, :].reshape(hi, 7, 2, 400), a_grid_ori_b)
        # 每个batch的三个

        batch_adv = net(X_pgd_w, target_time, graph_feature_adv,
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
    logger_info(logger, False, '攻击时间：Info:time:{:.3f} \t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f}'.format(
                time.time() - start,
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def random_impr(logger, cfgs, map, net, dataloader,  road_adj, risk_adj, poi_adj,
                grid_node_map, scaler, risk_mask, device, data_type='nyc'):
    '''
    fgsm攻击方法
    logger:日志保存
    cfgs：配置信息
    num_steps:迭代次数
    step_size：步长
    epsilon,：扰动大小
    map:攻击节点矩阵
    '''
    # fgsm只迭代一次
    num_steps = cfgs.ack_num_steps
    step_size = cfgs.ack_step_size
    epsilon = cfgs.ack_epsilon
    Random_noise = cfgs.Random_noise
    att_round = cfgs.att_round
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

    a_grid_ori = pd.read_pickle('data/chicago/grid_node_map.pkl')  # 400*243
    a_grid_ori = np.expand_dims(a_grid_ori, 0).repeat(7, axis=0)  # 7*400*243

    for map, test_loader_ues in ziped:
        batch_idx += 1
        # map 32*20*20,升维
        map = np.expand_dims(map, 1).repeat(41, axis=1)  # 32*48*20*20
        map = np.expand_dims(map, 1).repeat(7, axis=1)  # 32*7*48*20*20
        map = map.astype(np.float32)
        map = torch.from_numpy(map)
        feature, target_time, graph_feature, label = test_loader_ues
        # 对特征进行攻击
        start = time.time()
        X_pgd001 = feature.clone().detach()
        # 不添加随机噪声
        map, target_time, graph_feature, label = map.to(device), target_time.to(
            device), graph_feature.to(device), label.to(device)
        feature = feature.to(device)
        a_grid_ori_a = a_grid_ori.copy()
        a_grid_ori_a = np.expand_dims(a_grid_ori_a, 0).repeat(
            feature.shape[0], axis=0)  # 32*7*400*243
        a_grid_ori_b = torch.from_numpy(
            a_grid_ori_a).to(torch.float).to(device)
        # 生成对抗样本
        if Random_noise:
            random_noise = torch.FloatTensor(
                *feature.shape).uniform_(-epsilon/10, epsilon/10).cuda()
            feature = Variable(feature.data + map *
                               random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([feature], lr=1e-3)
            opt.zero_grad()
            feature = Variable(feature.data, requires_grad=True)

            X_pgd = feature  # 脱裤子放屁..一个bug,1670行查看会出错X_pgd.grad.data
            eta = torch.clamp(torch.randn(
                size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map*step_size
            # 均匀分布与随机分布
            #eta = torch.clamp(torch.rand(size=X_pgd.shape).cuda()-0.5, -epsilon, epsilon)*map
            X_pgd = Variable(feature.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                             requires_grad=True)
            feature = X_pgd
        if att_round:
            X_pgd_round = atc_round(feature, X_pgd001)  # 攻击后攻击前
        X_pgd001 = X_pgd001.to(device)

        # ----------##########取整就行
        X_pgd_round = X_pgd_round.to(device)
        X_pgd_h = vote_hours(X_pgd_round, device)
        X_pgd_d = vote_days(X_pgd_h, device)
        X_pgd_ho = vote_holiday(X_pgd_d, device)
        X_pgd_w = vote_weather_cho(X_pgd_ho, device)

        graph_feature_adv = graph_feature.clone().detach()
        hi = feature.shape[0]
        ddd = torch.matmul(X_pgd_round[:, :, 0, :, :].reshape(
            hi, 7, 400).unsqueeze(2), a_grid_ori_b).squeeze()  # 32*7*1*400

        graph_feature_adv[:, :, 1, :] = ddd
        graph_feature_adv[:, :, 1:3, :] = torch.matmul(
            X_pgd_round[:, :, 39:41, :, :].reshape(hi, 7, 2, 400), a_grid_ori_b)
        # 每个batch的三个量
        batch_adv = net(X_pgd_w, target_time, graph_feature_adv,
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
                batch_num_x, 495,
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
    logger_info(logger, False, '攻击时间：Info:time:{:.3f} \t  clean_RMSE: {:.4f} clean_Recall: {:.4f} clean_MAP: {:.4f} clean_RCR: {:.4f}   adv_RMSE: {:.4f} adv_Recall: {:.4f} adv_MAP: {:.4f} adv_RCR: {:.4f}'.format(
                time.time() - start,
                clean_RMSE, clean_Recall, clean_MAP, clean_RCR,
                adv_RMSE, adv_Recall, adv_MAP, adv_RCR,))

    return inverse_trans_pre, inverse_trans_label,  inverse_trans_clean_pre


def vote_hours(feature, device):
    # feature 32/24*7*48*20*20
    a, b, c, _, _ = feature.shape
    feature = feature.detach()
    for i in range(a):
        for j in range(b):
            vote_data = feature[i, j, 1:25, :, :].reshape(24, 400)
            vota_index = torch.sum(vote_data, axis=1)  # 24票
            vota_index = torch.argmax(vota_index)  # 获取位置
            if vota_index == 23:
                vota_index = 0
            feature[i, j, vota_index+1, :, :] = torch.ones(20, 20).to(device)
            if vota_index == 0:
                feature[i, j, 2:25, :, :] = torch.zeros(23, 20, 20).to(device)
            elif vota_index == 23:
                feature[i, j, 1:24, :, :] = torch.zeros(23, 20, 20).to(device)
            else:
                feature[i, j, 1:vota_index+1, :,
                        :] = torch.zeros(vota_index, 20, 20).to(device)
                feature[i, j, vota_index+2:25, :,
                        :] = torch.zeros(23-vota_index, 20, 20).to(device)
    return feature


def vote_days(feature, device):
    # feature 32/24*7*48*20*20--25-31
    a, b, c, _, _ = feature.shape
    feature = feature.detach()
    for i in range(a):
        for j in range(b):
            vote_data = feature[i, j, 25:32, :, :].reshape(7, 400)
            vota_index = torch.sum(vote_data, axis=1)  # 7票
            vota_index = torch.argmax(vota_index)  # 获取位置0-6
            if vota_index == 6:
                vota_index = 0
            feature[i, j, vota_index+25, :, :] = torch.ones(20, 20).to(device)
            if vota_index == 0:
                feature[i, j, 26:32, :, :] = torch.zeros(6, 20, 20).to(device)
            elif vota_index == 6:
                feature[i, j, 25:31, :, :] = torch.zeros(6, 20, 20).to(device)
            else:
                feature[i, j, 25:25+vota_index-1, :,
                        :] = torch.zeros(vota_index-1, 20, 20).to(device)
                feature[i, j, vota_index+26:32, :,
                        :] = torch.zeros(6-vota_index, 20, 20).to(device)
    return feature


def vote_weather(feature, device):
    # feature 32/24*7*48*20*20(41-45)25-31
    a, b, c, _, _ = feature.shape
    feature = feature.detach()
    for i in range(a):
        for j in range(b):
            vote_data = feature[i, j, 41:46, :, :].reshape(5, 400)
            vota_index = torch.sum(vote_data, axis=1)  # 5票
            vota_index = torch.argmax(vota_index)  # 获取位置
            if vota_index == 4:
                vota_index = 0
            feature[i, j, vota_index+41, :, :] = torch.ones(20, 20).to(device)
            if vota_index == 0:
                feature[i, j, 42:46, :, :] = torch.zeros(4, 20, 20).to(device)
            elif vota_index == 23:
                feature[i, j, 41:45, :, :] = torch.zeros(4, 20, 20).to(device)
            else:
                feature[i, j, 41:41+vota_index-1, :,
                        :] = torch.zeros(vota_index-1, 20, 20).to(device)
                feature[i, j, vota_index+42:46, :,
                        :] = torch.zeros(4-vota_index, 20, 20).to(device)
    return feature


def vote_weather_cho(feature, device):
    # feature 32/24*7*48*20*20(41-45)35-39
    a, b, c, _, _ = feature.shape
    feature = feature.detach()
    for i in range(a):
        for j in range(b):
            vote_data = feature[i, j, 35:40, :, :].reshape(5, 400)
            vota_index = torch.sum(vote_data, axis=1)  # 5票
            vota_index = torch.argmax(vota_index)  # 获取位置
            if vota_index == 4:
                vota_index = 0
            feature[i, j, vota_index+35, :, :] = torch.ones(20, 20).to(device)
            if vota_index == 0:
                feature[i, j, 36:40, :, :] = torch.zeros(4, 20, 20).to(device)
            elif vota_index == 4:
                feature[i, j, 35:39, :, :] = torch.zeros(4, 20, 20).to(device)
            else:
                feature[i, j, 35:35+vota_index-1, :,
                        :] = torch.zeros(vota_index-1, 20, 20).to(device)
                feature[i, j, vota_index+36:40, :,
                        :] = torch.zeros(4-vota_index, 20, 20).to(device)
    return feature


def vote_holiday(feature, device):
    # feature 32/24*7*48*20*20
    a, b, c, _, _ = feature.shape
    feature = feature.detach()
    for i in range(a):
        for j in range(b):
            vote_data = feature[i, j, 32, :, :].reshape(400)
            vota_index = torch.sum(vote_data)
            if vota_index > 200:
                feature[i, j, 32, :, :] = torch.ones(20, 20).to(device)
            else:
                feature[i, j, 32, :, :] = torch.zeros(20, 20).to(device)
    return feature
