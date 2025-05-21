

import numpy as np
import os
import sys
import random
import torch              # PyTorch深度学习框架
import torchvision        # 计算机视觉相关数据集和模型
import torchvision.transforms as transforms    # 图像变换
from utils.dataset_utils import check, separate_data, split_data, save_file   # 自定义数据集工具函数


random.seed(1)
np.random.seed(1)
num_clients = 10      # 客户端数量设为20
dir_path = "MNIST/"   # 数据存储目录路径


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):   #niid：是否非独立同分布(Non-IID) balance: 是否平衡分布 partition: 数据划分方式
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"  # 配置文件路径
    train_path = dir_path + "train/"     # 训练数据路径
    test_path = dir_path + "test/"    # 测试数据路径

    # 检查数据是否已生成，避免重复生成
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    # 定义数据转换：将图像转为张量并标准化（转为PyTorch张量，标准化(均值0.5，标准差0.5)）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # 下载并加载MNIST训练集
    trainset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata",
        train=True,  # 加载训练集
        download=True, # 如果不存在则下载
        transform=transform) # 应用定义的数据转换

    # 下载并加载MNIST测试集
    testset = torchvision.datasets.MNIST(
        root=dir_path+"rawdata",   # 原始数据存储路径
        train=False, # 加载测试集
        download=True,
        transform=transform)

    # 创建数据加载器，批量加载全部数据
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=len(trainset.data),  # 批量大小设为整个数据集
        shuffle=False)   # 不随机打乱
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset.data),
        shuffle=False)

    # 将数据加载到内存中
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []  # 存储图像数据
    dataset_label = []  # 存储标签数据

    # 合并训练集和测试集数据
    dataset_image.extend(trainset.data.cpu().detach().numpy())    # 转换为numpy数组
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    # 转换为numpy数组
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # 计算并打印类别数量
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)

    # 保存生成的数据集
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[0] == "noniid" else False    # 是否非独立同分布
    balance = True if sys.argv[0] == "balance" else False # 是否平衡分布
    partition = sys.argv[0] if sys.argv[0] != "-" else None # 数据划分方式

    generate_dataset(dir_path, num_clients, niid, balance, partition)  # 调用函数生成数据集