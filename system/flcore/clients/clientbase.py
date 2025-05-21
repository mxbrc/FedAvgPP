
import copy
import torch
import torch.nn as nn
import numpy as np
import os
import random
import hashlib
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_der_public_key
from utils.data_utils import read_client_data


class Client(object):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        # 用于盲化的种子和客户端列表
        self.num_clients = args.num_clients
        self.seeds = {}
        self.clients = []
        self.enable_blinding = args.enable_blinding if hasattr(args, 'enable_blinding') else False
        self.initialize_seeds()

        # 用于签名的密钥对
        self.enable_signing = args.enable_signing if hasattr(args, 'enable_signing') else False
        self.private_key = None
        self.public_key = None
        self.server_public_key = None
        if self.enable_signing:
            self.generate_keys()

    def generate_keys(self):
        """生成ECDSA密钥对"""
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    def set_server_public_key(self, server_public_key):
        """设置服务器公钥"""
        self.server_public_key = load_der_public_key(server_public_key)

    def initialize_seeds(self):
        if not self.enable_blinding:
            return
        for j in range(self.num_clients):
            if j != self.id:
                seed = int(f"{min(self.id, j)}{max(self.id, j)}")
                self.seeds[j] = seed

    def set_clients(self, clients):
        self.clients = clients

    def blind_parameters(self):
        if not self.enable_blinding:
            return {name: param.data.clone() for name, param in self.model.named_parameters()}

        blinded_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        for j in range(self.num_clients):
            if j == self.id:
                continue
            random.seed(self.seeds[j])
            sign = 1 if self.id < j else -1
            for name, param in self.model.named_parameters():
                mask = torch.tensor(np.random.normal(0, 0.001, param.shape), device=self.device, dtype=param.dtype)
                blinded_params[name] += sign * mask
        for name, param in blinded_params.items():
            param.clamp_(-1e10, 1e10)
        return blinded_params

    def sign_parameters(self, params):
        """对盲化参数生成ECDSA签名"""
        if not self.enable_signing:
            return None, None

        # 简化CRT：将参数按层分割，计算每层的哈希
        param_hashes = []
        for name, param in params.items():
            param_bytes = param.detach().cpu().numpy().tobytes()
            hash_obj = hashlib.sha256(param_bytes)
            param_hashes.append(hash_obj.hexdigest())

        # 组合哈希值和客户端ID
        message = f"{self.id}:{':'.join(param_hashes)}".encode()
        hash_obj = hashlib.sha256(message)
        signature = self.private_key.sign(hash_obj.digest(), ec.ECDSA(hashes.SHA256()))
        return hash_obj.hexdigest(), signature

    def verify_aggregated_parameters(self, aggregated_params, signature, client_ids, client_hashes):
        """验证聚合梯度的签名"""
        if not self.enable_signing:
            return True
        if not signature:
            print(f"Client {self.id}: No signature provided for aggregated parameters")
            return False  # 强制要求签名

        try:
            # 重新计算聚合梯度的哈希
            aggregated_hashes = []
            for name, param in aggregated_params.items():
                param_bytes = param.cpu().numpy().tobytes()
                hash_obj = hashlib.sha256(param_bytes)
                aggregated_hashes.append(hash_obj.hexdigest())

            # 组合所有客户端的ID和哈希
            message = f"{':'.join(map(str, client_ids))}:{':'.join(aggregated_hashes)}".encode()
            hash_obj = hashlib.sha256(message)

            # 使用服务器公钥验证签名
            self.server_public_key.verify(signature, hash_obj.digest(), ec.ECDSA(hashes.SHA256()))
            print(f"Client {self.id}: Signature verification successful")
            return True
        except Exception as e:
            print(f"Client {self.id}: Signature verification failed: {e}")
            return False

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if torch.isnan(output).any():
                    print(f"Client {self.id}: NaN detected in model output")
                    output = torch.nan_to_num(output, nan=0.0)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        if np.isnan(y_prob).any():
            print(f"Client {self.id}: NaN detected in y_prob, replacing with zeros")
            y_prob = np.nan_to_num(y_prob, nan=0.0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))