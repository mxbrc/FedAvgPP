import torch
import os
import numpy as np
import time
import random
import copy
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec  # 新增导入
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.proxy_signer = None
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def select_proxy_signer(self):
        """随机选择一个客户端作为代理签名者"""
        if self.enable_signing:
            self.proxy_signer = random.choice(self.clients)
            print(f"Selected proxy signer: Client {self.proxy_signer.id}")

    def re_sign_signatures(self):
        """代理签名者对客户端签名进行重签名并聚合"""
        if not self.enable_signing or not self.uploaded_signatures:
            return None

        try:
            aggregated_hashes = []
            for name, param in self.global_model.named_parameters():
                param_bytes = param.detach().cpu().numpy().tobytes()
                hash_obj = hashlib.sha256(param_bytes)
                aggregated_hashes.append(hash_obj.hexdigest())

            message = f"{':'.join(map(str, self.uploaded_ids))}:{':'.join(aggregated_hashes)}".encode()
            hash_obj = hashlib.sha256(message)
            signature = self.private_key.sign(hash_obj.digest(), ec.ECDSA(hashes.SHA256()))
            return signature
        except Exception as e:
            print(f"Proxy signer: Signature aggregation failed: {e}")
            return None

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            if self.enable_signing:
                self.select_proxy_signer()

            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()

            if i % self.eval_gap == 0:
                self.send_models()

                for client in self.selected_clients:
                    print(f"\nClient {client.id} starts training...")
                    client.train()

                self.receive_models()
                if len(self.uploaded_params) == 0:
                    print("No valid client parameters received, skipping aggregation")
                    continue

                self.aggregate_parameters()
                aggregated_signature = self.re_sign_signatures()

                # 分发聚合参数和签名
                for client in self.clients:
                    client.set_parameters(self.global_model)
                    if self.enable_signing and aggregated_signature:
                        client.verify_aggregated_parameters(
                            self.global_model.state_dict(),
                            aggregated_signature,
                            self.uploaded_ids,
                            self.uploaded_hashes
                        )

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        if self.rs_test_acc:
            self.print_(max(self.rs_test_acc), max(self.rs_test_auc), min(self.rs_train_loss))
        else:
            print("No evaluation results available.")
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Evaluate new clients-------------")
            self.evaluate()