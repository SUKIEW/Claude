"""
使用 FL 客户端的类处理本地训练、验证和提交计算
"""
import numpy as np
import time

import torch

import torchvision
import torchvision.transforms as transforms
from verify.LHH import *

from sklearn.utils import shuffle
from typing import Tuple, Dict, List, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from type_utils import ConfigDict, Params

import wandb


class Client:
    """
    客户端类用于保存客户的本地数据和训练/评估方法。
    它不会在本地保存模型，这就避免了在 GPU 内存中为每个客户端存储一个模型的需要。
    """

    def __init__(
        self,
        client_id: int,
        train_data: Dict[str, np.ndarray],
        eval_data: Dict[str, np.ndarray],
        config_dic: "ConfigDict",
    ) -> None:
        """
        执行初始客户端设置：
        client_id——客户端的数字标识符
        train_data——客户端本地保存的训练数据
        eval_data——客户端的评估数据
        config_dic——包含实验配置的字典
        """
        self.client_id = client_id
        self.bsize = config_dic["bsize"]
        self.augmenter: Optional[torchvision.transforms.transforms.Compose] = None
        self.verbose = False
        self.epochs = config_dic["local_rounds"]

        train_data_x = np.asarray(train_data["x"]).astype(np.float32)
        if config_dic["dataset"] == "cifar":
            train_data_x = np.reshape(train_data_x, (-1, 3, 32, 32))
            self.augmenter = transforms.Compose(
                [torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1)), transforms.RandomHorizontalFlip(0.5)]
            )
        else:
            train_data_x = np.reshape(train_data_x, (-1, 1, 28, 28))

        y_train = np.asarray(train_data["y"])
        train_data_x, y_train = shuffle(train_data_x, y_train)
        num_of_samples = len(y_train)

        if config_dic["dataset"] == "merged_femnist":
            print("merging")
            y_train = class_merging(y_train)

        self.train_data = (train_data_x[0 : int(num_of_samples * 0.8)], y_train[0 : int(num_of_samples * 0.8)])

        y_test = np.asarray(eval_data["y"])

        if config_dic["dataset"] == "merged_femnist":
            print("merging")
            y_test = class_merging(y_test)

        test_data_x = np.asarray(eval_data["x"]).astype(np.float32)

        if config_dic["dataset"] == "cifar":
            test_data_x = np.reshape(test_data_x, (-1, 3, 32, 32))
        else:
            test_data_x = np.reshape(test_data_x, (-1, 1, 28, 28))

        self.eval_data = (test_data_x, y_test)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config_dic = config_dic

    def train_loop(
        self, model: torch.nn.Module, opt: torch.optim.Optimizer
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, List[np.ndarray], List[np.ndarray]]:
        """
        通过客户数据训练模型
        model——聚合器发送的模型。
        opt——聚合器发送的优化器。如果这是一个有状态的优化器，它可能也有参数。
        返回值： 一个元组，包含
                1) 训练过的模型
                2) 优化器
                3) 包含每批准确率的列表
                4) 包含每批损失的列表
        """
        (x_train, y_train) = self.train_data
        model.train()
        num_of_batches = int(len(x_train) / self.bsize)

        
        x_train, y_train = shuffle(x_train, y_train)
        for i in range(self.epochs):
            acc = []
            loss_list = []
            for bnum in range(num_of_batches):
                x_batch = torch.from_numpy(np.copy(x_train[self.bsize * bnum : (bnum + 1) * self.bsize])).to(self.device)
                y_batch = (
                    torch.from_numpy(np.copy(y_train[self.bsize * bnum : (bnum + 1) * self.bsize]))
                    .type(torch.LongTensor)
                    .to(self.device)
                )

                if self.config_dic["dataset"] == "cifar" and self.augmenter is not None:
                    x_batch = self.augmenter(x_batch)

                # zero the parameter gradients
                opt.zero_grad()
                outputs = model(x_batch)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                opt.step()

                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                y_batch = y_batch.cpu().numpy()
                acc.append(np.mean(preds == y_batch))

                loss_list.append(loss.item())
                if bnum > 0 and bnum % 500 == 0 and self.verbose:
                    print(f"Bnum {bnum}: Loss {loss.item()} Acc {np.mean(acc)}")
        if self.verbose:
            print(f"Final client performance: Bnum {bnum}: Loss {loss.item()} Acc {np.mean(acc)}")
        return model, opt, acc, loss_list

    def eval_model(self, model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray, int]:
        (x_eval, y_eval) = self.eval_data
        model.eval()
        acc_list = []
        loss_list = []

        num_of_batches = int(len(x_eval) / self.bsize)
        if num_of_batches == 0:
            x_batch = torch.from_numpy(np.copy(x_eval[0:])).to(self.device)
            y_batch = torch.from_numpy(np.copy(y_eval[0:])).type(torch.LongTensor).to(self.device)

            outputs = model(x_batch)
            loss = self.criterion(outputs, y_batch)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

            acc = np.mean(outputs == y_batch.cpu().detach().numpy()) * 100
            acc_list.append(acc)
            loss_list.append(loss.cpu().detach())
        else:
            for bnum in range(num_of_batches):
                x_batch = torch.from_numpy(np.copy(x_eval[bnum * self.bsize : (bnum + 1) * self.bsize])).to(self.device)
                y_batch = (
                    torch.from_numpy(np.copy(y_eval[bnum * self.bsize : (bnum + 1) * self.bsize]))
                    .type(torch.LongTensor)
                    .to(self.device)
                )

                outputs = model(x_batch)
                loss = self.criterion(outputs, y_batch)
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

                acc = np.mean(outputs == y_batch.cpu().detach().numpy()) * 100
                acc_list.append(acc)
                loss_list.append(loss.cpu().detach())

        return np.mean(loss_list), np.mean(acc_list), len(x_eval)


def class_merging(labels):
    """
    Classes we merge:
        c class 38 with C class 12
        i class 44 with I class 18
        j class 45 with J class 19
        k class 46 with K class 20
        l class 47 with L class 21
        m class 48 with M class 22
        o class 50 with O class 24
        p class 51 with P class 25
        s class 54 with S class 28
        u class 56 with U class 30
        v class 57 with V class 31
        w class 58 with W class 32
        x class 59 with X class 33
        y class 60 with Y class 34
        z class 61 with Z class 35
    """
    class_labels = [38, 44, 45, 46, 47, 48, 50, 51, 54, 56, 57, 58, 59, 60, 61]

    for i in range(len(labels)):
        # 合并类标签
        if labels[i] in class_labels:
            labels[i] = labels[i] - 26
        else:
            # 调整标签，使类编号在 0 - 46 之间保持连续
            # 统计被删除的类数量
            # below the current samples
            num_of_missing_classes = np.sum(np.where(labels[i] > class_labels, 1, 0))
            labels[i] = labels[i] - num_of_missing_classes
    # print(np.amax(labels))
    # assert np.amax(labels) == 46
    return labels
