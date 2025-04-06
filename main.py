"""
Lightweight Privacy-protection Verify Federated Learning
"""
import os
import datetime
import copy
import concurrent.futures
import time
from typing import TypedDict, List, Dict, Optional, Tuple, TYPE_CHECKING
import wandb
import argparse

import numpy as np
import torch
from torch import optim


from fl.data import setup_mnist_clients, setup_femnist_clients, setup_cifar_clients
from fl.aggregator import Aggregator
from fl.fl_model import setup_model
from verify.blockchain import *
from verify.LHH import *
from verify.commit import *

if TYPE_CHECKING:
    from fl.client import Client
    from type_utils import Params


class ConfigDict(TypedDict):
    """
    一个 TypedDict 类，用于定义配置字典中的类型。
    """

    dataset: str
    num_clients: int
    clients_participating_per_round: int
    fl_rounds: int
    learning_rate: float
    bsize: int
    secure: bool
    malicious_aggregator: bool


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_clients(config_dic: ConfigDict) -> List["Client"]:
    """
    获取配置字典中指定的 fl 客户端列表
    配置字典——config_dic
    返回实验使用的客户端列表
    """
    if config_dic["dataset"] in ["femnist", "merged_femnist"]:
        return setup_femnist_clients(config_dic)
    if config_dic["dataset"] == "mnist":
        return setup_mnist_clients(config_dic)
    if config_dic["dataset"] == "cifar":
        return setup_cifar_clients(config_dic)
    raise ValueError("Provided Dataset does not match")


def flatten_model(client_state_dict: dict) -> np.ndarray:
    """
    辅助函数，用于将所有模型参数扁平化为单个 numpy 数组，以便从编码角度简化承诺的计算。
    从编码角度看，使承诺的计算更简单。

    client_state_dict——模型的 pytorch 状态
    返回值：包含扁平化客户端权重的 numpy 数组
    """
    flattened_model_weights = []
    for param_tensor in client_state_dict:
        flattened_model_weights.append(client_state_dict[param_tensor].detach().cpu().numpy().flatten())
    return np.concatenate(flattened_model_weights)


def round_tensor_values(state_dict: dict, config_dic: ConfigDict) -> dict:
    """
    将张量四舍五入到小数位
    """
    rounded_dict = {}
    for param in state_dict:
        rounded_dict[param] = torch.round(state_dict[param], decimals=config_dic["encoder_base"])
    return rounded_dict


def average_summed_values(model: torch.nn.Module, clients_participating_per_round: int) -> torch.nn.Module:
    """
    只支持安全求和，所以平均算法中的除法由客户端进行，两个参数是全局模型和客户端参与数量
    返回值：权重取平均后的全局模型
    """
    divided_vals = {}
    state_dict = model.state_dict()
    for param in model.state_dict():
        divided_vals[param] = state_dict[param] / clients_participating_per_round
    model.load_state_dict(divided_vals)
    return model

#并行计算线性同态哈希
def compute_parallel_LHH(
    list_of_clients: List["Client"],
    params: List[np.ndarray],
    config_dic: dict
) -> List[concurrent.futures._base.Future]:
    
    max_workers = config_dic["clients_participating_per_round"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(
                list_of_clients[i].compute_homomorphic_hash, params[i].tobytes()
            )
            for i in range(max_workers)
        ]
    concurrent.futures.wait(results)
    return results

def get_gradient_fragments(model, begin, size) -> list:
    vec = []
    for param in model.values():
        param_array = param.view(-1).cpu().numpy()
        int_array = [int(np.round(v * 1e64)) for v in param_array]
        vec.extend(int_array)
    return vec[begin:begin+size]

def fl_loop(
    config_dic: ConfigDict,
    list_of_clients: List["Client"],
    model: torch.nn.Module,
    opt: torch.optim.Optimizer
) -> None:
    
    prepare_time = []
    client_verification_times = []
    aggregator_times = []

    init_fl_round = 0
    global_acc = []
    locals_acc = []
    locals_loss = []

    
    for round_num in range(init_fl_round, config_dic["fl_rounds"]):
        start_time_of_round = time.time()

        initial_state_dic = copy.deepcopy(model.state_dict())
        initial_opt_state_dic = copy.deepcopy(opt.state_dict())

        client_models = []
        client_opts = []
        client_flattern = []
        client_hashes = []

        local_acc = []
        local_loss = []

        client_participating_index = np.random.choice(
            a=len(list_of_clients), size=config_dic["clients_participating_per_round"], replace=False
        )


        # 本地训练
        for cnum in client_participating_index:
            model.load_state_dict(copy.deepcopy(initial_state_dic))
            opt.load_state_dict(copy.deepcopy(initial_opt_state_dic))
            model, opt, client_acc, client_loss = list_of_clients[cnum].train_loop(model=model, opt=opt)
            #client_flattern.append(flatten_model(model))
            client_models.append(copy.deepcopy(model.state_dict()))
            client_opts.append(copy.deepcopy(opt.state_dict()))

            local_acc.append(client_acc)
            local_loss.append(client_loss)
        
            

        local_acc = np.concatenate(local_acc)
        local_loss = np.concatenate(local_loss)

        print(f"————————round {round_num}——————————")
        print("clients_avg")
        print(f"loss {np.mean(local_loss)} acc {np.mean(local_acc)*100}", flush=True)
        wandb.log(data={"clients loss":np.mean(local_loss),"clients acc":np.mean(local_acc)*100},step=round_num)
        locals_acc.append(np.mean(local_acc)*100)
        locals_loss.append(np.mean(local_loss))

        # 安全聚合
        verify = False
        if config_dic["secure"]:
            while verify is False:
                print("聚合器执行安全聚合")
                time_pre = datetime.datetime.now()
                model = Aggregator.fed_sum(model, client_models)
                aggregator_times.append((datetime.datetime.now() - time_pre).total_seconds())
                        
                #验证步骤
                #注意：验证步骤的精度和梯度的精度要保持一致
                if round_num > 0:
                    # 验证参数
                    t1 = time.time()
                    d = 1000
                    hom_hash = HomHash(d)
                    hom_hash.HGen()
                    hashes = []
                    commitments = []
                    
                    commit = CommitmentScheme()
                    client_num = config_dic["clients_participating_per_round"]
                    alphas = [1] * client_num
                    # 这里有精度转换的问题，需要注意
                    #check_vec = [0]*d
                    for i in range(client_num):
                        vec = get_gradient_fragments(client_models[i],0,d)
                        # for j in range(d):
                        #     check_vec[j] += vec[j]
                        hash_point = hom_hash.Hash(vec)
                        hashes.append(hash_point)
                        print(len(hash_point))
                        commitments.append(commit.generate_commitment(hash_point))

                    global_vec = get_gradient_fragments(model.state_dict(), 0, d)
                    start_time = time.time()
                    agg_hash = hom_hash.Eval(hashes,alphas)
                    global_hash = hom_hash.Hash(global_vec)
                    end_time = time.time()
                    if agg_hash == global_hash:
                        verify = True
                        print("Federated Learning Verification succeeded: Aggregated hash matches the global gradient hash!")
                    else:
                        print("Federated Learning Verification failed.")
                        verify = False
                    print("verify time",end_time-start_time)
                    print("total time",time.time()-t1)

                    is_valid = commit.verify_commitment(commitments, hashes)
                    if is_valid:
                        print("All commitments are valid!")
                    else:
                        print("Some commitments are invalid!")
                break

        model = average_summed_values(
            model=model, clients_participating_per_round=config_dic["clients_participating_per_round"]
        )

        if round_num > 0:
            running_test_loss, running_test_acc = compute_test_statistics(list_of_clients, model)
            global_acc.append(running_test_acc)
            print("server:")
            print(f"loss: {running_test_loss}, acc: {running_test_acc}", flush=True)
            wandb.log(data={"server loss":running_test_loss,"server acc":running_test_acc},step=round_num)


def compute_test_statistics(list_of_clients: List["Client"], model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    running_test_loss = []
    running_test_acc = []
    weighting = []
    for client in list_of_clients:
        test_loss, test_acc, num_samples = client.eval_model(model=model)
        running_test_loss.append(test_loss)
        running_test_acc.append(test_acc)
        weighting.append(num_samples)
    return np.average(running_test_loss, weights=weighting), np.average(running_test_acc, weights=weighting)


def init(
    config_dic: ConfigDict
) -> Tuple[
    List["Client"],
    torch.nn.Module,  
    torch.optim.Optimizer,
]:
    # time_pre = datetime.datetime.now()
    list_of_clients = get_clients(config_dic)
    model = setup_model(dataset=config_dic["dataset"])

    print("Total model parameters ", sum(p.numel() for p in model.parameters()))
    if config_dic['dataset'] != 'mnist':
        opt = optim.SGD(model.parameters(), lr=config_dic["learning_rate"])
    else:
        opt = optim.Adam(model.parameters(), lr=config_dic["learning_rate"])  # Adam optimizer with learning rate 0.001

    return list_of_clients, model, opt


def main(config_dic: ConfigDict) -> None:
    list_of_clients,model,opt = init(config_dic)
    fl_loop(config_dic, list_of_clients, model, opt)


if __name__ == "__main__":

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Configuration")

    # 添加命令行参数
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist, cifar, merged_femnist, femnist")
    parser.add_argument("--num_clients", type=int, default=20, help="Number of clients")
    parser.add_argument("--clients_participating_per_round", type=int, default=5, help="Clients participating per round")
    parser.add_argument("--local_rounds",type=int, default=1, help="本地训练轮次")
    parser.add_argument("--fl_rounds", type=int, default=50, help="全局训练轮次: MNIST-50, merged mnist-3000, CIFAR-3500")
    parser.add_argument("--optimizer", type=str, default='SGD', help="Learning rate for SGD optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for SGD optimizer")
    parser.add_argument("--bsize", type=int, default=32, help="Batch size for training:merged mnist-10, other datasets-32")
    parser.add_argument("--secure", type=bool, default=True, help="Enable secure aggregation")
    parser.add_argument("--malicious_aggregator", type=bool, default=False, help="If the aggregator cheats")

    # 解析命令行参数
    args = parser.parse_args()

    configuration: ConfigDict = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "clients_participating_per_round": args.clients_participating_per_round,
        "local_rounds":args.local_rounds,
        "fl_rounds": args.fl_rounds,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "bsize": args.bsize,
        "secure": args.secure,
        "malicious_aggregator": args.malicious_aggregator
    }
    assert configuration["num_clients"] >= configuration["clients_participating_per_round"]
    
    #初始化新的wandb实验，开记录实验的信息和结果
    wandb.init(
        project='LVFL',
        name = 'LVFL-CIFAR',
        mode = 'online',
        config = configuration
    )
    main(configuration)
    #结束实验记录
    wandb.finish()