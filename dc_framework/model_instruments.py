import logging
import numpy as np

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3, 
                 device = 'cpu', optimizer = None,
                 backend = 'nccl', init_method = 'localhost', auto_choose_devices = True):
        self.model = model.to(device)
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion
        self.backend = backend
        self.init_method = init_method
        self.auto_choose_devices = auto_choose_devices

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }
    
    def __choose_device(self):
        devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        count_devices = len(devices)
        if count_devices > 1:
            print('distributed learning')
            torch.distributed.init_process_group(self.backend, init_method = self.init_method,
                                                 world_size = count_devices)
            self.model = DistributedDataParallel(self.model)
            device = f"cuda:{torch.distributed.get_rank()}"
            self.is_distributed = True
        else:
            if count_devices == 1:
                print('learning on cuda')
                device = f"cuda:{torch.cuda.current_device()}"
            else:
                print('learning on cpu')
                device = 'cpu'
            self.is_distributed = False
        return device

    def __change_batch_device(self, batch, device):
        batch_device = []
        for elem in batch:
            batch_device.append(elem.to(device))
        return batch_device

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1, device = 'cpu'):
        if self.auto_choose_devices:
            device = self.__choose_device()
            self.model.to(device)
        else:
            self.is_distributed = False

        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size, is_distributed = self.is_distributed)

        self.train_loop(train_dataloader, device)
    
    def train_loop(self, train_dataloader, device):
        for batch in train_dataloader:
            batch  = self.__change_batch_device(batch, device)
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()
    
    def eval(self, eval_data: Dict[str, np.array], batch_size: int = 1):
        device = next(self.model.parameters()).device
        eval_data = Dataset(eval_data, device = device)

        eval_dataloader = eval_data.get_dataloader(batch_size = batch_size)
        total_loss = torch.tensor(0.0).to(device)
        
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self.__change_batch_device(batch, device)
                output = self.forward(*batch)
                total_loss += output["loss"]

        return total_loss / len(eval_dataloader)

    def to(self, device):
        self.model.to(device = device)

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path:Path):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

class MLP(nn.Module):
    def __init__(self, in_features, hiddens, out_features, func_activ, bias = True) -> None:
        super().__init__()
        assert isinstance(func_activ, list)
        self.fc1 = nn.Linear(in_features, hiddens, bias = bias)
        self.fc2 = nn.Linear(hiddens, out_features, bias = bias)
        self.func_act_1 = func_activ[0]
        self.func_act_2 = func_activ[1]
        self.count_devices = torch.distributed.get_world_size()
        self.bias = bias

    def forward(self, x): # (bs, in_features)
        if self.count_devices > 1:
            rank = torch.distributed.get_rank()
            device = f'cuda:{rank}'
            weights1_rank = self.fc1.weight[rank:rank+1, :].to(device)

            bias1_rank = 0
            bias2_rank = 0
            if self.bias:
                bias1_rank = self.fc1.bias[rank:rank+1, :].to(device)
                bias2_rank = self.fc2.bias[rank:rank+1, :].to(device)

            output1_rank = self.func_act_1(weights1_rank * x + bias1_rank)
            weights2_rank = self.fc2.weight[rank:rank+1, :].to(device)
            output2_rank = self.func_act_2(weights2_rank*output1_rank + bias2_rank)

            if rank == 0:
                output_list = [torch.zeros_like(output2_rank) for _ in range(self.count_devices)]
                torch.distributed.gather(output2_rank, output_list)
                output = torch.tensor([], dtype = output2_rank.dtype)
                for t in output_list:
                    output = torch.cat((output, t), dim = 0)
                return output
            else:
                torch.distributed.gather(output2_rank)
        else:
            output_1 = self.func_act_1(self.fc1(x))
            output = self.func_act_2(self.fc2(output_1))
            return output

