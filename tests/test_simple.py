import sys
sys.path.insert(0, '.')
import os

import numpy as np

import torch
import dc_framework
from models import AlexNet
from dc_framework.model_instruments import AutoParallelMLP


def train_simple_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    criterion = torch.nn.BCELoss()

    data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }

    model_dc_framework = dc_framework.init(model, criterion)
    model_dc_framework.to('cpu')
    model_dc_framework.train(train_data = data)
    model_dc_framework.eval(eval_data = data)
    model_dc_framework.save("tests/tmp.pt")
    model_dc_framework.load("tests/tmp.pt")
    
    model_dc_framework.train(train_data = data)
    model_dc_framework.eval(eval_data = data)
    
def test_AutoParallelMLP():
    model = AlexNet()
    input = torch.zeros((8, 3, 224, 224), requires_grad=True)
    APMLP = AutoParallelMLP(model, input)
   

if __name__ == "__main__":
    if torch.distributed.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group('nccl', rank = local_rank)
    train_simple_model()
