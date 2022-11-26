import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3, device = 'cpu', optimizer = None):
        self.model = model.to(device)
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion

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

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1):
        model_device = next(self.model.parameters()).device
        train_data = Dataset(train_data, device = model_device)

        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        
        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()
    
    def eval(self, eval_data: Dict[str, np.array], batch_size: int = 1):
        model_device = next(self.model.parameters()).device
        eval_data = Dataset(eval_data, device = model_device)

        eval_dataloader = eval_data.get_dataloader(batch_size = batch_size)
        total_loss = torch.tensor(0.0)
        
        
        with torch.no_grad():
            for batch in eval_dataloader:
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
