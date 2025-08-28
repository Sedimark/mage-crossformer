from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface
from crossformer.utils.metrics import hybrid_loss, metric
from mlflow_mage.mlflow_saver import MlflowSaver, register_model
import mlflow
from torch.optim import AdamW
import pandas as pd
import torch

class BaseManipulation(ABC):

    @abstractmethod
    def fit(
        self,
        **kwargs
    ):
        raise NotImplementedError
    
    @abstractmethod
    def inference(
        self,
        **kwargs
    ):
        raise NotImplementedError
    

class MageCrossFormer(BaseManipulation):
    def __init__(self, cfg: dict = {}):
        self.cfg = {
            "in_len": 24,
            "out_len": 24,
            "seg_len": 2,
            "window_size": 4,
            "factor": 10,
            "model_dim": 256,
            "feedforward_dim": 512,
            "head_num": 4,
            "layer_num": 6,
            "dropout": 0.2,
            "baseline": False,
            "learning_rate": 0.1,
            "batch_size": 8,
            "split": [0.7, 0.2, 0.1],
            "seed": 2024,
            "accelerator": "auto",
            "min_epochs": 1,
            "max_epochs": 2,
            "precision": 32,
            "patience": 5,
            "num_workers": 31,
        }

        if cfg != {}:
            self.cfg.update(cfg)

    def _prepare_data(self, df: pd.DataFrame)-> List:
        dm = DataInterface(df, **self.cfg)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        return [train_loader, val_loader, test_loader]
    
    def _create_signature(
        self,
        model
    ):
        input_example = torch.randn(1, self.cfg["in_len"], self.cfg["data_dim"]).to(model.device)
        model.eval()
        with torch.no_grad():
            output_example = model(input_example)
        input_example = input_example.cpu().numpy()
        output_example = output_example.cpu().numpy()
        return [input_example, output_example]  

    def train(
        self,
        df: pd.DataFrame,
    ):
        self.cfg["data_dim"] = df.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = CrossFormer(cfg=self.cfg).to(device)

        [input_example, output_example] = self._create_signature(model)
        train_loader, val_loader, test_loader = self._prepare_data(df)

        criterion = hybrid_loss
        optimizer = AdamW(model.parameters(), lr=self.cfg["learning_rate"])

        best_val_loss = float("inf")
        best_epoch = -1
        best_model_uri = None
        best_logs = {}

        

        with MlflowSaver(run_name="pytorch_training") as mlflow_saver:
            mlflow_saver.log_params(self.cfg)

            for epoch in range(self.cfg["max_epochs"]):
                with mlflow_saver.create_child_run(
                    run_name=f"epoch_{epoch}"
                ) as epoch_saver:
                    running_loss = 0.0

                    model.train()
                    for step, batch in enumerate(train_loader):
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                        train_loss = criterion(y_hat, y)
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        running_loss += train_loss.item()
                    epoch_log = {"epoch_train_loss": running_loss / len(train_loader)}

                    running_metrics = {}
                    with torch.no_grad():
                        model.eval()
                        for step, batch in enumerate(val_loader):
                            x, y = batch
                            x, y = x.to(device), y.to(device)
                            y_hat = model(x)
                            step_metrics = metric(y_hat, y)
                            
                            if running_metrics == {}:
                                running_metrics = {
                                    k: v for k, v in step_metrics.items()
                                }
                            else:
                                running_metrics = {
                                    k: running_metrics[k] + v
                                    for k, v in step_metrics.items()
                                }

                    epoch_metrics = {
                        k: v / len(val_loader) for k, v in running_metrics.items()
                    }
                    epoch_log.update(epoch_metrics)
                    epoch_saver.log_metrics(epoch_log, step=epoch)
                    epoch_saver.log_model(
                        model=model,
                        input_example=input_example,
                        output_example=output_example,
                        framework="pytorch",
                        pip_requirements=[f"torch=={torch.__version__}"],
                    )

                    try:
                        model_uri = epoch_saver.model_uri
                        if epoch_log["SCORE"] < best


