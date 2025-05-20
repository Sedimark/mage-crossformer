"""Wrap Script

This wrap script is to provide interface for pipelines to call inside the mage
"""

from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
import torch

cfg_base = {
    # "data_dim": 8,
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
    "max_epochs": 200,
    "precision": 32,
    "patience": 5,
    "num_workers": 31,
}


def setup_fit(
    cfg: dict, df: pd.DataFrame, callbacks: list = None, **kwargs
) -> tuple:
    """
    Fit the model with the given configuration and data.

    Args:
        cfg (dict): Configuration dictionary.
        df (pd.DataFrame): DataFrame containing the data.
        callbacks (list, optional): List of callbacks to use during training. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the fitted model and the training history.
    """

    # Automatically set the data_dim based on the input DataFrame
    cfg["data_dim"] = df.shape[1]

    model = CrossFormer(cfg=cfg)
    data = DataInterface(df, **cfg)

    if callbacks is None:
        # callbacks
        model_ckpt = ModelCheckpoint(
            monitor="val_SCORE",
            mode="min",
            save_top_k=1,
            save_weights_only=False,
        )

        early_stop = EarlyStopping(
            monitor="val_SCORE",
            patience=cfg["patience"],
            mode="min",
        )
        callbacks = [model_ckpt, early_stop]

    trainer = Trainer(
        accelerator=cfg["accelerator"],
        precision=cfg["precision"],
        min_epochs=cfg["min_epochs"],
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        callbacks=callbacks,
    )

    return model, data, trainer


def inference(
    model: CrossFormer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform inference using the trained model on the provided data.

    Args:
        model (CrossFormer): The trained CrossFormer model.
        df (pd.DataFrame): DataFrame containing the data for inference.

    Returns:
        pd.DataFrame: DataFrame containing the predictions.
    """
    model.eval()
    input_tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_tensor)
    df_predictions = pd.DataFrame(predictions.squeeze(0).numpy())
    return df_predictions
