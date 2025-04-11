"""Wrap Script

This wrap script is to provide interface for pipelines to call inside the mage
"""

from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
import torch


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
