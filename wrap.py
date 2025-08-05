"""Wrap Script

This wrap script is to provide interface for pipelines to call inside the mage
"""

# from typing import Optional
from crossformer.model.crossformer import CrossFormer
from crossformer.data_tools.data_interface import DataInterface
from crossformer.utils.metrics import hybrid_loss, metric
from mlflow_mage.mlflow_saver import MlflowSaver, register_model
import mlflow
from torch.optim import AdamW
import pandas as pd
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
    "max_epochs": 2,
    "precision": 32,
    "patience": 5,
    "num_workers": 31,
}


def unpack_batch(batch, device):
    (x, scale, y) = batch
    x, scale, y = x.to(device), scale.to(device), y.to(device)
    return x, scale, y


def forward_step(model, x, scale):
    y_hat = (
        model(x) * scale.unsqueeze(1) if scale._is_zerotensor() else model(x)
    )
    return y_hat


def dict_update(old_dict, new_dict):
    merged_dict = old_dict.copy()
    for key, value in new_dict.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict



def evaluate_model(model, loader, device, prefix="val"):
    model.eval()
    all_metrics = {}
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x, scale, y = unpack_batch(batch, device)
            y_hat = forward_step(model, x, scale)
            batch_metrics = metric(y_hat, y)
            all_metrics = dict_update(all_metrics, batch_metrics)
    avg_metrics = {f"{prefix}_{k}": v / len(loader) for k, v in all_metrics.items()}
    return avg_metrics


def train(cfg: dict, df: pd.DataFrame, **kwargs) -> None:
    """
    Fit the model with the given configuration and data.

    Args:
        cfg (dict): Configuration dictionary.
        df (pd.DataFrame): DataFrame containing the data.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the fitted model and the training history.
    """

    # update cfg with base settings

    # Automatically set the data_dim based on the input DataFrame
    cfg["data_dim"] = df.shape[1]

    # Setup model and data
    model = CrossFormer(cfg=cfg)
    data = DataInterface(df, **cfg)

    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = hybrid_loss
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    model.to(device)

    # input & output sample
    model.eval()
    input_example = torch.randn(1, cfg["in_len"], cfg["data_dim"])
    with torch.no_grad():
        output_example = model(input_example)
    input_example = input_example.cpu().numpy()
    output_example = output_example.cpu().numpy()

    # best performance tracking
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_uri = None
    best_logs = {}

    with MlflowSaver(run_name="pytorch_training") as mlflow_saver:
        mlflow_saver.log_params(cfg)

        for epoch in range(cfg["max_epochs"]):
            with mlflow_saver.create_child_run(
                run_name=f"epoch_{epoch}"
            ) as epoch_saver:
                running_loss = 0.0

                # === Training step ===
                model.train()
                for step, batch in enumerate(train_loader):
                    x, scale, y = unpack_batch(batch, device)
                    y_hat = forward_step(model, x, scale)
                    train_loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    running_loss += train_loss.item()

                # Epoch training logging
                epoch_loss = {"train_loss": running_loss / len(train_loader)}

                # === Validation step ===
                epoch_metrics = evaluate_model(model, val_loader, device, prefix="val")

                # === logging (epoch level only) ===
                epoch_metrics = dict_update(epoch_metrics, epoch_loss)
                epoch_saver.log_metrics(epoch_metrics, step=epoch)

                epoch_saver.log_model(
                    model=model,
                    input_example=input_example,
                    output_example=output_example,
                    model_name=f"crossformer_{epoch}",
                    framework="pytorch",
                    pip_requirements=[f"torch=={torch.__version__}"],
                )

                try:
                    model_uri = epoch_saver.model_uri

                    if epoch_metrics["SCORE"] < best_val_loss:
                        best_val_loss = epoch_metrics["SCORE"]
                        best_epoch = epoch
                        best_model_uri = model_uri
                        print(
                            f"New best model at epoch {epoch} with val_loss: {best_val_loss:.4f}"
                        )
                        best_logs = {
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "best_model_uri": best_model_uri,
                        }
                        best_logs.update(
                            {f"best_{k}": v for k, v in epoch_metrics.items()}
                        )
                except Exception as e:
                    print(f"Error getting model URI: {e}")

        mlflow_saver.log_params(best_logs)

        if best_model_uri:
            try:

                model_version = register_model(
                    model_uri=best_model_uri,
                    name="pytorch_crossformer",
                    description=f"Best model from epoch {best_epoch} with validation loss {best_val_loss:.4f}",
                    tags={**best_logs, "model_type": "pytorch_crossformer"},
                )

                print(f"Successfully registered model version: {model_version}")

            except Exception as e:
                print(f"Error registering model: {e}")

            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.transition_model_version_stage(
                name="pytorch_crossformer",
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )

            model = mlflow.pytorch.load_model("models:/pytorch_crossformer/Production")
            test_metrics = evaluate_model(model, test_loader, device, prefix="test")
            mlflow_saver.log_metrics(test_metrics)
            print(f"🧪 Test metrics: {test_metrics}")
        else:
            print("No best model URI found to register")

    return "pytorch_crossformer"


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
