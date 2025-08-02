import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from mlflow_mage.mlflow_saver import MlflowSaver, register_model


class SimpleModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def main():
    np.random.seed(42)
    X = np.random.rand(1000, 10).astype(np.float32)
    y = (
        (2 * np.sum(X, axis=1) + 0.5 + np.random.randn(1000) * 0.2)
        .astype(np.float32)
        .reshape(-1, 1)
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_size = 10
    hidden_size = 20
    output_size = 1
    learning_rate = 0.01
    num_epochs = 10

    model = SimpleModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    input_example = X_train[0:1]
    with torch.no_grad():
        output_example = model(torch.tensor(input_example)).numpy()

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_uri = None

    with MlflowSaver(run_name="pytorch_training") as mlflow_saver:
        mlflow_saver.log_params(
            {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            }
        )

        for epoch in range(num_epochs):
            with mlflow_saver.create_child_run(
                run_name=f"epoch_{epoch}"
            ) as epoch_saver:
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss}
                epoch_saver.log_metrics(epoch_metrics, step=epoch)

                epoch_saver.log_model(
                    model=model,
                    input_example=input_example,
                    output_example=output_example,
                    model_name=f"pytorch-cnn-{epoch}",
                    framework="pytorch",
                    pip_requirements=[f"torch=={torch.__version__}"],
                )

                try:
                    model_uri = epoch_saver.model_uri

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_model_uri = model_uri
                        print(
                            f"New best model at epoch {epoch} with val_loss: {val_loss:.4f}"
                        )
                except Exception as e:
                    print(f"Error getting model URI: {e}")

                mlflow_saver.log_metrics(
                    {
                        f"epoch_{epoch}_train_loss": train_loss,
                        f"epoch_{epoch}_val_loss": val_loss,
                    }
                )

                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        mlflow_saver.log_params(
            {"best_epoch": best_epoch, "best_val_loss": best_val_loss}
        )

        if best_model_uri:
            try:
                model_version = register_model(
                    model_uri=best_model_uri,
                    name="pytorch_regression_model",
                    description=f"Best model from epoch {best_epoch} with validation loss {best_val_loss:.4f}",
                    tags={
                        "best_epoch": str(best_epoch),
                        "val_loss": f"{best_val_loss:.4f}",
                        "model_type": "pytorch_regression",
                    },
                )
                print(f"Successfully registered model version: {model_version}")
            except Exception as e:
                print(f"Error registering model: {e}")
        else:
            print("No best model URI found to register")


if __name__ == "__main__":
    import os

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
        "http://localhost:9000"  # The endpoint for the minio api change only the port if modified in docker-compose.yaml
    )
    os.environ["AWS_ACCESS_KEY_ID"] = (
        "admin"  # The username for the Minio instance
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = (
        "minio_sedimark"  # The password for the Minio instance
    )
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"  # DO NOT MODIFY
    os.environ["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "mlflow_sedimark"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Default"
    os.environ["MLFLOW_TRACKING_USERNAME"] = (
        "admin"  # The username of the admin account for the MLFlow instance
    )
    os.environ["MLFLOW_TRACKING_PASSWORD"] = (
        "password"  # The password of the admin account for the MLFlow instance
    )
    os.environ["MLFLOW_TRACKING_URI"] = (
        "http://localhost:5000"  # The URL for the MLFlow instance
    )
    main()
