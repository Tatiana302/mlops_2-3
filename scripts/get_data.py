import pandas as pd
import os

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/deploy/mlflow"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

url = 'https://raw.githubusercontent.com/Tatiana302/mlops_2-1_data/main/train.csv'
with mlflow.start_run():
    data = pd.read_csv(url)
    mlflow.log_artifact(local_path="/home/deploy/mlops_2-3/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()
data.to_csv('/home/deploy/mlops_2-3/datasets/data.csv', sep=',', index = False)
