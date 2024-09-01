import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/SyedAmeerHamza1/MLOps_min_project.mlflow')
dagshub.init(repo_owner='SyedAmeerHamza1', repo_name='MLOps_min_project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 10)