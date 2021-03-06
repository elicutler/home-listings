
import os
import sagemaker

from sagemaker.pytorch import PyTorch

from constants import S3_PREFIX

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

hypers = {}
estimator = PyTorch(
    entry_point='train.py', source_dir='.', role=role,
    train_instance_count=1, train_instance_type='local',
    framework_version='1.2', hyperparameters={**hypers}
)
try:
    estimator.fit({
        'train_dir': f's3://{bucket}/{S3_PREFIX}/train',
        'val_dir': f's3://{bucket}/{S3_PREFIX}/val'
    })
finally:
    # otherwise docker tmp garbage will fill up disk
    os.system('sudo rm -rf /tmp/tmp*') 
    