
import sagemaker

from sagemaker.pytorch.estimator import PyTorch

role = sagemaker.get_execution_role()

pytorch_estimator = PyTorch(
    'null', 
    role=role, 
    train_instance_count=1, 
    train_instance_type='local',
    framework_version='1.1.0'
)
dir(pytorch_estimator)
