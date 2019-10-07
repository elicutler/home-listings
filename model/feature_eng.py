
import sys; sys.path.insert(0, '../utils')
import logging
import sagemaker

from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)

session = sagemaker.Session()
role = sagemaker.get_execution_role()




# empty_check = []
# for obj in boto3.resource('s3').Bucket(bucket).objects.all():
#     empty_check.append(obj.key)
#     print(obj.key)

# assert len(empty_check) !=0, 'S3 bucket is empty.'
# print('Test passed!')