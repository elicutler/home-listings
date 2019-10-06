
import sys; sys.path.insert(0, '../utils')
import logging
import sagemaker

from gen_utils import set_logger_defaults

logger = logging.getLogger(__name__)
set_logger_defaults(logger)