import logging
from rltools.environments import IEnvironment
from rltools.agents import IActor
from rltools.utils import LoggingConfig

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


def evaluate(env: IEnvironment, actor: IActor):
    logger.info('PLEASE IMPLEMENT ME')
    return True