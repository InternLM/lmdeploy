from typing import Dict
from lmdeploy.logger import get_logger

logger = get_logger("lmdeploy")


try:
    logger.debug("Registering DLSlime Backend")
    from .dlslime import DLSlimeBackend
except ImportError as e:
    logger.debug("Disable DLSlime Backend")

try:
    logger.debug("Registering Mooncake Backend")
    from .mooncake import MooncakeBackend
except ImportError as e:
    logger.debug("Disable Mooncake Backend")


try:
    logger.debug("Registering InfiniStoreBackend Backend")
    from .infinistore import InfiniStoreBackend
except ImportError as e:
    logger.debug("Disable InfiniStoreBackend Backend")
