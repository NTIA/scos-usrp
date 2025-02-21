import logging
import sys
from pathlib import Path

from environs import Env

logger = logging.getLogger(__name__)
env = Env()
CONFIG_DIR = Path(__file__).parent.resolve() / "configs"

__cmd = Path(sys.argv[0]).name

RUNNING_TESTS = "test" in __cmd
MOCK_SIGAN = env.bool("MOCK_SIGAN", default=False) or RUNNING_TESTS
MOCK_SIGAN_RANDOM = env.bool("MOCK_SIGAN_RANDOM", default=False)
if RUNNING_TESTS:
    logging.basicConfig(level=logging.DEBUG)
SIGAN_MODULE = env.str("SIGAN_MODULE", default=None)
SIGAN_CLASS = env.str("SIGAN_CLASS", default=None)
if RUNNING_TESTS:
    SIGAN_MODULE = "scos_usrp.hardware.usrp_sigan"
    SIGAN_CLASS = "USRPSignalAnalyzer"
logger.debug(f"scos-usrp: SIGAN_MODULE:{SIGAN_MODULE}")
logger.debug(f"scos-usrp: SIGAN_CLASS:{SIGAN_CLASS}")
USRP_CONNECTION_ARGS = env("USRP_CONNECTION_ARGS", default="num_recv_frames=650")
