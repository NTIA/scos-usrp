import logging
import sys
from os import path

from django.conf import settings
from environs import Env

env = Env()

CONFIG_DIR = path.join(path.dirname(path.abspath(__file__)), "configs")

__cmd = path.split(sys.argv[0])[-1]

RUNNING_TESTS = "test" in __cmd
MOCK_SIGAN = env.bool("MOCK_SIGAN", default=False) or RUNNING_TESTS
MOCK_SIGAN_RANDOM = env.bool("MOCK_SIGAN_RANDOM", default=False)
if RUNNING_TESTS:
    logging.basicConfig(level=logging.DEBUG)
SIGAN_MODULE = env.str("SIGAN_MODULE", default="scos_usrp.hardware.usrp_sigan")
SIGAN_CLASS = env.str("SIGAN_CLASS", default="USRPSignalAnalyzer")
USRP_CONNECTION_ARGS = env("USRP_CONNECTION_ARGS", default="num_recv_frames=650")
