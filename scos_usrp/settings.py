import sys
from os import path

from django.conf import settings
from environs import Env

env = Env()

CONFIG_DIR = path.join(path.dirname(path.abspath(__file__)), "configs")
ACTION_DEFINITIONS_DIR = path.join(CONFIG_DIR, "actions")


# SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, "sensor_calibration.json.example")
if not settings.configured or not hasattr(settings, "SIGAN_CALIBRATION_FILE"):
    SIGAN_CALIBRATION_FILE = path.join(CONFIG_DIR, "sigan_calibration.json.example")
else:
    SIGAN_CALIBRATION_FILE = settings.SIGAN_CALIBRATION_FILE
if not settings.configured or not hasattr(settings, "SENSOR_CALIBRATION_FILE"):
    SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, "sensor_calibration.json.example")
else:
    SENSOR_CALIBRATION_FILE = settings.SENSOR_CALIBRATION_FILE

__cmd = path.split(sys.argv[0])[-1]
RUNNING_TESTS = "test" in __cmd

if not settings.configured or not hasattr(settings, "MOCK_RADIO"):
    MOCK_RADIO = env.bool("MOCK_RADIO", default=False) or RUNNING_TESTS
else:
    MOCK_RADIO = settings.MOCK_RADIO
if not settings.configured or not hasattr(settings, "MOCK_RADIO_RANDOM"):
    MOCK_RADIO_RANDOM = env.bool("MOCK_RADIO_RANDOM", default=False)
else:
    MOCK_RADIO_RANDOM = settings.MOCK_RADIO_RANDOM

USRP_CONNECTION_ARGS = env("USRP_CONNECTION_ARGS", default="num_recv_frames=512")
