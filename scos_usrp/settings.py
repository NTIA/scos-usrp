import os
import sys
from os import path
from django.conf import settings

SCOS_ACTIONS_BASE_DIR = path.dirname(path.abspath(__file__))
SCOS_ACTIONS_REPO_ROOT = path.dirname(SCOS_ACTIONS_BASE_DIR)

SCOS_ACTIONS_CONFIG_DIR = path.join(SCOS_ACTIONS_REPO_ROOT, "configs")

CONFIG_DIR = path.join(path.dirname(path.abspath(__file__)), "configs")
ACTION_DEFINITIONS_DIR = path.join(CONFIG_DIR, "actions")

# APP_ROOT = os.environ["APP_ROOT"]
# if not APP_ROOT:
#     APP_ROOT = SCOS_ACTIONS_REPO_ROOT
# # Healthchecks - the existance of any of these indicates an unhealthy state
# SDR_HEALTHCHECK_FILE = path.join(APP_ROOT, "sdr_unhealthy")
# SCHEDULER_HEALTHCHECK_FILE = path.join(APP_ROOT, "scheduler_dead")



#SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, "sensor_calibration.json")
if not settings.configured or not hasattr(settings, 'SIGAN_CALIBRATION_FILE'):
    SIGAN_CALIBRATION_FILE = path.join(CONFIG_DIR, "sigan_calibration.json")
else:
    SIGAN_CALIBRATION_FILE = settings.SIGAN_CALIBRATION_FILE
if not settings.configured or not hasattr(settings, 'SENSOR_CALIBRATION_FILE'):
    SENSOR_CALIBRATION_FILE = path.join(CONFIG_DIR, "sensor_calibration.json")
else:
    SENSOR_CALIBRATION_FILE = settings.SENSOR_CALIBRATION_FILE

__cmd = path.split(sys.argv[0])[-1]
RUNNING_TESTS = "test" in __cmd

if not settings.configured or not hasattr(settings, 'MOCK_RADIO'):
    MOCK_RADIO = RUNNING_TESTS
else:
    MOCK_RADIO = settings.MOCK_RADIO
if not settings.configured or not hasattr(settings, 'MOCK_RADIO_RANDOM'):
    MOCK_RADIO_RANDOM = False
else:
    MOCK_RADIO_RANDOM = settings.MOCK_RADIO_RANDOM