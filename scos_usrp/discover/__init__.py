import logging

from scos_actions.actions.monitor_radio import RadioMonitor
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover import init

from scos_usrp.hardware import radio, gps
from scos_usrp.settings import ACTION_DEFINITIONS_DIR

logger = logging.getLogger(__name__)

actions = {
    "monitor_usrp": RadioMonitor(radio),
    "sync_gps": SyncGps(gps),
}

logger.debug("scos_usrp: ACTION_DEFINITIONS_DIR =  " + ACTION_DEFINITIONS_DIR)
yaml_actions, yaml_test_actions = init(radio=radio, yaml_dir=ACTION_DEFINITIONS_DIR)
actions.update(yaml_actions)


def get_last_calibration_time():
    return radio.last_calibration_time
