import logging

from scos_actions.actions.monitor_sigan import MonitorSignalAnalyzer
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover import init

from scos_usrp.hardware import gps, sigan
from scos_usrp.settings import ACTION_DEFINITIONS_DIR

logger = logging.getLogger(__name__)

actions = {
    "monitor_usrp": MonitorSignalAnalyzer(
        parameters={"name": "monitor_usrp"}, sigan=sigan
    ),
    "sync_gps": SyncGps(gps, {"name": "sync_gps"}, sigan),
}

logger.debug("scos_usrp: ACTION_DEFINITIONS_DIR =  " + ACTION_DEFINITIONS_DIR)
yaml_actions, yaml_test_actions = init(
    sigan=sigan, gps=gps, yaml_dir=ACTION_DEFINITIONS_DIR
)
actions.update(yaml_actions)


def get_last_calibration_time():
    return sigan.last_calibration_time
