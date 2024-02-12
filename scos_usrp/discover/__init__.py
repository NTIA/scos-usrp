import logging

from scos_actions.actions.monitor_sigan import MonitorSignalAnalyzer
from scos_actions.actions.sync_gps import SyncGps
from scos_actions.discover import init

from scos_usrp.hardware import gps
from scos_usrp.settings import CONFIG_DIR, SIGAN_MODULE, SIGAN_CLASS

logger = logging.getLogger(__name__)

actions = {
    "monitor_usrp": MonitorSignalAnalyzer(
        parameters={"name": "monitor_usrp"}
    ),
    "sync_gps": SyncGps(gps, {"name": "sync_gps"}),
}
test_actions = {}

ACTION_DEFINITIONS_DIR = CONFIG_DIR / "actions"
logger.debug("scos_usrp: ACTION_DEFINITIONS_DIR =  " + ACTION_DEFINITIONS_DIR)
yaml_actions, yaml_test_actions = init(
    yaml_dir=ACTION_DEFINITIONS_DIR
)
actions.update(yaml_actions)
logger.debug(f"scos-usrp: SIGAN_MODULE = {SIGAN_MODULE}")
logger.debug(f"scos-usrp: SIGAN_CLASS = {SIGAN_CLASS}")
if SIGAN_MODULE == "scos_usrp.hardware.usrp_sigan" and SIGAN_CLASS == "USRPSignalAnalyzer":
    logger.debug("scos-usrp: loading test action configs")
    TEST_ACTION_DEFINITIONS_DIR = CONFIG_DIR / "test"
    logger.debug(f"scos-usrp: TEST_ACTION_DEFINITIONS_DIR = {TEST_ACTION_DEFINITIONS_DIR}")
    _, yaml_test_actions = init(yaml_dir=TEST_ACTION_DEFINITIONS_DIR)
    logger.debug(f"scos-usrp: Found {len(yaml_test_actions)} test action configs")
    test_actions.update(yaml_test_actions)
logger.debug(f"scos-usrp: len(test_actions) = {len(test_actions)}")