import json
import time

import numpy as np
from scos_actions.signals import measurement_action_completed

from scos_usrp.discover import actions

iq_action = actions["acquire_iq_700MHz_ATT_DL"]
schedule_entry_json = {
    "name": "test_iq_single_1",
    "start": "2020-04-10T18:47:00.000Z",
    "stop": "2020-04-10T18:47:02.000Z",
    "interval": 1,
    "priority": 10,
    "id": "test_iq_single_1",
}
sensor = {
    "id": "",
    "sensor_spec": {"id": "", "model": "greyhound"},
    "antenna": {"antenna_spec": {"id": "", "model": "L-com HG3512UP-NF"}},
    "signal_analyzer": {"sigan_spec": {"id": "", "model": "USRP B210"}},
    "computer_spec": {"id": "", "model": "Dell Latitude E5570"},
}

_data = None
_metadata = None
_task_id = 0


def callback(sender, **kwargs):
    global _data
    global _metadata
    global _task_id
    _task_id = kwargs["task_id"]
    _data = kwargs["data"]
    _metadata = kwargs["metadata"]


measurement_action_completed.connect(callback)
iq_action(schedule_entry_json, 1)
number_of_zeros = len([x for x in _data if x == 0.0])
print(f"number_of_zeros = {number_of_zeros}")
print("metadata:")
print(json.dumps(_metadata, indent=4))
