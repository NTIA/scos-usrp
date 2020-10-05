# scos_usrp

USRP support for scos-sensor

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`)

This repository includes many 700MHz band actions in
[scos_usrp/configs/actions](scos_usrp/configs/actions). The signals and the action
classes, RadioInterface and GPSInterface, are used from
[scos-actions](https://github.com/ntia/scos_actions).

For information on adding actions, see the
[scos-actions documentation](https://github.com/NTIA/scos-actions/blob/master/DEVELOPING.md#adding-actions).

This plugin requires the UHD drivers to be installed. The easiest way to install the
drivers is to use an existing docker image as the base image. In scos-sensor in the
env.template file, ensure `BASE_IMAGE` is set to the following:
`BASE_IMAGE=smsntia/uhd_b2xx_py3`.
