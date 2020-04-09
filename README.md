# scos_usrp
USRP support for scos-sensor

This plugin requires the UHD drivers to be installed. The easiest way to install the drivers is to use an existing
docker image as the base image. The default base image in scos-sensor has the UHD drivers. However, it requires
[authentication to GitHub packages](https://help.github.com/en/packages/using-github-packages-with-your-projects-ecosystem/configuring-docker-for-use-with-github-packages#authenticating-to-github-packages)
using a
[GitHub personal access token](https://help.github.com/en/packages/publishing-and-managing-packages/about-github-packages#about-tokens).
This is temporary until this repository is made public. The environment variable should be set to: `BASE_IMAGE=docker.pkg.github.com/ntia/scos_usrp/uhd_b2xx_py3:3.13.1.0-rc1`. This is the
default in `scos-sensor/env.template`. ***Alternatively, you can use the image hosted on Docker Hub
without authentication:*** `BASE_IMAGE=smsntia/uhd_b2xx_py3`.
