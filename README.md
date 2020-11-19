# 1. Title: NTIA/ITS SCOS USRP Plugin

This repository is a plugin to add support for the Ettus B2xx line of signal analyzers
to scos-sensor. See the [scos-sensor documentation](
    https://github.com/NTIA/scos-sensor/blob/SMBWTB475_refactor_radio_interface/README.md)
for more information about scos-sensor, especially the section about
[Actions and Hardware Support](
    https://github.com/NTIA/scos-sensor/blob/SMBWTB475_refactor_radio_interface/DEVELOPING.md#actions-and-hardware-support).

This repository includes many 700MHz band actions in scos_usrp/configs/actions. Action
classes, RadioInterface, GPSInterface, and signals are used from scos_actions.

For information on adding actions, see the [scos_actions documentation](
    https://github.com/NTIA/scos-actions/blob/PublicRelease/README.md#adding-actions).

## 2. Table of Contents

- [Overview of Repo Structure](#3-overview-of-repo-structure)
- [Running in scos-sensor](#4-running-in-scos-sensor)
- [Development](#5-development)
- [License](#6-license)
- [Contact](#7-contact)

## 3. Overview of Repo Structure
- scos_usrp/configs: This folder contains the yaml files with the parameters used to
  initialize the USRP supported actions and sample calibration files.
- scos_usrp/discover: This includes the code to read yaml files and make actions
  available to scos-sensor.
- scos_usrp/hardware: This includes the USRP implementation of the radio interface and
  GPS interface. It also includes supporting calibration and test code.

## 4. Running in scos-sensor

Requires pip>=18.1 (upgrade using python3 -m pip install --upgrade pip).

Below are steps to run scos-sensor with the scos-usrp plugin:

1.	Clone scos-sensor: `git clone https://github.com/NTIA/scos-sensor.git`
1.	Navigate to scos-sensor: `cd scos-sensor`
1.	If it does not exist, create env file while in the root scos-sensor directory:
    `cp env.template ./env`
1.	Make sure the scos-usrp dependency is in requirements.txt in scos-sensor/src
    folder. If you are using a different branch than master, change master in the
    following line to the branch you are using:
    `scos_usrp @ git+https://github.com/NTIA/scos-usrp@master#egg=scos_usrp.`
1.	Make sure `BASE_IMAGE` is set to `BASE_IMAGE=smsntia/uhd_b2xx_py3` in env file.
1.	Get environment variables: `source ./env`
1.	Build and start containers: `docker-compose up -d --build --force-recreate`

## 5. Development

### Requirements and Configuration
Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`) and
python>=3.6.

It is highly recommended that you first initialize a virtual development environment
using a tool such a conda or venv. The following commands create a virtual environment
using venv and install the required dependencies for development and testing.

```python
python3 -m venv ./venv
source venv/bin/activate
python3 -m pip install --upgrade pip # upgrade to pip>=18.1
python3 -m pip install -r requirements-dev.txt
```

### Running Tests
Since the UHD drivers are required, a docker container is used for testing. [Install
Docker](https://docs.docker.com/get-docker/).

```bash
docker build -f docker/Dockerfile-test -t usrp_test .
docker run usrp_test
```

### Committing

Besides running the test suite and ensuring that all tests are passing, we also expect
all Python code that's checked in to have been run through an auto-formatter.

This project uses a Python auto-formatter called Black. Additionally, import statement
sorting is handled by isort.

There are several ways to autoformat your code before committing. First, IDE
integration with on-save hooks is very useful. Second, if you've already pip-installed
the dev requirements from the section above, you already have a utility called
pre-commit installed that will automate setting up this project's git pre-commit hooks.
Simply type the following once, and each time you make a commit, it will be
appropriately autoformatted.

```bash
pre-commit install
```

You can manually run the pre-commit hooks using the following command.

```bash
pre-commit run --all-files
```

In addition to Black and isort, various other pre-commit tools are enabled including
markdownlint. Markdownlint will show an error message if it detects any style issues in
markdown files. See .pre-commit-config.yaml for the list of pre-commit tools enabled
for this repository.

## 6. License

See [LICENSE](LICENSE.md).

## 7. Contact

For technical questions about scos-usrp, contact Justin Haze, jhaze@ntia.gov
