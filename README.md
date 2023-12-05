# NTIA/ITS SCOS USRP Plugin

This repository is a scos-sensor plugin to add support for the Ettus B2xx line of
signal analyzers. See the [scos-sensor README](
https://github.com/NTIA/scos-sensor/blob/master/README.md)
for more information about scos-sensor, especially the [Architecture](
https://github.com/NTIA/scos-sensor/blob/master/README.md#architecture
) and the [Actions and Hardware Support](
https://github.com/NTIA/scos-sensor/blob/master/README.md#actions-and-hardware-support
) sections which explain the scos-sensor plugin architecture.

This repository includes many 700MHz band actions in [scos_usrp/configs/actions](
scos_usrp/configs/actions). Action classes, SignalAnalyzerInterface,
GPSInterface, and signals are used from [scos_actions](https://github.com/NTIA/scos-actions).

For information on adding actions, see the [scos_actions documentation](
https://github.com/NTIA/scos-actions/blob/master/README.md#adding-actions).

## Table of Contents

- [Overview of Repo Structure](#overview-of-repo-structure)
- [Running in scos-sensor](#running-in-scos-sensor)
- [Development](#development)
- [License](#license)
- [Contact](#contact)

## Overview of Repo Structure

- scos_usrp/configs: This folder contains the yaml files with the parameters
used to initialize the USRP supported actions and sample calibration files.
- scos_usrp/discover: This includes the code to read yaml files and make actions
  available to scos-sensor.
- scos_usrp/hardware: This includes the USRP implementation of the signal analyzer
  interface and GPS interface. It also includes supporting calibration and test code.

## Running in scos-sensor

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`).

Below are steps to run scos-sensor with the scos-usrp plugin:

1. Clone scos-sensor: `git clone https://github.com/NTIA/scos-sensor.git`
1. Navigate to scos-sensor: `cd scos-sensor`
1. If it does not exist, create env file while in the root scos-sensor directory:
   `cp env.template ./env`
1. Make sure the `scos_usrp` dependency is in `requirements.txt` in `scos-sensor/src`
   folder. If you are using a different branch than master, change master in the
   following line to the branch you are using:
   `scos_usrp @ git+https://github.com/NTIA/scos-usrp@master#egg=scos_usrp.`
1. Make sure `BASE_IMAGE` is set to `BASE_IMAGE=ghcr.io/ntia/scos-usrp/scos_usrp_uhd:latest` in env file.
1. Get environment variables: `source ./env`
1. Build and start containers: `docker-compose up -d --build --force-recreate`

## Development

### Requirements and Configuration

Set up a development environment using a tool like [Conda](https://docs.conda.io/en/latest/)
or [venv](https://docs.python.org/3/library/venv.html#module-venv),
with `python>=3.8`. This repository dependends on the Python UHD library. In
Ubuntu, you can get this by installing the `python3-uhd` package. Then, you can
get access to this package in your 'venv' virtual environment using the
`--system-site-packages` option. Then, from the cloned directory, install the
development dependencies by running:

```bash
pip install .[dev]
```

This will install the project itself, along with development dependencies for pre-commit
hooks, building distributions, and running tests. Set up pre-commit, which runs
auto-formatting and code-checking automatically when you make a commit, by running:

```bash
pre-commit install
```

The pre-commit tool will auto-format Python code using [Black](https://github.com/psf/black)
and [isort](https://github.com/pycqa/isort). Other pre-commit hooks are also
enabled, and can be found in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

### Building New Releases

This project uses [Hatchling](https://github.com/pypa/hatch/tree/master/backend)
as a backend. Hatchling makes versioning and building new releases easy. The
package version can be updated easily by using any of the following commands.

```bash
hatchling version major   # 1.0.0 -> 2.0.0
hatchling version minor   # 1.0.0 -> 1.1.0
hatchling version micro   # 1.0.0 -> 1.0.1
hatchling version "X.X.X" # 1.0.0 -> X.X.X
```

To build a new release (both wheel and sdist/tarball), run:

```bash
hatchling build
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
pre-commit installed that will automate setting up this project's git pre-commit
hooks. Simply type the following once, and each time you make a commit, it will be
appropriately autoformatted.

```bash
pre-commit install
```

You can manually run the pre-commit hooks using the following command.

```bash
pre-commit run --all-files
```

In addition to Black and isort, various other pre-commit tools are enabled including
markdownlint. Markdownlint will show an error message if it detects any style
issues in markdown files. See [.pre-commit-config.yaml](.pre-commit-config.yaml)
for the list of pre-commit tools enabled for this repository.

### Updating the scos_usrp_uhd package

Run the following commands to build, tag, and push the docker image to the Github
Container Registry. Replace X.X.X with the desired version number.

```bash
docker build -f docker/Dockerfile-uhd -t scos_usrp_uhd .
docker tag scos_usrp_uhd ghcr.io/ntia/scos-usrp/scos_usrp_uhd:X.X.X
docker push ghcr.io/ntia/scos-usrp/scos_usrp_uhd:X.X.X.
```

## License

See [LICENSE](LICENSE.md).

## Contact

For technical questions about scos-usrp, contact Justin Haze, jhaze@ntia.gov
