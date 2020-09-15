# Developing in scos_usrp

This document describes development practices for this repository.

## Running Tests

First set `DOCKER_GIT_CREDENTIALS` by running
`export DOCKER_GIT_CREDENTIALS="$(cat ~/.git-credentials)"` if using git credential
manager or by running
`export DOCKER_GIT_CREDENTIALS=https://<username>:<password>@github.com` replacing
\<username\> with your GitHub username and \<password\> with your GitHub password.
Since the UHD drivers are required for testing, the tests are run inside a container
with the UHD drivers.

```base
docker build -f docker/Dockerfile-test -t usrp_test --build-arg DOCKER_GIT_CREDENTIALS .
docker run usrp_test
```

## Committing

Besides running the test suite and ensuring that all tests are passing, we also expect
all Python code that's checked in to have been run through an auto-formatter.

This project uses a Python auto-formatter called black. Additionally, import statement
sorting is handled by isort.

There are several ways to autoformat your code before committing. First, IDE
integration with on-save hooks is very useful. Second, if you've already pip-installed
the dev requirements from the section above, you already have a utility called
`pre-commit` installed that will automate setting up this project's git pre-commit
hooks. Simply type the following _once_, and each time you make a commit, it will be
appropriately autoformatted.

```bash
pre-commit install
```

You can manually run the pre-commit hooks using the following command.

```bash
pre-commit run --all-files
```

In addition to black and isort, various other pre-commit tools are enabled. See
[.pre-commit-config.yaml](.pre-commit-config.yaml) to see the list of pre-commit
tools enabled for this repository.
