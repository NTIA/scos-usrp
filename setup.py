import os

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

repo_root = os.path.dirname(os.path.realpath(__file__))
requirements_path = repo_root + "/requirements.txt"
install_requires = []  # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        lines = f.read().splitlines()
        for line in lines:
            install_requires.append(os.path.expandvars(line))

setuptools.setup(
    name="scos_usrp",
    version="0.0.0",
    author="The Institute for Telecommunication Sciences",
    # author_email="author@example.com",
    description="USRP support for scos-sensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NTIA/scos_usrp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    package_data={"scos_usrp": ["configs/*.example", "configs/actions/*.yml"]},
)
