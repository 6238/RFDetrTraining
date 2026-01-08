# setup.py
from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "rfdetr",
    "cloudml-hypertune",
]

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
)
