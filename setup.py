from setuptools import setup, find_packages

setup(
    name='BinaryNet',
    packages=find_packages(include=["BinaryNet", "BinaryNet.*"]),
)