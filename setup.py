from setuptools import find_packages
from distutils.core import setup

setup(
    name='sevae',
    version='0.0.1',
    author='Mihir Kulkarni',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='mihir.kulkarni@ntnu.no',
    description='Package for training and evaluating Semantically-enhanced VAEs for collision prediction',
    install_requires=['torch>=1.13',
                      'numpy']
)