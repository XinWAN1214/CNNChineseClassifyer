#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='CNNChineseClassifyer',
    version='0.1.4',
    description='CCC package for Humachine Studio',
    long_description=readme,
    author='Winnerineast',
    author_email='winnerineast@gmail.com',
    url='https://github.com/winnerineast/CNNChineseClassifyer',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
