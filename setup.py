#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(
    name='nti.data',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['click==6.7',
                      'MarkupSafe==0.23',
                      'mysqlclient==1.3.10',
                      'SQLAlchemy==1.1.13',
                      'matplotlib==2.0.2',
                      'wxPython==4.0.0'],
    author="Austin Graham",
    author_email="austin.graham@nextthought.com",
    url="https://nextthought.com"
)