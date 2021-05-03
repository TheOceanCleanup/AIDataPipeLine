# AIDAtaPipeLine - A series of examples and utilities for Azure Machine Learning Services
# Copyright (C) 2020-2021 The Ocean Cleanup™
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import setuptools

setuptools.setup(
    name="toc_azurewrapper",
    version="0.0.1",
    description=
        "Wrapper around Azure code, that simplifies model training and "
        "deployment for The Ocean Cleanup.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'azureml-sdk==1.16.0',
    ],
)
