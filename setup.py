#
#   Copyright 2021 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import os
from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader


version = (
    SourceFileLoader("maggy.version", os.path.join("maggy", "version.py")).load_module().__version__
)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='maggy',
    version=version,
    install_requires=[
        'numpy==1.21.1', 'scikit-optimize==0.9.0', 'statsmodels==0.12.2', 'scipy==1.6.3'
    ],
    extras_require={
        'pydoop': ['pydoop'],
        'tf': ['tensorflow==2.4.1'],
        'torch': ['torch==1.7.1'],  # Should be 1.8.1 if we want to support PyTorch's ZeRO.
        'zero': ['deepspeed==0.3.13',
                 'fairscale==0.3.0'],
        'docs': [
            'mkdocs==1.3.0',
            'mike==1.1.2',
            'mkdocs-material==8.2.8',
            'markdown-include==0.6.0',
        ],
        'dev': [
            'black==20.8b1',
            'flake8==3.9.0',
            'pre-commit==2.11.1',
        ],
        'spark': ['pyspark==2.4.3']
    },
    author='Moritz Meister',
    author_email='moritz@logicalclocks.com',
    description='Distribution transparent Machine Learning experiments on Apache Spark ',
    license='Apache License 2.0',
    keywords='Hyperparameter, Optimization, Distributed, Training, Keras, PyTorch, TensorFlow, Spark',
    url='https://github.com/logicalclocks/maggy',
    download_url='',
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Utilities',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Developers',
    ]
)
