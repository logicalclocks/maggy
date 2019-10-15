import os
from setuptools import setup, find_packages
from maggy.version import __version__


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='maggy',
    version=__version__,
    install_requires=[
        'numpy==1.16.5'
    ],
    extras_require={
        'pydoop': ['pydoop'],
        'tf': ['tensorflow==1.14.0'],
        'docs': [
            'sphinx==1.8.5',
            'sphinx-autobuild',
            'recommonmark',
            'sphinx_rtd_theme',
            'jupyter_sphinx_theme'
        ],
        'test': [
            'pylint',
            'pytest',
        ],
        'spark': ['pyspark==2.4.3']
    },
    author='Moritz Meister',
    author_email='meister.mo@gmail.com',
    description='Efficient asynchronous optimization of expensive black-box functions on top of Apache Spark',
    license='GNU Affero General Public License v3',
    keywords='Hyperparameter, Optimization, Auto-ML, Hops, Hadoop, TensorFlow, Spark',
    url='https://github.com/logicalclocks/maggy',
    download_url='http://snurran.sics.se/hops/maggy/maggy-' + __version__ + '.tar.gz',
    packages=find_packages(),
    long_description=read('README.rst'),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3',
    ]
)
