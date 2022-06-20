from setuptools import setup, find_packages
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

VERSION = '0.0.7' 
DESCRIPTION = 'a Deep Learning Framework for Multi-target Prediction'

def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name='DeepMTP', 
        version=VERSION,
        author='Dimitris Iliadis',
        author_email='<dimitrios.iliadis@ugent.be>',
        description=DESCRIPTION,
        license='MIT',
        long_description=readme(),
        packages=find_packages(),
        long_description_content_type='text/markdown',
        url='https://github.com/diliadis/DeepMTP',
        install_requires=requirements,
        # dependency_links=['https://download.pytorch.org/whl/cpu'],
        # extras_require={
        #     'cpu': ['torch==1.11.0+cpu', 'torchaudio==0.11.0+cpu', 'torchvision==0.12.0+cpu'],
        #     'gpu': ['torch==1.11.0', 'torchaudio==0.11.0', 'torchvision==0.12.0']
        # },
        classifiers= [
            'Intended Audience :: Education',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
        ]
)