from setuptools import setup, find_packages
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

VERSION = '0.0.14' 
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
        install_requires=[
            'liac-arff==2.5.0',
            'matplotlib>=3.5.2',
            'matplotlib-inline>=0.1.2',
            'more-itertools>=8.13.0',
            'numpy>=1.21.6',
            'pandas>=1.3.5',
            'pillow>=9.1.1',
            'prettytable==3.3.0',
            'scikit-learn>=1.0.2',
            'scikit-multilearn==0.2.0',
            'scipy>=1.7.3',
            'wandb>=0.12.18',
            'wget==3.2',
        ],
        # install_requires=requirements,
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