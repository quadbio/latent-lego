from pathlib import Path
from setuptools import setup, find_packages

author = 'Jonas Simon Fleck'
author_email = 'jonas.fleck@bsse.ethz.ch'
description = 'Latent spaces for single cells'

long_description = Path('README.md').read_text('utf-8')
requirements = [l.strip()
    for l in Path('requirements.txt').read_text('utf-8').splitlines()]

setup(
    name='latent',
    version='0.1',
    author=author,
    author_email=author_email,
    description=description,
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8.5'
)
