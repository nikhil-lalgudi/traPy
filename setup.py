import os
import re

import setuptools


here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'traPy', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="traPy",
    version=version,
    author="Nikhil Vaidyanathan",
    author_email="nikhillv@umich.edu",
    description="Most comprehensive technical analysis library for Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nikhil-lalgudi/traPy",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.5",
        "torch>=1.6.0",
        "matplotlib>=3.3.0",
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)