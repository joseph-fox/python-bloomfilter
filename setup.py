#!/usr/bin/env python3
"""Setup script for pybloom_live - Python 3 Bloom Filter implementation."""
from setuptools import setup

VERSION = "4.0.0"
DESCRIPTION = "Bloom filter: A Probabilistic data structure"
LONG_DESCRIPTION = """
A fast, pure-Python Bloom filter implementation with optimized hash functions.

This bloom filter is forked from pybloom, with a tightening ratio of 0.9
consistently used throughout. Choosing a ratio around 0.8-0.9 results in better
average space usage across a wide range of growth patterns, which is why the
default mode is set to LARGE_SET_GROWTH.

This module provides two implementations:
- BloomFilter: Fixed-capacity filter for known dataset sizes
- ScalableBloomFilter: Dynamically growing filter that scales automatically

Features:
- Fast xxHash for optimal performance
- Space-efficient bit array storage
- Set operations (union, intersection)
- Binary serialization support
- Python 3.6+ only (no legacy Python 2 baggage)
"""

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Operating System :: OS Independent",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Utilities",
]

setup(
    name="pybloom_live",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/plain",
    classifiers=CLASSIFIERS,
    keywords=[
        "bloom filter",
        "probabilistic",
        "data structures",
        "set membership",
        "scalable",
        "xxhash",
        "big data",
    ],
    author="Joseph Fox",
    author_email="fox7299@gmail.com",
    url="https://github.com/joseph-fox/python-bloomfilter",
    license="MIT License",
    platforms=["any"],
    python_requires=">=3.6",
    install_requires=["bitarray>=0.3.4", "xxhash>=3.0.0"],
    packages=["pybloom_live"],
    zip_safe=True,
)
