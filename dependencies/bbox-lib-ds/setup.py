""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

exec(open("bbox/version.py").read())
setup(
    name="bbox-utility",
    version=__version__,
    description="BBox Utility for Object Detection & Instance Segmentation. https://github.com/awsaf49/bbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awsaf49/bbox",
    author="Awsaf",
    author_email="awsaf49@gmail.com",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="bbox utility for object detection & instance segmentation",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=["numba", "joblib", "numpy", "pandas", "tqdm", "loguru"],
    python_requires=">=3.6",
    license="MIT",
)
