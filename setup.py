from setuptools import find_namespace_packages, setup

packages = find_namespace_packages(include=["jaca"])

setup(
    name="jaca",
    version="0.1.0",
    packages=packages,
    install_requires=[
        "rerun-sdk",
        "numpy",
        "scipy",
        "pyyaml",
        "urdfpy",
    ],
    author="Haoyang Li",
    author_email="hyli1606@gmail.com",
    description="A visualization tool for robotics tasks using Rerun",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiHaoyang0616/Jaca.git",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
