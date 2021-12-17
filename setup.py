#!/usr/bin/env python

import io
import os

from pathlib import Path
from setuptools import find_packages, setup


# requires_runtime = ["openvino==2021.4.2", "matplotlib", "opencv-python"]
requires_dev = ["openvino_dev","librosa","wrapt"]
requires_tf = requires_dev + ["tensorflow ==2.4.*", "tensorflow-datasets ==4.2.0", "nncf[tf]"]
requires_torch = requires_dev + ["onnx", "geffnet ==0.9.8", "fastseg", "nncf[torch]", "torchmetrics", "transformers"]
requires_torch += ['torch==1.7.1; sys_platform == "darwin"', 'torchvision==0.8.2; sys_platform == "darwin"', 'torch==1.7.1+cpu; sys_platform != "darwin"', 'torchvision==0.8.2+cpu; sys_platform != "darwin"'],
# pin paddlenlp because otherwise a new dependency is pulled in with version conflicts
requires_paddle = requires_dev + ["paddlepaddle ==2.1.*", "paddlehub", "paddle2onnx", "ppgan", "onnx"]

requires_all = requires_tf + requires_torch + requires_paddle

setup(
    name="openvino_notebooks",
    version="2021.4.2",
    author="Helena Kloosterman",
    author_email="helena.kloosterman@intel.com",
    maintainer="OpenVINO Notebooks",
    maintainer_email="helena.kloosterman@intel.com",
    package_dir = {"":".", "utils":"./notebooks/utils", "tools":"tools"},
    # package_dir = {"utils":"./notebooks/utils"},
    packages=["utils", "tools"],
    # entry_points={"console_scripts": ["openvino check"=check_install.py"]},
    entry_points={
        'console_scripts': [
                'check_install = tools.check_install:check_install',
                    ],
                    },
    extras_require={
        "all": requires_all,
        "torch": requires_torch,
        "paddle": requires_paddle,
        "tf": requires_tf,
 #       "runtime": requires_runtime
    },
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ],
    platforms=["Linux"],
    url="https://github.com/openvinotoolkit/openvino_notebooks",
    license="Apache 2.0",
    description="OpenVINO Notebook tutorials",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    install_requires=["setuptools>=56.0.0", "openvino==2021.4.2", "matplotlib", "opencv-python==4.5.4.58", "jupyterlab", "yaspin", "gdown", "pytube", "ipywidgets", "jedi==0.17.2", "ipykernel==5.*"],
    keywords=["openvino", "tutorials"],
    download_url="https://github.com/openvinotoolkit/openvino_notebooks",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Education",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    zip_safe=True,
)
