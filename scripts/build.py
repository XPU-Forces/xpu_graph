# build_backend.py
import shutil

from setuptools.build_meta import *
from setuptools.build_meta import build_wheel as setuptools_build_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    shutil.copy("docs/RELEASE.md", "xpu_graph")
    return setuptools_build_wheel(
        wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory
    )
