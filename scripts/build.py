# build_backend.py
import glob
import shutil
from os.path import abspath, dirname

from setuptools import Extension
from setuptools.build_meta import *
from setuptools.build_meta import build_wheel as setuptools_build_wheel
from setuptools.command.build_ext import build_ext as setuptools_build_ext


class CMakeExtension(Extension):
    def __init__(self, name, cmake_list):
        super().__init__(name, sources=[])
        self.cmake_list = cmake_list


class build_ext(setuptools_build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        # super().run()

    def build_cmake(self, ext):
        build_type = f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}"
        self.spawn(
            [
                "cmake",
                "-B",
                self.build_temp,
                "-S",
                dirname(ext.sources[0]),
                build_type,
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={dirname(abspath(self.get_ext_fullpath(ext.name)))}",
            ]
        )
        if not self.dry_run:
            self.spawn(["cmake", "--build", self.build_temp])
        ...


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    shutil.copy("docs/RELEASE.md", "src/xpu_graph")
    return setuptools_build_wheel(
        wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory
    )
