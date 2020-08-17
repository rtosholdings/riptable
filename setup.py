from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import os
import sys
import pathlib
import datetime
import logging

package_name='riptable'
version_num = '1.0.3'

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class cmake_build_ext(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cmake_dirpath = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent
        extdir.mkdir(parents=True, exist_ok=True)

        default_build_type = os.environ.get('CMAKE_BUILD_TYPE', 'Release')
        config = 'Debug' if self.debug else default_build_type
        cmake_args = [
                f'-DCMAKE_BUILD_TYPE={config}',
                f'-DCMAKE_INSTALL_PREFIX={str(extdir.absolute())}'
        ]
        cmake_generator = os.environ.get('CUSTOM_CMAKE_GENERATOR')
        if not cmake_generator:
            cmake_generator = os.environ.get('CMAKE_GENERATOR')
        if cmake_generator:
            cmake_args.extend(['-G', cmake_generator])
            if cmake_generator == "Visual Studio 16 2019":
                # Explicitly set platform type to x64 for VS 16 https://cmake.org/pipermail/cmake/2019-April/069379.html
                cmake_args.extend(['-A', 'x64'])

        if sys.platform.startswith('win'):
            # On Windows, compile with MSVC by default (and use the 64-bit compiler) but allow the
            # toolset to be overridden if desired.
            # For example, set the toolset to 'ClangCL' to build with clang-cl when it's installed
            # as an optional feature in VS2019 or newer; or, specify 'v140', 'v141', 'v142' to use
            # different versions of MSVC (if multiple versions are installed).
            cmake_toolset = os.environ.get('CUSTOM_CMAKE_TOOLSET', default='host=x64')
            cmake_args.extend(['-T', cmake_toolset, '-DPython_FIND_REGISTRY=NEVER'])

        build_args = ['--config', config]
        if self.verbose:
            build_args.append('--verbose')
        parallel = os.environ.get('PARALLEL_BUILD')
        if self.parallel:
            build_args.extend(['--parallel', str(self.parallel)])
        elif parallel is not None:
            try:
                parallel_num = int(parallel)
                build_args.extend(['--parallel', str(parallel_num)])
            except ValueError:
                if parallel.lower() == 'true':
                    build_args.append('--parallel')
        if not self.dry_run:
            os.chdir(str(build_temp))
            self.spawn(['cmake', str(cmake_dirpath)] + cmake_args)
            self.spawn(['cmake', '--build', '.'] + build_args)
            self.spawn(['cmake', '--build', '.', '--config', config, '--target', 'install'])
            os.chdir(str(cmake_dirpath))


class dev_build_py(build_py):
    _version_str = 'DEV'
    def run(self):
        import datetime
        try:
            with open('riptable/_version.py', 'w') as f:
                f.write(f"__version__ = '{self._version_str}'")
        except PermissionError:
            logging.warning("permission denied for writing version to _version.py.")
        build_py.run(self)


cmdclass = {'build_ext': cmake_build_ext}
try:
    version = os.environ['BUILD_VERSION']
except KeyError:
    # N.B. "python setup.py install" will not remove the old version of the same package, so you
    #      can end up with multiple versions of riptable installed, but only the last installed one
    #      will be used. Here I fix it to 0.0.0 to avoid confusion and redundant installations.
    version = version_num
    time_str = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
    dev_build_py._version_str = time_str
    cmdclass['build_py'] = dev_build_py


setup(
    name = package_name,
    packages = [package_name],
    package_dir = {package_name: 'riptable'},
    version = version,
    description = 'Python Package for riptable studies framework',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    url="https://github.com/rtosholdings/riptable",
    install_requires=['numpy','riptide_cpp','ansi2html','numba'],
    include_package_data=True,
    package_data={package_name: ['tests/*','tests/test_files/*', 'benchmarks/*', 'hypothesis_tests/*','Utils/*', 'test_tooling_integration/*']},
    classifiers=[
         "Development Status :: 4 - Beta",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ],

    #ext_modules=[CMakeExtension('riptide_cpp')],
    cmdclass=cmdclass
    )
