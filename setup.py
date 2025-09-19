import os
import re
import sys
import sysconfig
import platform
import subprocess
from pathlib import Path

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

def print_sep(msg, width=60):
    num_dash = width - len(msg) - 2
    num_left = num_dash // 2
    num_right = num_dash - num_left
    print('-'*num_left, msg, '-'*num_right)


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        self.cfg = cfg
        build_args = ['--config', cfg]

        # Assuming Makefiles
        if not sys.platform.startswith('win'):
            num_job = max(os.cpu_count()-1, 1)
            build_args += ['--', f'-j{num_job}']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        march = env.pop('PYFEFI_ARCH', 'native')
        cmake_args += [f'-DPYFEFI_ARCH={march}']

        c_compiler = env.pop('CC', None)
        if c_compiler:
            cmake_args += [f'-DCMAKE_C_COMPILER={c_compiler}']
        cxx_compiler = env.pop('CXX', None)
        if cxx_compiler:
            cmake_args += [f'-DCMAKE_CXX_COMPILER={cxx_compiler}']
        build_system = env.pop('BUILD_SYS', None)
        if build_system:
            cmake_args += [f'-G{build_system}']

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print_sep('Running CMake prepare')
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print_sep('Building extensions')
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        so_dir = build_temp
        if sys.platform == 'win32':
            so_dir_release = build_temp / 'Release'
            if so_dir_release.exists():
                so_dir = so_dir_release
        build_lib = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        for path in so_dir.rglob('*'):
            if path.is_file():
                if sys.platform == 'win32' and not path.name.endswith('.pyd'):
                    continue
                if sys.platform != 'win32' and not path.name.endswith('.so'):
                    continue
                dest_path = build_lib / path.relative_to(build_temp)
                dest_dir = dest_path.parent
                if dest_dir.name == 'Release':
                    dest_dir = dest_dir.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                self.copy_file(path, dest_dir / 'pyfefi2')
        
        
ext_modules = [
    CMakeExtension('pyfefi2'),
]

setup(
    name = 'pyfefi2',
    version = '0.1',
    description = 'Pyfefi version 2',
    author = 'Junyi Ren',
    packages = find_packages(),
    ext_modules = ext_modules,
    cmdclass = dict(build_ext=CMakeBuild),
    zip_safe = False,
)
