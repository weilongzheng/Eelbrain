# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from contextlib import ContextDecorator
from distutils.version import LooseVersion
import os
import platform
from subprocess import Popen
import sys
from warnings import warn

IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'


class Caffeinator(ContextDecorator):
    """Keep track of processes blocking idle sleep"""
    def __init__(self):
        self.n_processes = 0

    def __enter__(self):
        if self.n_processes == 0 and IS_OSX:
            self._popen = Popen('caffeinate')
        self.n_processes += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_processes -= 1
        if self.n_processes == 0 and IS_OSX:
            self._popen.terminate()


caffeine = Caffeinator()
