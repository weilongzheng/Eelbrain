# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from distutils.version import LooseVersion
import os
import platform
from subprocess import Popen
import sys
from warnings import warn

IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'
if IS_OSX:
    import appnope
else:
    appnope = None


class Caffeinator(object):
    """Keep track of processes blocking idle sleep"""
    #  ~ 7.5 ms on my old MacBook Pro
    #  >>> timeit with caffeine: 1+2
    def __init__(self):
        self.n_processes = 0
        if IS_OSX:
            x_version = LooseVersion(platform.mac_ver()[0])
            if x_version >= LooseVersion('10.8'):
                self.enabled = True
                return
            else:
                warn("Since you are using an outdated version of OS X, your "
                     "computer might go to sleep before computations finish. "
                     "Update to at least OS X 10.8, or make sure automatic "
                     "sleep mode is turned off.")
        self.enabled = False

    def __enter__(self):
        if self.n_processes == 0 and self.enabled:
            self._popen = Popen('caffeinate')
            self._nope = appnope.nope_scope()
            self._nope.__enter__()
        self.n_processes += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_processes -= 1
        if self.n_processes == 0 and self.enabled:
            self._popen.terminate()
            self._nope.__exit__(exc_type, exc_val, exc_tb)


caffeine = Caffeinator()
