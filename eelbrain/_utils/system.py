# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from contextlib import ContextDecorator
import os
import sys
from .app_nap import disable_sleep, endActivity

IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'


class Caffeinator(ContextDecorator):
    """Keep track of processes blocking idle sleep"""
    def __init__(self):
        self.n_processes = 0

    def __enter__(self):
        if self.n_processes == 0 and IS_OSX:
            self._activity = disable_sleep()
        self.n_processes += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_processes -= 1
        if self.n_processes == 0 and IS_OSX:
            endActivity(self._activity)


caffeine = Caffeinator()
