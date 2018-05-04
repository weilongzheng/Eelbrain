"""NSActivityOptions

https://developer.apple.com/library/content/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html

License
-------
Incorporates code from `appnope <https://github.com/minrk/appnope>`_ under
2-clause BSD, copyright (c) 2013, Min Ragan-Kelley

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# cf. https://github.com/ipython/ipython/tree/master/IPython/terminal/pt_inputhooks/osx.py
from contextlib import contextmanager
import ctypes.util

objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))

void_p = ctypes.c_void_p
ull = ctypes.c_uint64

objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p
objc.objc_msgSend.restype = void_p
objc.objc_msgSend.argtypes = [void_p, void_p]

msg = objc.objc_msgSend


def _utf8(s):
    """ensure utf8 bytes"""
    if not isinstance(s, bytes):
        s = s.encode('utf8')
    return s


def n(name):
    """create a selector name (for methods)"""
    return objc.sel_registerName(_utf8(name))


def C(classname):
    """get an ObjC Class by name"""
    return objc.objc_getClass(_utf8(classname))


NSProcessInfo = objc.objc_getClass(b'NSProcessInfo')
NSString = objc.objc_getClass(b'NSString')


# constants from Foundation
NSActivityIdleDisplaySleepDisabled = (1 << 40)
NSActivityIdleSystemSleepDisabled = (1 << 20)
NSActivitySuddenTerminationDisabled = (1 << 14)
NSActivityAutomaticTerminationDisabled = (1 << 15)
NSActivityUserInitiated = (0x00FFFFFF | NSActivityIdleSystemSleepDisabled)
NSActivityUserInitiatedAllowingIdleSystemSleep = (NSActivityUserInitiated & ~NSActivityIdleSystemSleepDisabled)
NSActivityBackground = 0x000000FF
NSActivityLatencyCritical = 0xFF00000000


def beginActivityWithOptions(options, reason=""):
    """Begin an activity using the given options and reason

    Returns
    -------
    token : ?
        An object token representing the activity.

    Notes
    -----
    Indicate completion of the activity by calling :func:`endActivity` passing the returned object as the argument.    """
    reason = msg(NSString, n(b'stringWithUTF8String:'), _utf8(reason))
    info = msg(NSProcessInfo, n(b'processInfo'))
    # the next command is where multiple activity declaration hangs
    activity = msg(info,
                   n(b'beginActivityWithOptions:reason:'),
                   ull(options),
                   void_p(reason)
                   )
    return activity


def endActivity(activity):
    """end a process activity assertion"""
    info = msg(NSProcessInfo, n('processInfo'))
    msg(info, n("endActivity:"), void_p(activity))


def disable_sleep(reason='Eelbrain potentially long-running computation'):
    """Disable macOS system sleep


    ???Freezes child process when parent process runs in nope context
    """
    return beginActivityWithOptions(NSActivityUserInitiated | NSActivityLatencyCritical, reason)


@contextmanager
def sleep_disabled(reason='Eelbrain potentially long-running computation'):
    """context manager for beginActivityWithOptions.

    Within this context, App Nap will be disabled.
    """
    # activity = beginActivityWithOptions(NSActivityUserInitiated | NSActivityLatencyCritical, reason)
    try:
        yield
    finally:
        pass
    #     endActivity(activity)
