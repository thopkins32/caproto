from typing import Annotated as A

from bluesky import RunEngine
from ophyd_async.core import (
    StandardReadable,
    StandardReadableFormat as Format,
    SignalRW,
    init_devices,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix

RE = RunEngine({})


class XRTToroidMirror(StandardReadable, EpicsDevice):
    pitch: A[SignalRW[str], PvSuffix("pitch"), Format.HINTED_SIGNAL]
    roll: A[SignalRW[float], PvSuffix("roll"), Format.HINTED_SIGNAL]
    yaw: A[SignalRW[float], PvSuffix("yaw"), Format.HINTED_SIGNAL]


with init_devices():
    toroid_mirror = XRTToroidMirror(
        "ca://xrt:myTestBeamline:toroidMirror01:", name="toroid_mirror"
    )
