from typing import Annotated as A

from bluesky import RunEngine
from bluesky.run_engine import call_in_bluesky_event_loop
import ophyd_async.epics.core._aioca as _ophyd_aioca
from ophyd_async.core import (
    StandardReadable,
    StandardReadableFormat as Format,
    SignalRW,
    init_devices,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix

# RunEngine imports pyepics. In this pixi env, reusing pyepics' CA context makes
# aioca/ophyd-async signal connections time out against this caproto IOC.
_ophyd_aioca._use_pyepics_context_if_imported = lambda: None

RE = RunEngine({})


class XRTToroidMirror(StandardReadable, EpicsDevice):
    pitch: A[SignalRW[float], PvSuffix("pitch"), Format.HINTED_UNCACHED_SIGNAL]
    roll: A[SignalRW[float], PvSuffix("roll"), Format.HINTED_UNCACHED_SIGNAL]
    yaw: A[SignalRW[float], PvSuffix("yaw"), Format.HINTED_UNCACHED_SIGNAL]


with init_devices(connect=True):
    toroid_mirror = XRTToroidMirror(
        "xrt:myTestBeamline:toroidMirror01:", name="toroid_mirror"
    )
