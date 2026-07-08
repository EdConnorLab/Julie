"""
probe_geometry.py — physical layout of the 32-channel linear probe.

The manually sorted units live on Intan channels named like ``C-021``. Whether
two units *could* be the same neuron depends on how physically close their
channels are on the probe: the same extracellular spike bleeds onto adjacent
contacts, so duplicates are expected between neighbouring contacts and are
suspicious between far-apart ones.

``PROBE_CHANNEL_ORDER`` is copied from
``spikesorting/sort_spikes/sort_spikes.py`` — it maps *contact index* (physical
position along the linear probe, 0 = one end) to the Intan *native order*
(the integer in the channel name, e.g. ``C-021`` -> 21). The contacts are
spaced ``Y_PITCH_UM`` micrometres apart.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# 32-channel linear probe: contact index -> Intan native_order.
# (Identical to PROBE_CHANNEL_ORDER in sort_spikes/sort_spikes.py.)
PROBE_CHANNEL_ORDER = np.array([
    25, 6, 21, 10, 26, 5, 20, 11, 22, 9, 27, 4, 19, 12, 28, 3,
    24, 7, 18, 13, 29, 2, 17, 14, 31, 0, 23, 8, 30, 1, 16, 15,
])

# Centre-to-centre spacing between adjacent contacts, in micrometres.
Y_PITCH_UM = 65.0

# native_order -> contact index (inverse of PROBE_CHANNEL_ORDER)
_NATIVE_TO_CONTACT = {int(native): int(idx) for idx, native in enumerate(PROBE_CHANNEL_ORDER)}


def native_order_of(channel_name: str) -> Optional[int]:
    """Return the Intan native order for a channel name like ``'C-021'``.

    Returns ``None`` if the name cannot be parsed.
    """
    try:
        return int(str(channel_name).split("-")[-1].replace("_", "-").split("-")[-1])
    except (ValueError, IndexError):
        return None


def contact_index_of(channel_name: str) -> Optional[int]:
    """Physical contact index (0 = one end of the probe) for a channel name.

    Returns ``None`` for channels not present on this probe map.
    """
    native = native_order_of(channel_name)
    if native is None:
        return None
    return _NATIVE_TO_CONTACT.get(native)


def depth_um_of(channel_name: str) -> Optional[float]:
    """Physical position of a channel along the probe, in micrometres."""
    contact = contact_index_of(channel_name)
    if contact is None:
        return None
    return contact * Y_PITCH_UM


def channel_distance_um(channel_a: str, channel_b: str) -> Optional[float]:
    """Physical distance between two channels in micrometres.

    Returns ``None`` if either channel is not on the probe map.
    """
    da = depth_um_of(channel_a)
    db = depth_um_of(channel_b)
    if da is None or db is None:
        return None
    return abs(da - db)


def contacts_apart(channel_a: str, channel_b: str) -> Optional[int]:
    """Number of contacts separating two channels (0 = same, 1 = adjacent)."""
    ca = contact_index_of(channel_a)
    cb = contact_index_of(channel_b)
    if ca is None or cb is None:
        return None
    return abs(ca - cb)
