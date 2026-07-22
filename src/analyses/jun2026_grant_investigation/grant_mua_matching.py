"""
Pure matching logic: deprecated grant `Cell` names  ->  threshold-MUA `NeuronID`s.

The grant xlsx (common.HIS_XLSX) names a cell only by its channel string, e.g.
``Channel.C_020`` (unsorted whole-channel MUA) or ``Channel.C_027_Unit 1`` (sorted
single unit), with Date / Round No. in sibling columns. The modern threshold-MUA cache
names every unit with a full NeuronID ``{Location}_{Date}_{Round}_{Channel}``, e.g.
``AMG_2023-09-26_2_Channel.C_020``.

We do NOT reconstruct the NeuronID (which would mean guessing Location). Instead the
caller supplies, per session, the exact NeuronID strings the threshold-MUA source emits,
and we match the grant channel against them. `Location` is therefore copied verbatim
from the cache, never guessed, and the SI channel-remap is irrelevant (it only touches
sorted units, which we exclude).

This module has NO pandas / numpy / spike-source dependency on purpose, so the matching
logic can be unit-tested standalone (see test_grant_mua_matching.py). The pandas-backed
count extraction that consumes these matches lives in spike_count_connector.py.
"""

from collections import Counter

# A grant Cell name is a sorted single unit iff it carries this marker; its absence
# marks an unsorted whole-channel MUA unit. Same rule replicate_analysis / raster_review
# use ('Unit' in the Cell name).
UNIT_MARKER = '_Unit'

# Problem kinds surfaced by match_grant_cells_to_mua_neuronids (never raised silently).
PROBLEM_KINDS = (
    'skipped_sorted',   # cell carried '_Unit' -> not MUA (defensive; callers pre-filter)
    'missing_session',  # source had no cache for this (date, round)
    'no_match',         # channel absent from that session's MUA NeuronIDs
    'ambiguous',        # channel matched >1 MUA NeuronID (should be impossible per channel)
    'no_window',        # cell has no parseable Time Window -> cannot form a count window
)


def is_mua_name(name):
    """True if `name` (a grant Cell or a NeuronID) is an unsorted whole-channel MUA unit."""
    return UNIT_MARKER not in str(name)


def mua_neuron_ids(candidate_ids):
    """Keep only the MUA (whole-channel) NeuronIDs from a session's candidate list."""
    return [nid for nid in candidate_ids if is_mua_name(nid)]


def _channel_suffix(channel):
    """The '_<channel>' tail a full NeuronID ends with for this channel.

    The leading underscore anchors on the Round<->Channel boundary of
    ``{Location}_{Date}_{Round}_{Channel}`` so a zero-padded channel cannot match a
    longer one (e.g. ``Channel.C_002`` never matches ``..._Channel.C_1002``).
    """
    return '_' + str(channel).strip()


def match_channel(channel, candidate_ids):
    """Return every NeuronID in `candidate_ids` whose channel field equals `channel`.

    Exact whole-channel match via ``NeuronID.endswith('_' + channel)``. Because MUA
    channels are pre-filtered (no '_Unit'), a MUA channel ``Channel.C_018`` will NOT
    match the sorted NeuronID ``..._Channel.C_018_Unit 1`` (different tail).
    """
    suffix = _channel_suffix(channel)
    return [nid for nid in candidate_ids if str(nid).endswith(suffix)]


def match_grant_cells_to_mua_neuronids(cells, session_ids_fn):
    """Match grant MUA cells to their threshold-MUA NeuronIDs.

    Args:
      cells: iterable of objects each exposing
               .date        str, 'YYYY-MM-DD'
               .round_no    int
               .match_value str, the grant channel, e.g. 'Channel.C_020'
               .window_ms   (lo_ms, hi_ms) tuple or None
             (unit_lists.RasterRequest satisfies this duck type verbatim).
      session_ids_fn: callable(date, round_no) -> list[str] | None
             the NeuronID strings the threshold-MUA source emits for that session
             (None or empty => the session's cache is missing). Injected so this
             function is testable without the real spike cache. Results are cached
             per (date, round_no), so the source is asked once per session.

    Returns:
      (matched, problems)
        matched:  list of {'cell', 'neuron_id', 'window_ms'} in input order
        problems: list of {'kind', 'cell', 'detail'}; kind in PROBLEM_KINDS
      Nothing is dropped silently: every cell appears in exactly one of the two lists.
    """
    matched, problems = [], []
    session_cache = {}

    for cell in cells:
        if not is_mua_name(cell.match_value):
            problems.append({'kind': 'skipped_sorted', 'cell': cell,
                             'detail': f'{cell.match_value} carries {UNIT_MARKER!r}'})
            continue

        key = (cell.date, int(cell.round_no))
        if key not in session_cache:
            ids = session_ids_fn(*key)
            session_cache[key] = None if not ids else mua_neuron_ids(ids)
        candidates = session_cache[key]

        if candidates is None:
            problems.append({'kind': 'missing_session', 'cell': cell,
                             'detail': f'no threshold-MUA cache for session {key}'})
            continue

        hits = match_channel(cell.match_value, candidates)
        if len(hits) == 0:
            problems.append({'kind': 'no_match', 'cell': cell,
                             'detail': f'{cell.match_value} not among {len(candidates)} '
                                       f'MUA channels in session {key}'})
            continue
        if len(hits) > 1:
            problems.append({'kind': 'ambiguous', 'cell': cell,
                             'detail': f'{cell.match_value} matched {hits}'})
            continue
        if cell.window_ms is None:
            problems.append({'kind': 'no_window', 'cell': cell,
                             'detail': f'matched {hits[0]} but Time Window is missing'})
            continue

        matched.append({'cell': cell, 'neuron_id': hits[0], 'window_ms': cell.window_ms})

    return matched, problems


def summarize_problems(problems):
    """One-line-per-kind counts, e.g. {'no_match': 2, 'missing_session': 1}."""
    return dict(Counter(p['kind'] for p in problems))


# CSV column order for the eyeball-able match table.
MATCH_TABLE_COLUMNS = ('grant_cell', 'date', 'round_no', 'time_window',
                       'neuron_id', 'status', 'detail')


def _intervals_overlap(a_start, a_end, b_start, b_end):
    """True if [a_start, a_end) and [b_start, b_end) share an interior point. Touching
    endpoints (e.g. (0,300) & (300,600)) do NOT count -- spike counting is half-open, so
    adjacent windows don't double-count."""
    return a_start < b_end and b_start < a_end


def overlap_keep_mask(rows):
    """Decide which windows to keep when a cell has overlapping ones.

    rows: sequence of (key, start, end). Windows are grouped by `key` (the cell identity);
    within a group, whenever two windows overlap the WIDER one is kept and the narrower is
    dropped. Disjoint windows are all kept (distinct response epochs). Deterministic order:
    widest first, then earliest start, then original position. Returns a list[bool] aligned
    to `rows` (True = keep). This is how DROP_OVERLAPPING_WINDOWS collapses grant windows.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (key, _s, _e) in enumerate(rows):
        groups[key].append(i)
    keep = [True] * len(rows)
    for idxs in groups.values():
        # widest first (so it wins), then earliest start, then original order -- deterministic
        order = sorted(idxs, key=lambda i: (-(rows[i][2] - rows[i][1]), rows[i][1], i))
        kept = []
        for i in order:
            s, e = rows[i][1], rows[i][2]
            if any(_intervals_overlap(s, e, rows[j][1], rows[j][2]) for j in kept):
                keep[i] = False
            else:
                kept.append(i)
    return keep


def match_rows(matched, problems):
    """Flatten (matched, problems) into ordered, CSV-ready row dicts -- one per grant
    cell, matched first then problems, all sorted by (date, round, channel). Columns
    follow MATCH_TABLE_COLUMNS. Pure (no pandas) so it stays unit-testable."""
    rows = []
    for m in matched:
        c = m['cell']
        rows.append({'grant_cell': c.match_value, 'date': c.date,
                     'round_no': int(c.round_no), 'time_window': c.window_ms,
                     'neuron_id': m['neuron_id'], 'status': 'matched', 'detail': ''})
    for p in problems:
        c = p['cell']
        rows.append({'grant_cell': c.match_value, 'date': c.date,
                     'round_no': int(c.round_no), 'time_window': c.window_ms,
                     'neuron_id': '', 'status': p['kind'], 'detail': p['detail']})
    rows.sort(key=lambda r: (r['date'], r['round_no'], r['grant_cell']))
    return rows
