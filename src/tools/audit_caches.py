"""
audit_caches.py -- read-only integrity audit of every spike cache under the data tree.

WHY THIS EXISTS
---------------
The caches were regenerated repeatedly, on two machines, while sessions were being
manually re-sorted and SpikeInterface sorting was re-run. That makes it easy to end
up with a set of caches that are individually loadable but mutually inconsistent:
a filtered cache built from a previous sort, a pre-stimulus variant that predates a
re-sort, an exploded cache silently merged from two disagreeing ``compiled.pkl``
files. This walks every cache and reports what does not line up.

IT NEVER WRITES INTO A CACHE. Every cache file is opened read-only. The only thing
it can write is the report, and only when you pass --report / --json.

WHAT IT CHECKS
--------------
per file      loads at all; required columns; trial/neuron/channel counts; task ids
              carrying more than one epoch; duplicate (trial, channel) rows; epoch
              durations far from the session median
metadata      monkey names that differ only by case (the '114J' vs '114j' drift);
              MonkeyId dtype drift; null MonkeyName/MonkeyGroup; Location='Unknown'
              and the 'Unknown_...' NeuronIDs it produces
windows       a strict cache must have NO spikes before stimulus onset; a pre{N}ms
              cache must actually have them, and roughly N ms of them
leakage       two sessions sharing task ids -- how 2023-10-27 round 4 ended up
              holding round 3's trials
exploded      per trial, do the manually-sorted units and the unsorted channels
              cover the same trials? (they must; if not, the two compiled.pkl
              sources disagree and the merge silently became a concatenation)
freshness     each derived cache against its parent: session coverage, NeuronID
              containment, file mtimes, and spike-count direction for pre-caches
summaries     sorted_spike_summary's 'Units in agreement: N' against the unit count
              actually in sorted_spike_cache -- the sharpest staleness detector
              after a re-sort
analysis      analysis_cache significance pkls: do their NeuronIDs still resolve
              against the cache they were derived from?

HOW TO RUN
----------
    cd src
    python -m tools.audit_caches                       # audit everything
    python -m tools.audit_caches --cache sorted_spike_cache exploded_spike_cache
    python -m tools.audit_caches --report ~/cache_audit.txt --json ~/cache_audit.json

Or just open it in PyCharm and hit Run -- the defaults below need no arguments.
Data root comes from JULIE_DATA_PATH (via project_util), override with --data-root.

This deliberately does NOT import the analysis pipeline (no cache_utils, no
RecordingMetadataReader, no spikeinterface). It reads the metadata workbook with
pandas directly, so a broken pipeline import can never mask a cache problem.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults. project_util is optional -- the audit must run even from a checkout
# where the pipeline does not import cleanly.
# ---------------------------------------------------------------------------
try:
    from project_util import DATA_BASE_PATH, SUBJECT_MONKEY
except Exception:                                          # pragma: no cover
    DATA_BASE_PATH = os.environ.get("JULIE_DATA_PATH",
                                    "/home/connorlab/Documents/JulieData")
    SUBJECT_MONKEY = "Cortana"

DEFAULT_REPO = os.environ.get("JULIE_PROJECT_PATH",
                              "/home/connorlab/Documents/GitHub/Julie")
DEFAULT_INTAN = os.environ.get("INTAN_BASE_PATH",
                               "/home/connorlab/Documents/IntanData")

REQUIRED_COLS = ["TaskField", "MonkeyName", "MonkeyGroup", "Channel",
                 "SpikeTimes", "EpochStartStop", "NeuronID"]

SESSION_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_round_(\d+)")
PRE_MS_RE = re.compile(r"_pre(\d+)ms$")
UNITS_RE = re.compile(r"nits in agreement:\s*(\d+)")
NO_UNITS_RE = re.compile(r"no units in agreement", re.I)

EPS = 1e-6          # seconds; float slop when comparing spike times to epochs


# ---------------------------------------------------------------------------
# Unpickling compatibility
# ---------------------------------------------------------------------------
def install_pickle_shims() -> None:
    """Let old cache pickles resolve their Channel enum.

    Caches written under different clat vintages reference the enum as either
    ``clat.intan.channels.Channel`` or the older top-level ``intan.channels``.
    Alias whichever is missing onto whichever is importable so a stale pickle
    reports its real problems instead of dying on ModuleNotFoundError.
    """
    channel_cls = None
    for mod in ("clat.intan.channels", "intan.channels"):
        try:
            channel_cls = __import__(mod, fromlist=["Channel"]).Channel
            break
        except Exception:
            continue
    if channel_cls is None:
        return
    for name in ("intan", "intan.channels", "clat", "clat.intan",
                 "clat.intan.channels"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["intan.channels"].Channel = channel_cls
    sys.modules["clat.intan.channels"].Channel = channel_cls


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


@dataclass
class Finding:
    severity: str          # ERROR | WARN | INFO
    cache: str
    code: str              # short stable slug, greppable
    message: str
    session: str = ""

    def as_dict(self):
        return {"severity": self.severity, "cache": self.cache,
                "session": self.session, "code": self.code,
                "message": self.message}


class Report:
    def __init__(self):
        self.findings: list[Finding] = []
        self.lines: list[str] = []

    def add(self, severity, cache, code, message, session=""):
        self.findings.append(Finding(severity, cache, code, message, session))

    def say(self, line=""):
        self.lines.append(line)
        print(line)

    def counts(self):
        c = Counter(f.severity for f in self.findings)
        return c["ERROR"], c["WARN"], c["INFO"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def session_of(path: Path):
    """'2023-09-26_round_1[...].pkl' -> ('2023-09-26', 1), else None."""
    m = SESSION_RE.match(path.stem)
    return (m.group(1), int(m.group(2))) if m else None


def session_label(date, round_no):
    return f"{date}_round_{round_no}"


def summary_name(date, round_no):
    return f"{date.replace('-', '')[2:]}_round{round_no}_sorting_summary.txt"


def channel_str(ch) -> str:
    return str(ch)


def is_unit_channel(ch) -> bool:
    return "_Unit" in str(ch)


def base_channel(ch) -> str:
    """Any channel spelling -> 'C_020'. Handles the enum ('C-020'), the sorted-unit
    string ('Channel.C_020_Unit 1') and the bare name."""
    s = str(ch).split("Channel.")[-1].split("_Unit")[0].replace("-", "_").strip()
    if s.startswith("C_") and s[2:].isdigit():
        return f"C_{int(s[2:]):03d}"
    return s


def spike_array(x) -> np.ndarray:
    if x is None:
        return np.empty(0)
    a = np.asarray(x, dtype=float).ravel()
    return a


def first_last(x):
    """(min, max) of a spike list, or (nan, nan) when empty. Does not assume sorted."""
    a = spike_array(x)
    if a.size == 0:
        return np.nan, np.nan
    return float(a.min()), float(a.max())


def fmt_sessions(items, limit=8):
    items = list(items)
    head = ", ".join(str(i) for i in items[:limit])
    return head + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def dir_mtime_range(paths):
    ts = [p.stat().st_mtime for p in paths]
    if not ts:
        return None, None
    return min(ts), max(ts)


def human_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------
@dataclass
class CacheSpec:
    name: str
    kind: str                  # sorted | exploded | mua | unknown
    pre_ms: int = 0
    parent: str | None = None
    path: Path = field(default=None)
    files: list = field(default_factory=list)


DECLARED = [
    ("sorted_spike_cache", "sorted", None),
    ("sorted_spike_cache_filtered", "sorted", "sorted_spike_cache"),
    ("exploded_spike_cache", "exploded", None),
    ("threshold_mua_spike_cache", "mua", None),
]


def discover_caches(data_root: Path, only=None) -> list[CacheSpec]:
    """Every *_spike_cache* directory under the data root, declared or not.

    Auto-discovery matters: a variant nobody declared (sorted_spike_cache_pre200ms,
    exploded_spike_cache_gitrecovered) is exactly the kind of thing that gets left
    behind and later mistaken for the real cache.
    """
    specs: dict[str, CacheSpec] = {}

    def kind_of(name):
        if name.startswith("sorted_spike_cache"):
            return "sorted"
        if name.startswith("exploded_spike_cache"):
            return "exploded"
        if name.startswith("threshold_mua_spike_cache"):
            return "mua"
        return "unknown"

    for child in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = child.name
        if "spike_cache" not in name:
            continue
        m = PRE_MS_RE.search(name)
        pre_ms = int(m.group(1)) if m else 0
        parent = name[: m.start()] if m else None
        specs[name] = CacheSpec(name=name, kind=kind_of(name), pre_ms=pre_ms,
                                parent=parent, path=child)

    for name, kind, parent in DECLARED:
        if name in specs:
            if parent and not specs[name].parent:
                specs[name].parent = parent
        else:
            specs[name] = CacheSpec(name=name, kind=kind, parent=parent,
                                    path=data_root / name)

    for spec in specs.values():
        spec.files = sorted(spec.path.glob("*.pkl")) if spec.path.exists() else []

    out = [specs[k] for k in sorted(specs)]
    if only:
        keep = set(only)
        out = [s for s in out if s.name in keep]
    return out


# ---------------------------------------------------------------------------
# Expected session scope, read straight from the metadata workbook
# ---------------------------------------------------------------------------
def load_scope(repo: Path, rep: Report):
    """(all_rounds, analysis_rounds, location_by_round) from the workbook.

    all_rounds      -- 'Channels' sheet: every recorded round
    analysis_rounds -- 'InitialRegression' sheet: the rounds the exploded cache and
                       the downstream analyses are scoped to
    """
    xlsx = repo / "recording_metadata" / f"{SUBJECT_MONKEY}_Recording_Metadata.xlsx"
    if not xlsx.exists():
        rep.add("WARN", "-", "workbook-missing",
                f"metadata workbook not found at {xlsx}; scope checks skipped")
        return None, None, {}

    xl = pd.ExcelFile(xlsx)

    def rounds(sheet):
        df = xl.parse(sheet)
        df["_d"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        out = set()
        for _, r in df.iterrows():
            try:
                out.add((r["_d"], int(r["Round No."])))
            except (TypeError, ValueError):
                continue          # sheet has a few '2+' style round labels
        return out, df

    all_rounds, _ = rounds("Channels")
    analysis_rounds, ir = rounds("InitialRegression")
    loc = {}
    for _, r in ir.iterrows():
        try:
            loc[(r["_d"], int(r["Round No."]))] = str(r["Location"])
        except (TypeError, ValueError):
            continue
    return all_rounds, analysis_rounds, loc


def load_sorted_scope(repo: Path):
    """The (date, round) list the sorting batch script actually runs."""
    sh = repo / "src" / "spikesorting" / "sort_spikes" / "run_analyze_in_batches.sh"
    if not sh.exists():
        return None
    m = re.search(r"COMBOS=\((.*?)\n\)", sh.read_text(), re.S)
    if not m:
        return None
    out = set()
    for line in m.group(1).strip().splitlines():
        parts = line.strip().strip('"').split()
        if len(parts) == 2:
            out.add((parts[0], int(parts[1])))
    return out


# ---------------------------------------------------------------------------
# Per-file checks
# ---------------------------------------------------------------------------
@dataclass
class FileStats:
    label: str
    rows: int = 0
    trials: int = 0
    neurons: int = 0
    channels: int = 0
    unit_channels: int = 0
    spikes: int = 0
    task_ids: set = field(default_factory=set)
    neuron_ids: set = field(default_factory=set)
    monkey_names: set = field(default_factory=set)
    observed_pre_s: float = 0.0
    mtime: float = 0.0


def check_file(df: pd.DataFrame, spec: CacheSpec, label: str, rep: Report) -> FileStats:
    st = FileStats(label=label)
    cache = spec.name

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        rep.add("ERROR", cache, "missing-columns",
                f"missing column(s): {', '.join(missing)}", label)
        return st
    if df.empty:
        rep.add("ERROR", cache, "empty-cache", "cache file has zero rows", label)
        return st

    st.rows = len(df)
    st.trials = df["TaskField"].nunique()
    st.neurons = df["NeuronID"].nunique()
    chans = df["Channel"].map(channel_str)
    st.channels = chans.nunique()
    st.unit_channels = chans[chans.str.contains("_Unit")].nunique()
    st.task_ids = set(df["TaskField"].tolist())
    st.neuron_ids = set(df["NeuronID"].astype(str))
    st.monkey_names = set(df["MonkeyName"].dropna().astype(str))

    # -- trial table -------------------------------------------------------
    pairs = df[["TaskField", "EpochStartStop"]].drop_duplicates()
    multi = [k for k, v in Counter(pairs["TaskField"]).items() if v > 1]
    if multi:
        rep.add("WARN", cache, "taskid-multiple-epochs",
                f"{len(multi)} task id(s) carry more than one epoch -- the two "
                f"compiled.pkl sources disagree on trial timing", label)

    dup = df.duplicated(subset=["TaskField", "Channel"]).sum()
    if dup:
        rep.add("ERROR", cache, "duplicate-trial-channel",
                f"{dup} duplicated (TaskField, Channel) row(s)", label)

    durations = np.array([float(b) - float(a) for a, b in pairs["EpochStartStop"]])
    if durations.size:
        med = float(np.median(durations))
        short = int((durations < 0.5 * med).sum())
        if short:
            rep.add("WARN", cache, "short-epochs",
                    f"{short}/{durations.size} epoch(s) shorter than half the "
                    f"session median ({med:.2f}s) -- likely junk trials", label)

    # -- metadata ----------------------------------------------------------
    by_lower = defaultdict(set)
    for n in st.monkey_names:
        by_lower[n.lower()].add(n)
    collisions = {k: v for k, v in by_lower.items() if len(v) > 1}
    if collisions:
        shown = "; ".join(f"{'/'.join(sorted(v))}" for v in list(collisions.values())[:5])
        rep.add("ERROR", cache, "monkey-name-case",
                f"{len(collisions)} monkey(s) spelled inconsistently: {shown}", label)

    id_types = sorted({type(x).__name__ for x in df["MonkeyId"]}) \
        if "MonkeyId" in df.columns else []
    if len(id_types) > 1:
        rep.add("WARN", cache, "monkeyid-dtype",
                f"MonkeyId holds mixed python types {id_types}", label)

    for col in ("MonkeyName", "MonkeyGroup"):
        nulls = df[col].isna().sum()
        if nulls:
            rep.add("WARN", cache, "null-metadata",
                    f"{nulls} row(s) with null {col}", label)

    if "Location" in df.columns:
        locs = set(df["Location"].astype(str))
        if "Unknown" in locs:
            rep.add("WARN", cache, "unknown-location",
                    "Location='Unknown' -> NeuronIDs prefixed 'Unknown_'", label)
    if df["NeuronID"].astype(str).str.startswith("Unknown").any():
        rep.add("WARN", cache, "unknown-neuronid",
                "NeuronIDs begin with 'Unknown_'; region filters will drop them",
                label)

    # -- spike windows -----------------------------------------------------
    starts = np.array([float(a) for a, _ in df["EpochStartStop"]])
    stops = np.array([float(b) for _, b in df["EpochStartStop"]])
    mins = np.empty(len(df))
    maxs = np.empty(len(df))
    total = 0
    for i, x in enumerate(df["SpikeTimes"]):
        a = spike_array(x)
        total += a.size
        if a.size:
            mins[i], maxs[i] = a.min(), a.max()
        else:
            mins[i] = maxs[i] = np.nan
    st.spikes = total

    empty_frac = float(np.isnan(mins).mean())
    if empty_frac == 1.0:
        rep.add("ERROR", cache, "no-spikes", "every row has an empty spike list", label)

    valid = ~np.isnan(mins)
    is_unit = chans.str.contains("_Unit").to_numpy()
    pre = (starts - mins)[valid]              # >0 means spikes before onset
    st.observed_pre_s = float(pre.max()) if pre.size else 0.0

    if spec.pre_ms == 0:
        if st.observed_pre_s > EPS:
            rep.add("ERROR", cache, "prestim-in-strict-cache",
                    f"spikes up to {st.observed_pre_s * 1000:.0f} ms BEFORE onset in a "
                    f"strict-window cache -- a pre-stimulus build was written here",
                    label)
    else:
        want = spec.pre_ms / 1000.0
        # Only manually-sorted units get a real baseline in the exploded variant:
        # unsorted channels come from the already-clipped compiled.pkl, so judge
        # the window on the unit rows alone there.
        relevant = pre[is_unit[valid]] if spec.kind == "exploded" else pre
        got = float(relevant.max()) if relevant.size else 0.0
        if got <= EPS:
            rep.add("ERROR", cache, "prestim-missing",
                    f"pre{spec.pre_ms}ms cache has NO spikes before onset -- it is a "
                    f"strict-window cache under a pre-stimulus name", label)
        elif got < 0.5 * want:
            rep.add("WARN", cache, "prestim-short",
                    f"deepest pre-stimulus spike is {got * 1000:.0f} ms before onset, "
                    f"expected ~{spec.pre_ms} ms", label)

    # 1 ms, not EPS: the manual and SI streams bound the epoch differently (one
    # uses < epoch_stop, the other <=), so sub-millisecond overruns are expected
    # and say nothing about cache integrity.
    late = (maxs - stops)[valid]
    if late.size and late.max() > 1e-3:
        rep.add("WARN", cache, "spikes-after-offset",
                f"spikes up to {late.max() * 1000:.1f} ms after epoch stop", label)

    # -- exploded-only: do both streams cover the same trials? -------------
    # Only meaningful when the file actually carries both streams. A cache holding
    # nothing but sorted units (the old curated-channels-only builds) is not a
    # mismatch, it is a different scope.
    both_streams = st.unit_channels and st.unit_channels < st.channels
    if spec.kind == "exploded" and both_streams:
        g = df.assign(_u=is_unit).groupby("TaskField")["_u"].agg(["any", "all"])
        sorted_only = int(g["all"].sum())
        unsorted_only = int((~g["any"]).sum())
        if sorted_only or unsorted_only:
            rep.add("ERROR", cache, "stream-mismatch",
                    f"manual-sort and unsorted streams cover different trials: "
                    f"{sorted_only} trial(s) with units only, {unsorted_only} with "
                    f"unsorted channels only (of {len(g)}). The two compiled.pkl "
                    f"sources disagree and the merge became a concatenation", label)

    # -- unit numbering ----------------------------------------------------
    per_base = defaultdict(set)
    for c in chans.unique():
        if is_unit_channel(c):
            n = str(c).split("_Unit")[-1].strip()
            if n.isdigit():
                per_base[base_channel(c)].add(int(n))
    gaps = {b: sorted(v) for b, v in per_base.items()
            if v and sorted(v) != list(range(1, max(v) + 1))}
    if gaps:
        rep.add("WARN", cache, "unit-numbering-gap",
                f"non-contiguous unit numbers: "
                f"{', '.join(f'{b}={v}' for b, v in list(gaps.items())[:4])}", label)

    return st


# ---------------------------------------------------------------------------
# Cache-level pass
# ---------------------------------------------------------------------------
def audit_cache(spec: CacheSpec, rep: Report, scope, sorted_scope, max_files=None):
    rep.say()
    rep.say("=" * 78)
    rep.say(f" {spec.name}   [{spec.kind}"
            + (f", pre{spec.pre_ms}ms" if spec.pre_ms else ", strict window") + "]")
    rep.say("=" * 78)

    if not spec.path.exists():
        rep.say(f"  directory does not exist: {spec.path}")
        rep.add("INFO", spec.name, "cache-absent",
                f"directory does not exist: {spec.path}")
        return {}
    if not spec.files:
        rep.say(f"  directory is EMPTY: {spec.path}")
        rep.add("WARN", spec.name, "cache-empty", f"no .pkl files in {spec.path}")
        return {}

    lo, hi = dir_mtime_range(spec.files)
    size = sum(p.stat().st_size for p in spec.files)
    rep.say(f"  {len(spec.files)} file(s), {human_size(size)}, "
            f"written {human_ts(lo)} -> {human_ts(hi)}")
    if hi - lo > 7 * 86400:
        rep.add("WARN", spec.name, "mixed-vintage",
                f"files span {(hi - lo) / 86400:.0f} days ({human_ts(lo)} -> "
                f"{human_ts(hi)}); this cache was written in more than one run")

    stats: dict[str, FileStats] = {}
    files = spec.files
    if max_files and max_files < len(files):
        # Never let a truncated run read as a clean bill of health.
        files = files[:max_files]
        rep.say(f"  --max-files: reading only {len(files)} of {len(spec.files)} "
                f"file(s); the rest were NOT checked")
        rep.add("INFO", spec.name, "partial-run",
                f"only {len(files)}/{len(spec.files)} file(s) were read "
                f"(--max-files); coverage and cross-checks are incomplete")
    for path in files:
        sess = session_of(path)
        label = session_label(*sess) if sess else path.stem
        try:
            df = pd.read_pickle(path)
        except Exception as exc:
            rep.add("ERROR", spec.name, "unreadable",
                    f"{type(exc).__name__}: {exc}", label)
            continue
        st = check_file(df, spec, label, rep)
        st.mtime = path.stat().st_mtime
        stats[label] = st

    # ---- coverage against the expected scope -----------------------------
    have = {session_of(p) for p in files if session_of(p)}
    all_rounds, analysis_rounds, _ = scope
    expected, scope_name = None, ""
    if spec.kind == "exploded" and analysis_rounds:
        expected, scope_name = analysis_rounds, "InitialRegression sheet"
    elif spec.kind in ("sorted", "mua") and sorted_scope:
        expected, scope_name = sorted_scope, "sorting batch list"
    if expected:
        missing = expected - have
        extra = have - expected
        rep.say(f"  coverage: {len(have)}/{len(expected)} sessions from the {scope_name}")
        if missing:
            rep.add("WARN", spec.name, "sessions-missing",
                    f"{len(missing)} session(s) in the {scope_name} have no cache "
                    f"file: {fmt_sessions(session_label(*s) for s in sorted(missing))}")
        if extra:
            rep.add("INFO", spec.name, "sessions-extra",
                    f"{len(extra)} cached session(s) outside the {scope_name}: "
                    f"{fmt_sessions(session_label(*s) for s in sorted(extra))}")
        if all_rounds and spec.kind in ("sorted", "mua"):
            never = all_rounds - have
            if never:
                rep.add("INFO", spec.name, "rounds-never-cached",
                        f"{len(never)} recorded round(s) have no entry in this cache")

    # ---- task-id leakage between sessions --------------------------------
    owner = {}
    overlaps = []
    for label, st in stats.items():
        for other, ost in stats.items():
            if other >= label:
                continue
            shared = st.task_ids & ost.task_ids
            if shared:
                overlaps.append((label, other, len(shared)))
    for a, b, n in overlaps:
        rep.add("ERROR", spec.name, "taskid-leak",
                f"{a} and {b} share {n} task id(s) -- one of them holds the other's "
                f"trials", a)

    # ---- monkey-name spelling across the whole cache ---------------------
    seen = defaultdict(set)
    for st in stats.values():
        for n in st.monkey_names:
            seen[n.lower()].add(n)
    cross = {k: v for k, v in seen.items() if len(v) > 1}
    if cross:
        rep.add("ERROR", spec.name, "monkey-name-case-cache",
                f"{len(cross)} monkey(s) spelled two ways ACROSS sessions in this "
                f"cache: {'; '.join('/'.join(sorted(v)) for v in list(cross.values())[:6])}")

    if stats:
        tot_neurons = sum(s.neurons for s in stats.values())
        tot_trials = sum(s.trials for s in stats.values())
        rep.say(f"  totals: {tot_neurons} neurons, {tot_trials} trials, "
                f"{sum(s.spikes for s in stats.values()):,} spikes")
        if spec.pre_ms:
            deepest = max((s.observed_pre_s for s in stats.values()), default=0)
            rep.say(f"  deepest pre-stimulus spike observed: {deepest * 1000:.0f} ms "
                    f"(expected ~{spec.pre_ms} ms)")
    return stats


# ---------------------------------------------------------------------------
# Cross-cache checks
# ---------------------------------------------------------------------------
def check_derived(child: CacheSpec, parent: CacheSpec, cstats, pstats, rep: Report):
    """A derived cache must be a consistent descendant of its parent."""
    if not cstats or not pstats:
        return
    rep.say()
    rep.say(f"--- {child.name}  vs  {parent.name} ---")

    missing = sorted(set(pstats) - set(cstats))
    extra = sorted(set(cstats) - set(pstats))
    if extra:
        rep.add("ERROR", child.name, "derived-orphan",
                f"{len(extra)} session(s) present here but NOT in {parent.name}: "
                f"{fmt_sessions(extra)} -- derived from a superseded parent")
    if missing:
        sev = "INFO" if child.name.endswith("_filtered") else "WARN"
        rep.add(sev, child.name, "derived-incomplete",
                f"{len(missing)} session(s) in {parent.name} have no counterpart "
                f"here: {fmt_sessions(missing)}")

    stale, id_drift, fewer, trial_drift = [], [], [], []
    for label in sorted(set(cstats) & set(pstats)):
        c, p = cstats[label], pstats[label]
        if c.mtime < p.mtime - 60:
            stale.append(label)
        # A filtered/pre-stim child may hold FEWER neurons, never different ones.
        strangers = c.neuron_ids - p.neuron_ids
        if strangers:
            id_drift.append((label, len(strangers), sorted(strangers)[:2]))
        if child.pre_ms:
            # Widening the window changes which spikes are kept, never which
            # trials exist or how many spikes there are in total.
            if p.spikes and c.spikes < p.spikes:
                fewer.append(label)
            if c.trials != p.trials:
                trial_drift.append((label, p.trials, c.trials))

    if stale:
        rep.add("ERROR", child.name, "derived-stale",
                f"{len(stale)} file(s) OLDER than their {parent.name} source -- "
                f"rebuild needed: {fmt_sessions(stale)}")
    for label, n, sample in id_drift:
        rep.add("ERROR", child.name, "derived-neuronid-drift",
                f"{n} NeuronID(s) not present in {parent.name}, e.g. {sample} -- "
                f"this file was built from a different sort", label)
    if fewer:
        rep.add("WARN", child.name, "prestim-fewer-spikes",
                f"{len(fewer)} pre-stimulus file(s) hold FEWER spikes than the strict "
                f"parent, which should be impossible: {fmt_sessions(fewer)}")
    for label, want, got in trial_drift:
        rep.add("ERROR", child.name, "prestim-trial-drift",
                f"{got} trials here vs {want} in {parent.name}; widening the window "
                f"cannot change the trial table, so the two were built from "
                f"different inputs", label)
    if not (stale or id_drift or extra or fewer or trial_drift):
        rep.say("  consistent with parent (coverage, NeuronIDs, trials, mtimes)")


def check_summaries(data_root: Path, sorted_stats, rep: Report):
    """sorted_spike_summary vs what sorted_spike_cache actually contains.

    The summary is written by the same analyze_sorted_spikes run that writes the
    cache, so a disagreement means one of them survived from an earlier sort.
    """
    sdir = data_root / "sorted_spike_summary"
    rep.say()
    rep.say("--- sorted_spike_summary  vs  sorted_spike_cache ---")
    if not sdir.exists():
        rep.add("WARN", "sorted_spike_summary", "summary-dir-missing",
                f"{sdir} does not exist")
        return
    txts = sorted(sdir.glob("*_sorting_summary.txt"))
    lo, hi = dir_mtime_range(txts)
    rep.say(f"  {len(txts)} summary file(s), written {human_ts(lo)} -> {human_ts(hi)}")

    mismatch, orphan_cache, orphan_summary = [], [], []
    for label, st in sorted(sorted_stats.items()):
        m = SESSION_RE.match(label)
        if not m:
            continue
        path = sdir / summary_name(m.group(1), int(m.group(2)))
        if not path.exists():
            orphan_cache.append(label)
            continue
        text = path.read_text()
        if NO_UNITS_RE.search(text):
            rep.add("ERROR", "sorted_spike_cache", "summary-says-no-units",
                    "summary says 'no units in agreement' but a cache pickle exists "
                    "-- one of the two is left over from a previous sort", label)
            continue
        hit = UNITS_RE.search(text)
        if not hit:
            continue
        declared = int(hit.group(1))
        actual = st.channels          # one channel entry per consensus unit
        if declared != actual:
            mismatch.append((label, declared, actual))
        if st.mtime and abs(st.mtime - path.stat().st_mtime) > 6 * 3600:
            rep.add("WARN", "sorted_spike_cache", "summary-cache-mtime-gap",
                    f"cache and summary written more than 6 h apart "
                    f"({human_ts(st.mtime)} vs {human_ts(path.stat().st_mtime)})",
                    label)

    for path in txts:
        m = re.match(r"(\d{6})_round(\d+)_sorting_summary", path.stem)
        if not m:
            continue
        d = f"20{m.group(1)[:2]}-{m.group(1)[2:4]}-{m.group(1)[4:]}"
        label = session_label(d, int(m.group(2)))
        if label not in sorted_stats and not NO_UNITS_RE.search(path.read_text()):
            orphan_summary.append(label)

    for label, declared, actual in mismatch:
        rep.add("ERROR", "sorted_spike_cache", "summary-unit-count-mismatch",
                f"summary says {declared} units in agreement, cache holds {actual} "
                f"-- the cache and the summary come from different sorting runs",
                label)
    if orphan_cache:
        rep.add("WARN", "sorted_spike_cache", "summary-missing",
                f"{len(orphan_cache)} cached session(s) have no summary: "
                f"{fmt_sessions(orphan_cache)}")
    if orphan_summary:
        rep.add("WARN", "sorted_spike_summary", "cache-missing",
                f"{len(orphan_summary)} summary(ies) report units but have no cache "
                f"pickle: {fmt_sessions(orphan_summary)}")
    if not (mismatch or orphan_cache or orphan_summary):
        rep.say("  every summary agrees with its cache")


def check_analysis_cache(data_root: Path, all_stats, rep: Report):
    """Do the significance pkls still point at neurons that exist?"""
    adir = data_root / "analysis_cache"
    rep.say()
    rep.say("--- analysis_cache  vs  the caches it was derived from ---")
    if not adir.exists():
        rep.add("INFO", "analysis_cache", "absent", f"{adir} does not exist")
        return
    pkls = sorted(adir.glob("*.pkl"))
    if not pkls:
        rep.add("WARN", "analysis_cache", "empty", f"no .pkl files in {adir}")
        return

    universes = {
        "si_sorted": set().union(*[s.neuron_ids for s in
                                   all_stats.get("sorted_spike_cache_filtered", {}).values()]
                                 or [set()]),
        "threshold_mua": set().union(*[s.neuron_ids for s in
                                       all_stats.get("threshold_mua_spike_cache", {}).values()]
                                     or [set()]),
    }
    fallback = set().union(*[s.neuron_ids for s in
                             all_stats.get("sorted_spike_cache", {}).values()] or [set()])

    for path in pkls:
        try:
            df = pd.read_pickle(path)
        except Exception as exc:
            rep.add("ERROR", "analysis_cache", "unreadable",
                    f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(df, pd.DataFrame) or "NeuronID" not in df.columns:
            continue
        prefix = "threshold_mua" if path.name.startswith("threshold_mua") else "si_sorted"
        universe = universes.get(prefix) or fallback
        if not universe:
            rep.add("INFO", "analysis_cache", "no-universe",
                    f"{path.name}: the cache it derives from is not present, cannot verify")
            continue
        ids = set(df["NeuronID"].astype(str))
        gone = ids - universe
        rep.say(f"  {path.name}: {len(ids) - len(gone)}/{len(ids)} NeuronIDs resolve")
        if gone:
            rep.add("ERROR", "analysis_cache", "stale-neuronids",
                    f"{path.name}: {len(gone)}/{len(ids)} NeuronID(s) no longer exist "
                    f"in {prefix}, e.g. {sorted(gone)[:3]} -- rebuilt after a re-sort?")


def check_compiled_sources(data_root: Path, intan_base: Path, scope, rep: Report):
    """The two compiled.pkl trees the exploded cache merges.

    The unsorted stream comes from <data>/Cortana/compiled/<PickleFileName>.pkl and
    the manual-sort stream from <intan>/<monkey>/<date>/<folder>/compiled.pkl. When
    those two disagree the merge silently unions instead of failing, which is how a
    session ends up holding another round's trials.
    """
    rep.say()
    rep.say("--- compiled.pkl sources ---")
    cdir = data_root / "compiled"
    if not cdir.exists():
        rep.add("ERROR", "compiled", "compiled-dir-missing",
                f"{cdir} does not exist -- every exploded/MUA rebuild will fail. "
                f"NOTE: recording_metadata_reader.py still looks for this under the "
                f"REPO, not the data tree")
    else:
        pkls = list(cdir.glob("*.pkl"))
        lo, hi = dir_mtime_range(pkls)
        rep.say(f"  {cdir}: {len(pkls)} file(s), {human_ts(lo)} -> {human_ts(hi)}")

    repo_shadow = Path(DEFAULT_REPO) / SUBJECT_MONKEY / "compiled"
    if repo_shadow.exists():
        rep.add("ERROR", "compiled", "compiled-shadow-copy",
                f"a second compiled/ tree still exists inside the repo at "
                f"{repo_shadow}; recording_metadata_reader.py reads THIS one while "
                f"everything else reads {cdir}")

    if not intan_base.exists():
        rep.add("INFO", "compiled", "intan-absent",
                f"{intan_base} not reachable; per-session compiled.pkl comparison skipped")
        return

    all_rounds, _, _ = scope
    checked = mismatched = 0
    for date, round_no in sorted(all_rounds or []):
        folder = f"{date.replace('-', '')[2:]}_round{round_no}"
        rp = intan_base / SUBJECT_MONKEY / date / folder / "compiled.pkl"
        if not rp.exists():
            continue
        try:
            rdf = pd.read_pickle(rp)
        except Exception:
            continue
        checked += 1
        col = "TaskField" if "TaskField" in rdf.columns else (
            "TaskId" if "TaskId" in rdf.columns else None)
        if col is None:
            rep.add("ERROR", "compiled", "compiled-no-taskcol",
                    f"{rp} has neither TaskField nor TaskId",
                    session_label(date, round_no))
            continue
        if "EpochStartStop" in rdf.columns and len(rdf):
            span = (min(a for a, _ in rdf["EpochStartStop"]),
                    max(b for _, b in rdf["EpochStartStop"]))
        else:
            span = (np.nan, np.nan)
        rep.add("INFO", "compiled", "round-compiled",
                f"{rp.parent.name}/compiled.pkl: {rdf[col].nunique()} trials, "
                f"epochs {span[0]:.1f}-{span[1]:.1f}s",
                session_label(date, round_no))
    rep.say(f"  inspected {checked} per-round compiled.pkl file(s) under {intan_base}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Read-only integrity audit of the spike caches.")
    p.add_argument("--data-root", default=None,
                   help=f"data tree holding the caches "
                        f"(default: $JULIE_DATA_PATH/{SUBJECT_MONKEY})")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help="repo root, for the metadata workbook and the batch list")
    p.add_argument("--intan-base", default=DEFAULT_INTAN,
                   help="Intan tree, for the per-round compiled.pkl comparison")
    p.add_argument("--cache", nargs="*", default=None,
                   help="audit only these cache directories (default: all found)")
    p.add_argument("--max-files", type=int, default=None,
                   help="only read the first N files of each cache (quick smoke test)")
    p.add_argument("--report", default=None, help="also write the text report here")
    p.add_argument("--json", dest="json_path", default=None,
                   help="also write the findings as JSON here")
    args = p.parse_args(argv)

    install_pickle_shims()

    data_root = Path(args.data_root) if args.data_root \
        else Path(DATA_BASE_PATH) / SUBJECT_MONKEY
    repo = Path(args.repo)
    rep = Report()

    rep.say("=" * 78)
    rep.say(" SPIKE CACHE AUDIT   (read-only -- no cache file is modified)")
    rep.say("=" * 78)
    rep.say(f" data root : {data_root}")
    rep.say(f" repo      : {repo}")
    rep.say(f" intan     : {args.intan_base}")
    rep.say(f" run at    : {datetime.now():%Y-%m-%d %H:%M:%S}")

    if not data_root.exists():
        rep.say(f"\nFATAL: data root does not exist: {data_root}")
        return 2

    scope = load_scope(repo, rep)
    sorted_scope = load_sorted_scope(repo)
    if scope[1]:
        rep.say(f" scope     : {len(scope[0])} recorded rounds, "
                f"{len(scope[1])} in InitialRegression"
                + (f", {len(sorted_scope)} in the sorting batch list"
                   if sorted_scope else ""))

    specs = discover_caches(data_root, only=args.cache)
    all_stats = {}
    for spec in specs:
        all_stats[spec.name] = audit_cache(spec, rep, scope, sorted_scope,
                                           max_files=args.max_files)

    rep.say()
    rep.say("=" * 78)
    rep.say(" CROSS-CACHE CONSISTENCY")
    rep.say("=" * 78)
    by_name = {s.name: s for s in specs}
    for spec in specs:
        parent = by_name.get(spec.parent) if spec.parent else None
        if parent:
            check_derived(spec, parent, all_stats.get(spec.name),
                          all_stats.get(parent.name), rep)

    if "sorted_spike_cache" in all_stats:
        check_summaries(data_root, all_stats["sorted_spike_cache"], rep)
    check_analysis_cache(data_root, all_stats, rep)
    check_compiled_sources(data_root, Path(args.intan_base), scope, rep)

    # ---- findings --------------------------------------------------------
    rep.say()
    rep.say("=" * 78)
    rep.say(" FINDINGS")
    rep.say("=" * 78)
    n_err, n_warn, n_info = rep.counts()
    rep.say(f" {n_err} error(s), {n_warn} warning(s), {n_info} note(s)")

    ordered = sorted(rep.findings,
                     key=lambda f: (SEVERITY_ORDER[f.severity], f.cache, f.code,
                                    f.session))
    current = None
    for f in ordered:
        if f.severity != current:
            current = f.severity
            rep.say()
            rep.say(f" --- {current} ---")
        where = f"{f.cache}" + (f" / {f.session}" if f.session else "")
        rep.say(f"  [{f.code}] {where}")
        for line in _wrap(f.message, 72):
            rep.say(f"      {line}")

    if not rep.findings:
        rep.say(" nothing to report -- every cache checked out clean")

    if args.report:
        Path(args.report).write_text("\n".join(rep.lines) + "\n")
        print(f"\nreport written to {args.report}")
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([f.as_dict() for f in ordered], indent=2) + "\n")
        print(f"findings written to {args.json_path}")

    return 1 if n_err else 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + len(w) + 1 > width:   # never emit a blank line for
            out.append(line)                          # an over-long token (a path)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
