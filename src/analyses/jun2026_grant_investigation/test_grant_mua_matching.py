"""
Unit tests for grant_mua_matching (deprecated grant Cell -> threshold-MUA NeuronID).

Pure stdlib (unittest + csv); no pandas / numpy / spike cache required, so these run
anywhere. Covers every guard path plus a real-data check against the verbatim NeuronID
strings in Cortana/cell_list_investigation/si_sorted_*_passed.csv.

Run:  python3 -m unittest test_grant_mua_matching -v
  or: python3 test_grant_mua_matching.py
"""

import csv
import os
import sys
import unittest
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grant_mua_matching import (  # noqa: E402
    is_mua_name, mua_neuron_ids, match_channel,
    match_grant_cells_to_mua_neuronids, summarize_problems,
)

# Minimal duck-type for a grant cell (unit_lists.RasterRequest exposes the same names).
Cell = namedtuple('Cell', 'date round_no match_value window_ms')

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_KW_CSV = REPO_ROOT / 'Cortana' / 'cell_list_investigation' / \
    'si_sorted_Zombies_significant_windows_pKW_passed.csv'


def const_session(ids):
    """A session_ids_fn returning the same id list for every (date, round)."""
    return lambda date, round_no: ids


class TestNamePredicates(unittest.TestCase):
    def test_is_mua_name(self):
        self.assertTrue(is_mua_name('Channel.C_020'))
        self.assertTrue(is_mua_name('AMG_2023-09-26_2_Channel.C_018'))
        self.assertFalse(is_mua_name('Channel.C_027_Unit 1'))
        self.assertFalse(is_mua_name('AMG_2023-09-26_2_Channel.C_018_Unit 1'))

    def test_mua_neuron_ids_filters_sorted(self):
        ids = ['AMG_2023-09-26_2_Channel.C_018',
               'AMG_2023-09-26_2_Channel.C_018_Unit 1',
               'AMG_2023-09-26_2_Channel.C_021_Unit 2',
               'ER_2023-10-03_1_Channel.C_005']
        self.assertEqual(mua_neuron_ids(ids),
                         ['AMG_2023-09-26_2_Channel.C_018', 'ER_2023-10-03_1_Channel.C_005'])


class TestMatchChannel(unittest.TestCase):
    def test_exact_whole_channel_match(self):
        ids = ['AMG_2023-09-26_2_Channel.C_018', 'AMG_2023-09-26_2_Channel.C_020']
        self.assertEqual(match_channel('Channel.C_018', ids), ['AMG_2023-09-26_2_Channel.C_018'])

    def test_zero_padding_not_confused(self):
        # C_002 must not match C_020 or C_012, and vice versa.
        ids = ['AMG_2023-09-26_2_Channel.C_002',
               'AMG_2023-09-26_2_Channel.C_020',
               'AMG_2023-09-26_2_Channel.C_012']
        self.assertEqual(match_channel('Channel.C_002', ids), ['AMG_2023-09-26_2_Channel.C_002'])
        self.assertEqual(match_channel('Channel.C_020', ids), ['AMG_2023-09-26_2_Channel.C_020'])

    def test_mua_channel_does_not_match_sorted_unit(self):
        # The key precision guarantee: an unsorted channel token is not a suffix of a
        # sorted NeuronID for the same channel.
        sorted_ids = ['AMG_2023-09-26_2_Channel.C_018_Unit 1',
                      'AMG_2023-09-26_2_Channel.C_018_Unit 2']
        self.assertEqual(match_channel('Channel.C_018', sorted_ids), [])


class TestMatchGrantCells(unittest.TestCase):
    def test_happy_path_single_match(self):
        cells = [Cell('2023-09-26', 2, 'Channel.C_018', (50.0, 150.0))]
        session = const_session(['AMG_2023-09-26_2_Channel.C_018',
                                 'AMG_2023-09-26_2_Channel.C_021'])
        matched, problems = match_grant_cells_to_mua_neuronids(cells, session)
        self.assertEqual(problems, [])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]['neuron_id'], 'AMG_2023-09-26_2_Channel.C_018')
        self.assertEqual(matched[0]['window_ms'], (50.0, 150.0))

    def test_no_match_reported(self):
        cells = [Cell('2023-09-26', 2, 'Channel.C_099', (0.0, 300.0))]
        matched, problems = match_grant_cells_to_mua_neuronids(
            cells, const_session(['AMG_2023-09-26_2_Channel.C_018']))
        self.assertEqual(matched, [])
        self.assertEqual(summarize_problems(problems), {'no_match': 1})

    def test_ambiguous_reported(self):
        # Two candidates sharing the channel suffix -> flagged, not silently taken.
        cells = [Cell('2023-09-26', 2, 'Channel.C_018', (0.0, 300.0))]
        dupes = ['AMG_2023-09-26_2_Channel.C_018', 'ER_2023-09-26_2_Channel.C_018']
        matched, problems = match_grant_cells_to_mua_neuronids(cells, const_session(dupes))
        self.assertEqual(matched, [])
        self.assertEqual(summarize_problems(problems), {'ambiguous': 1})

    def test_missing_session_reported(self):
        cells = [Cell('2099-01-01', 9, 'Channel.C_018', (0.0, 300.0))]
        matched, problems = match_grant_cells_to_mua_neuronids(
            cells, lambda d, r: None)  # session cache absent
        self.assertEqual(matched, [])
        self.assertEqual(summarize_problems(problems), {'missing_session': 1})

    def test_sorted_cell_skipped(self):
        cells = [Cell('2023-09-26', 2, 'Channel.C_018_Unit 1', (0.0, 300.0))]
        matched, problems = match_grant_cells_to_mua_neuronids(
            cells, const_session(['AMG_2023-09-26_2_Channel.C_018_Unit 1']))
        self.assertEqual(matched, [])
        self.assertEqual(summarize_problems(problems), {'skipped_sorted': 1})

    def test_no_window_reported(self):
        cells = [Cell('2023-09-26', 2, 'Channel.C_018', None)]
        matched, problems = match_grant_cells_to_mua_neuronids(
            cells, const_session(['AMG_2023-09-26_2_Channel.C_018']))
        self.assertEqual(matched, [])
        self.assertEqual(summarize_problems(problems), {'no_window': 1})

    def test_every_cell_accounted_for(self):
        cells = [
            Cell('2023-09-26', 2, 'Channel.C_018', (0.0, 300.0)),   # match
            Cell('2023-09-26', 2, 'Channel.C_099', (0.0, 300.0)),   # no_match
            Cell('2023-09-26', 2, 'Channel.C_021_Unit 1', (0.0, 1.0)),  # skipped_sorted
        ]
        session = const_session(['AMG_2023-09-26_2_Channel.C_018',
                                 'AMG_2023-09-26_2_Channel.C_021_Unit 1'])
        matched, problems = match_grant_cells_to_mua_neuronids(cells, session)
        self.assertEqual(len(matched) + len(problems), len(cells))
        self.assertEqual(len(matched), 1)

    def test_session_fn_called_once_per_session(self):
        calls = []

        def counting_fn(date, round_no):
            calls.append((date, round_no))
            return ['AMG_2023-09-26_2_Channel.C_018', 'AMG_2023-09-26_2_Channel.C_021']

        cells = [Cell('2023-09-26', 2, 'Channel.C_018', (0.0, 300.0)),
                 Cell('2023-09-26', 2, 'Channel.C_021', (0.0, 300.0))]
        matched, problems = match_grant_cells_to_mua_neuronids(cells, counting_fn)
        self.assertEqual(len(matched), 2)
        self.assertEqual(calls, [('2023-09-26', 2)])  # cached: one call, not two


class TestRealNeuronIDs(unittest.TestCase):
    """Validate the matching mechanism against the verbatim NeuronID strings shipped in
    the repo (these are sorted units, so we exercise match_channel directly)."""

    @classmethod
    def setUpClass(cls):
        if not SI_KW_CSV.exists():
            raise unittest.SkipTest(f"reference CSV not present: {SI_KW_CSV}")
        cls.by_session = {}
        with open(SI_KW_CSV, newline='') as fh:
            for row in csv.DictReader(fh):
                key = (row['Date'], int(row['Round No.']))
                cls.by_session.setdefault(key, set()).add(row['NeuronID'])
        cls.by_session = {k: sorted(v) for k, v in cls.by_session.items()}
        cls.all_nids = [nid for nids in cls.by_session.values() for nid in nids]
        if not cls.all_nids:
            raise unittest.SkipTest("reference CSV had no rows")

    def test_neuronid_structure(self):
        for nid in self.all_nids:
            parts = nid.split('_', 3)
            self.assertEqual(len(parts), 4, f"unexpected NeuronID shape: {nid}")
            self.assertTrue(parts[3].startswith('Channel.'), f"no Channel token: {nid}")

    def test_full_channel_token_matches_uniquely_within_session(self):
        # For each real NeuronID, its channel token resolves to exactly itself within
        # its own session -- proving the endswith rule is unambiguous on real data.
        checked = 0
        for (date, round_no), nids in self.by_session.items():
            for nid in nids:
                channel_token = nid.split('_', 3)[3]           # e.g. 'Channel.C_018_Unit 1'
                hits = match_channel(channel_token, nids)
                self.assertEqual(hits, [nid], f"{channel_token} in {date}/{round_no} -> {hits}")
                checked += 1
        self.assertGreater(checked, 0)
        print(f"\n[real-data] verified unique channel->NeuronID match for {checked} "
              f"units across {len(self.by_session)} sessions")

    def test_unsorted_channel_excludes_sorted_units(self):
        # Deriving the bare MUA channel from a sorted token must NOT match that sorted
        # unit -- the guarantee that grant MUA cells map only to MUA NeuronIDs.
        for nid in self.all_nids:
            channel_token = nid.split('_', 3)[3]
            if '_Unit' not in channel_token:
                continue
            mua_channel = channel_token.split('_Unit')[0]      # 'Channel.C_018'
            self.assertNotIn(nid, match_channel(mua_channel, [nid]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
