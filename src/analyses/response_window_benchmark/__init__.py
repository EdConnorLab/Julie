"""
response_window_benchmark — compare response-window detection algorithms.

A standalone bake-off harness that runs several candidate high-firing-rate /
response-window detectors on a curated set of real example cells, marks each
method's detected windows on the raster + PSTH, and scores them against windows
the user annotates by eye.

This package deliberately does NOT modify the existing detector
(``analyses.response_window_finder``), the grant investigation
(``analyses.jun2026_grant_investigation``), or the plotting engine
(``analyses.zombies_raster_review``) — it reuses them.

See ``README.md`` for the workflow and ``run_benchmark.py`` for the entrypoint.
"""
