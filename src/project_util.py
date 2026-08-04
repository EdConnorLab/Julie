import os

# Path to the git repository itself (code, tracked inputs, small outputs).
PROJECT_BASE_PATH = os.environ.get(
    "JULIE_PROJECT_PATH", "/home/connorlab/Documents/GitHub/Julie"
)

# Path to the data tree, deliberately kept OUTSIDE the repository so that git
# checkouts can never delete or overwrite it. Everything that used to live
# under <repo>/Cortana/ now lives under <DATA_BASE_PATH>/Cortana/.
DATA_BASE_PATH = os.environ.get(
    "JULIE_DATA_PATH", "/home/connorlab/Documents/JulieData"
)

SUBJECT_MONKEY = "Cortana"

# Convenience root for the subject's data directories, e.g.
# DATA_ROOT / "analysis_cache" / "si_sorted_Zombies_response_windows.pkl"
DATA_ROOT = os.path.join(DATA_BASE_PATH, SUBJECT_MONKEY)
