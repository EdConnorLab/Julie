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


def canonical_monkey_name(name):
    """The one spelling of a stimulus monkey's name: fully upper case.

    Every name in social_data/monkeyinfo.csv is upper case ('114J', 'G701',
    'DF2I'), and the social workbooks are keyed on that spelling. The recording
    database has since started returning the trailing letter in lower case for
    seven of them ('114j', '19j', '26j', '36j', '40j', '87j', '151j'), so trial
    metadata compiled at different times disagrees. Analyses that group or merge
    on MonkeyName then split one monkey into two, and an inner merge against the
    social data drops those trials without a word (see linear_regression.py).

    Canonicalise here, at every point where trial metadata enters the pipeline,
    rather than trusting any one source to be consistent.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        return name
    return name.strip().upper()

