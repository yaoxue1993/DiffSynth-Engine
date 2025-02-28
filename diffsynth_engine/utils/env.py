import os

DIFFSYNTH_CACHE = os.environ.get("DIFFSYNTH_CACHE", os.path.join(os.environ.get("HOME"), ".cache", "diffsynth"))

DIFFSYNTH_FILELOCK_DIR = os.environ.get(
    "DIFFSYNTH_FILELOCK_DIR", os.path.join(os.environ.get("HOME"), ".cache", "diffsynth", "filelocks")
)
