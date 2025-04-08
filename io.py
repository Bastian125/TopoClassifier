"""Input/output methods for checking that folders exist or creating them if neccessary."""

# ---------- Imports ---------- #
import os


# ---------- I/O Functions ---------- #
def ensure_dir_exists(path):
    """
    Ensure that the directory at 'path' exists.
    If it doesn't, create it.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory ensured: {path}")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        raise
