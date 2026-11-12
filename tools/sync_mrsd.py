"""
Copy the MRSD package (the 'mrsd_subgroups' package of the 'subgroups'
repository next to this add-on, or of a repository given on the command line)
into orangecontrib/subgroups/mrsd.

    python tools/sync_mrsd.py [path/to/subgroups]

Only the Python modules of the package and README.md / DEV_NOTES.md of the
repository are copied.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON = os.path.dirname(HERE)
TARGET = os.path.join(ADDON, "orangecontrib", "subgroups", "mrsd")


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(ADDON), "subgroups")
    src = os.path.join(repo, "mrsd_subgroups")
    if not os.path.isfile(os.path.join(src, "__init__.py")):
        sys.exit(f"{repo} does not look like the MRSD repository (no mrsd_subgroups/__init__.py)")
    os.makedirs(TARGET, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(src, name), os.path.join(TARGET, name))
            copied.append(name)
    for name in ("README.md", "DEV_NOTES.md"):
        if os.path.isfile(os.path.join(repo, name)):
            shutil.copy2(os.path.join(repo, name), os.path.join(TARGET, name))
            copied.append(name)
    print(f"copied {len(copied)} files from {src} to {TARGET}:")
    for c in copied:
        print("  ", c)


if __name__ == "__main__":
    main()
