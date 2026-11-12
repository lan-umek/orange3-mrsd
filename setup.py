#!/usr/bin/env python
"""Orange3 add-on: Subgroups (Multiple Responses Subgroup Discovery)."""
import os

from setuptools import find_packages, setup

NAME = "Orange3-MRSD"
VERSION = "0.1.0"
DESCRIPTION = "Orange add-on for discovering subgroups in data with several dependent variables (MRSD)."
AUTHOR = "Lan Umek"
URL = "https://github.com/lan-umek/orange3-mrsd"
LICENSE = "MIT"
KEYWORDS = ("orange3 add-on", "subgroup discovery", "multiple responses", "MRSD", "exceptional model mining")

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

PACKAGES = find_packages()

PACKAGE_DATA = {
    "orangecontrib.subgroups.widgets": ["icons/*.svg"],
    "orangecontrib.subgroups.datasets": ["*.tab", "*.csv"],
    "orangecontrib.subgroups.mrsd": ["*.md"],
}

INSTALL_REQUIRES = [
    "Orange3>=3.34",
    "numpy>=1.23",
    "pandas>=1.5",
    "scipy>=1.9",
    "scikit-learn>=1.2",
    "pyqtgraph",
]

EXTRAS_REQUIRE = {
    "excel": ["openpyxl"],
    "test": ["pytest"],
}

ENTRY_POINTS = {
    # Orange add-on registration
    "orange3.addon": ("subgroups = orangecontrib.subgroups",),
    # widget category
    "orange.widgets": ("Subgroups = orangecontrib.subgroups.widgets",),
    # bundled datasets, found by name by Table(...) and the File widget
    "orange.data.io.search_paths": ("subgroups = orangecontrib.subgroups.datasets",),
}


CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Information Analysis",
]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        url=URL,
        license=LICENSE,
        keywords=KEYWORDS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        classifiers=CLASSIFIERS,
        python_requires=">=3.9",
        zip_safe=False,
    )
