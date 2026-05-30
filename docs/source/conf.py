# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

#os.environ['NUMBA_DISABLE_JIT'] = '1'

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CubeX'
copyright = '2026, Ramlal Unnikrishnan'
author = 'Ramlal Unnikrishnan'
release = 'v0.3-preview'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the Markdown parser and the autodoc extension
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax'
]

# Force Sphinx to document private (_) and undocumented members
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,  # This is the magic key!
    'show-inheritance': True,
}

# Source Suffixes (Tell Sphinx to read both types)
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Enable MyST math rendering extensions
myst_enable_extensions = [
    "dollarmath",
    "amsmath"
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_css_files = ['custom.css']
html_static_path = ['_static']

myst_heading_anchors = 3


# Tell Sphinx to fake these imports so it doesn't crash on the cloud server
autodoc_mock_imports = [
    # Core Data & Math
    "numpy",
    "scipy",
    "pandas",
    
    # Astropy Ecosystem
    "astropy",
    "astroquery",
    "spectral_cube",
    
    # GUI & Visualization
    "PyQt5",
    "pyqtgraph",
    "matplotlib",
    "qtawesome",
    
    # Optional performance libraries
    "numba"
]