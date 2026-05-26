# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_css_files = ['custom.css']
html_static_path = ['_static']

myst_heading_anchors = 3