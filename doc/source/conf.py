# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RGS-CDSS'
copyright = '2025, Eodyne Systems'
author = 'Eodyne Systems'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",         # Extracts docstrings from code
    "sphinx.ext.napoleon",        # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.autosummary",     # Generates API documentation
    "sphinx.ext.viewcode",        # Adds links to source code
    "sphinx_autodoc_typehints",   # Shows type hints in documentation
    "sphinx.ext.graphviz",        # Enables Graphviz diagrams
    "sphinx.ext.inheritance_diagram",  # Class hierarchy visualization
]

# -- Napoleon Settings (For Google/NumPy Docstrings) ------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False  # Don't document __init__ separately
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc Settings (Controls Function/Class Documentation) ----------------
autodoc_default_options = {
    "members": True,           # Include all documented members
    "undoc-members": False,    # Exclude undocumented members
    "private-members": False,  # Exclude private methods (_method_name)
    "special-members": "__init__",  # Include __init__ method
    "show-inheritance": True,  # Show class inheritance
}
autosummary_generate = True  # Automatically create stub pages for modules

# -- Type Hint Settings ------------------------------------------------------
autodoc_typehints = "description"  # Show type hints inline with parameter descriptions

# -- HTML Output Configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme (pip install sphinx-rtd-theme)
html_static_path = ["_static"]
templates_path = ['_templates']

# -- Exclude Patterns (Ignore Certain Files) ---------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Graphviz Diagrams ------------------------------------------------------
graphviz_output_format = "svg"

# -- Extensions for Better Documentation -------------------------------------
source_suffix = ".rst"
master_doc = "index"
