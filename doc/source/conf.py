# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

import ai_cdss

sys.path.insert(0, os.path.abspath('.'))
sys.path.append(str(Path(".").resolve()))

# -- Project information -----------------------------------------------------

project = 'Clinical Decision Support System (CDSS)'
copyright = '2025, Eodyne Systems'
author = 'Eodyne Systems'

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # Enables Markdown support
    "sphinx_design",
    "sphinx.ext.autodoc",         # Extracts docstrings from code
    "sphinx.ext.autosummary",     # Generates API documentation
    "sphinx.ext.napoleon",        # Supports Google-style & NumPy-style docstrings
    "sphinx.ext.viewcode",        # Adds links to source code
    "sphinx_autodoc_typehints",   # Shows type hints in documentation
    "sphinx.ext.graphviz",        # Enables Graphviz diagrams
    "sphinx.ext.inheritance_diagram",  # Class hierarchy visualization
    "sphinxcontrib.sphinx_pandera",  # Render pandera schemas
    # "sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    "style.css",
]

htmlhelp_basename = 'CDSSDoc'

# -- PyData heme Configuration -----------------------------------------------

html_title = "CDSS"  # You can modify this as well
html_show_sourcelink = False

html_theme_options = {
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/dabadav/ai-cdss",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "navbar_align": "left",
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "navigation_depth": 3,
    "collapse_navigation": False,  # Keep expanded navigation
    "show_toc_level": 0,
    "show_nav_level": 2,
    "show_prev_next": False,        # Hides the previous/next buttons
    "footer_start": ["copyright"],
    "footer_end": []
}

# Custom navbar texts and additional tabs
html_context = {
    "navbar_title": "CDSS",
    # Add these lines for the edit page button
    "github_user": "dabadav",  # Your GitHub username
    "github_repo": "ai-cdss",  # Your repository name
    "github_version": "main",  # The branch name (e.g., main, master)
    "doc_path": "doc",        # The path to your documentation in the repository
    'carousel': [
        {'title': 'Test 1', 'text': 'This is test card 1', 'url': '#'},
        {'title': 'Test 2', 'text': 'This is test card 2', 'url': '#'}
    ]
    # "carousel": [
    #     dict(
    #         title="Activity Evaluation",
    #         text="Receptive field estima\u00adtion with optional smooth\u00adness priors.",
    #         url="guide/introduction.html",
    #         # img="",
    #         # alt="",
    #     ),
    #     dict(
    #         title="Therapeutic Interchange",
    #         text="Receptive field estima\u00adtion with optional smooth\u00adness priors.",  # noqa E501
    #         url="api/index.html",
    #         # img=".png",
    #         # alt="STRF",
    #     ),
    # ]
}

# -- Sphinx Gallery conf ----------------------------------------------------

# sphinx_gallery_conf = {
#     'examples_dirs': None,  # Disable examples scanning
# }


# -- Napoleon Settings (For Google/NumPy Docstrings) ------------------------

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False  # Don't document __init__ separately
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc Settings (Controls Function/Class Documentation) ----------------

autosummary_generate = True  # Automatically create stub pages for modules
autodoc_default_options = {"inherited-members": None}
autodoc_inherit_docstrings = False

# autodoc_default_options = {
#     "members": True,           # Include all documented members
#     "undoc-members": False,    # Exclude undocumented members
#     "private-members": False,  # Exclude private methods (_method_name)
#     "special-members": "__init__",  # Include __init__ method
#     "show-inheritance": True,  # Show class inheritance
# }

# -- Type Hint Settings ------------------------------------------------------

autodoc_typehints = "description"  # Show type hints inline with parameter descriptions

# -- Graphviz Diagrams ------------------------------------------------------

graphviz_output_format = "svg"

# -- MySt Enable Math --------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


# -- Extensions for Better Documentation -------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = "index"
