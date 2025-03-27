# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

import ai_cdss

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(__file__))
# sys.path.append(str(Path(".").resolve()))
# sys.path.insert(0, os.path.abspath("../../src"))  # adjust if needed

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
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",        # Adds links to source code
    "sphinx_autodoc_typehints",   # Shows type hints in documentation
    "sphinx.ext.graphviz",        # Enables Graphviz diagrams
    "sphinx.ext.inheritance_diagram",  # Class hierarchy visualization
    "numpydoc",
    "sphinxcontrib.mermaid",
    "sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    "style.css",
]
html_js_files = ['modal.js']

htmlhelp_basename = 'CDSSDoc'

# -- PyData theme Configuration -----------------------------------------------

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
    "collapse_navigation": True,  # Keep expanded navigation
    "show_toc_level": 0,
    "show_nav_level": 2,
    "show_prev_next": True,        # Hides the previous/next buttons
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
}

# -- NumPyDoc configuration -----------------------------------------------------

numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False  # <- This hides Config

# -- Sphinx Gallery conf ----------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],  # Your example scripts live here
    "gallery_dirs": ["auto_examples"],  # Where to output the generated gallery
    "backreferences_dir": "generated",  # Optional: for cross-links from API
    "filename_pattern": r".*\.py$",  # Include all .py files
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "line_numbers": False,
    "plot_gallery": True,
    "abort_on_example_error": True,
    "min_reported_time": 0.0,
    "doc_module": ("ai_cdss",),
    "reference_url": {"ai_cdss": None},
}

# -- Autodoc Settings (Controls Function/Class Documentation) ----------------

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_options = {"inherited-members": None}

# -- Type Hint Settings ------------------------------------------------------

autodoc_typehints = "none"  # Show type hints inline with parameter descriptions

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

# --- Pandera Schema Docstring Injection ---

import pandera as pa
from ai_cdss import models  # Make sure ai_cdss is on sys.path

def format_check(check) -> str:
    """Format a Pandera Check into a readable string."""
    name = check.name
    stats = check.statistics

    try:
        if name == "greater_than":
            return f"> {stats.get('min_value', stats.get('value'))}"
        elif name == "greater_than_or_equal_to":
            return f"≥ {stats.get('min_value', stats.get('value'))}"
        elif name == "less_than":
            return f"< {stats.get('max_value', stats.get('value'))}"
        elif name == "less_than_or_equal_to":
            return f"≤ {stats.get('max_value', stats.get('value'))}"
        elif name == "in_range":
            min_val = stats["min_value"]
            max_val = stats["max_value"]
            inclusive = stats.get("inclusive", (True, True))
            left = f"{min_val} ≤" if inclusive[0] else f"{min_val} <"
            right = f"x ≤ {max_val}" if inclusive[1] else f"x < {max_val}"
            return f"{left} {right}"
        elif name == "isin":
            values = stats.get("allowed_values", [])
            return f"isin: {list(values)}"
        else:
            return name  # Fallback for other checks
    except Exception:
        return str(check)  # fallback if formatting fails

def generate_schema_rst(schema_class: pa.DataFrameModel) -> str:
    """
    Generate a reStructuredText list-table from a Pandera DataFrameModel schema.
    """
    lines = [
        "",
        ".. list-table:: Schema Fields",
        "   :header-rows: 1",
        "",
        "   * - Column",
        "     - Type",
        "     - Validation"
    ]

    for col_name, col in schema_class.to_schema().columns.items():
        dtype = str(col.dtype)
        checks = []

        if col.nullable:
            checks.append("nullable")
        if not col.required:
            checks.append("optional")
        if col.checks:
            checks.extend(format_check(check) for check in col.checks)

        validation_str = "; ".join(checks) if checks else "—"

        lines.append(f"   * - {col_name}")
        lines.append(f"     - {dtype}")
        lines.append(f"     - {validation_str}")

    return "\n".join(lines)

def inject_schema_docstrings():
    """
    Append autogenerated table to schema class docstrings.
    """
    schema_classes = [
        models.PPFSchema,
        models.PCMSchema,
        models.SessionSchema,
        models.TimeseriesSchema,
        models.ScoringSchema,
    ]

    for schema in schema_classes:
        schema.__doc__ = (schema.__doc__ or "") + generate_schema_rst(schema)

def skip_inherited_and_config(app, what, name, obj, skip, options):
    if name == "Config":
        return True
    if hasattr(obj, "__objclass__"):
        defining_class = obj.__objclass__
        if "pa" in defining_class.__module__:
            return True
    return skip

# Sphinx hook to run custom logic before build
def setup(app):
    inject_schema_docstrings()
    app.connect("autodoc-skip-member", skip_inherited_and_config)