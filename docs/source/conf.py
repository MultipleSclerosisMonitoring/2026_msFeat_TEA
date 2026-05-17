import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "MS-Feat"
author = "Teresa Estevan Autrán"
release = "0.2.0"

html_theme = "sphinx_rtd_theme"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "alabaster"
html_static_path = ["_static"]

graphviz_output_format = "svg"
