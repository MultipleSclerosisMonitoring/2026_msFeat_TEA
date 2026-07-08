import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor_py"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import furo

project = "MS-Feat"
author = "Teresa Estevan Autrán"
release = "0.2.0"

extensions = [
    "furo.sphinxext",
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
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"
autoclass_content = "both"
python_use_unqualified_type_names = True
add_module_names = False
toc_object_entries_show_parents = "hide"

html_theme = "furo"
html_theme_path = [
    str(Path(__file__).resolve().parent / "_vendor_py" / "furo" / "theme"),
    str(Path(__file__).resolve().parent / "_vendor_py" / "sphinx_basic_ng" / "theme"),
]
html_static_path = ["_static"]
html_title = "MS-Feat Documentation"
html_short_title = "MS-Feat"
html_logo = "_static/upm-logo.png"
html_favicon = "_static/upm-logo.png"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0b4f6c",
        "color-brand-content": "#9a6b14",
        "color-api-name": "#0b4f6c",
        "color-api-pre-name": "#9a6b14",
        "color-background-primary": "#fffdf8",
        "color-background-secondary": "#f7f1e6",
        "color-sidebar-background": "#123047",
        "color-sidebar-background-border": "#1f4f67",
        "color-sidebar-link-text": "#f3e7c9",
        "color-sidebar-link-text--top-level": "#ffffff",
        "color-sidebar-item-background--hover": "#1f4f67",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8cc2d9",
        "color-brand-content": "#f0c879",
        "color-api-name": "#8cc2d9",
        "color-api-pre-name": "#f0c879",
    },
}
html_css_files = ["custom.css"]

graphviz_output_format = "svg"


def setup(app):
    return furo.setup(app)
