import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "MS-Feat"
author = "Teresa Estevan Autrán"
release = "0.2.0"

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
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"
autoclass_content = "both"
python_use_unqualified_type_names = True
add_module_names = False
toc_object_entries_show_parents = "hide"

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "MS-Feat Documentation"
html_short_title = "MS-Feat"
html_theme_options = {
    "description": "Clinical gait-analysis pipeline documentation",
    "fixed_sidebar": True,
    "page_width": "1220px",
    "sidebar_width": "290px",
    "code_font_size": "0.84em",
    "body_text_align": "justify",
    "show_powered_by": False,
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}
html_css_files = ["custom.css"]

graphviz_output_format = "svg"
