# -*- coding: utf-8 -*-
#
# kafe2 documentation build configuration file, adapted from old kafe conf.py
# Thu Mar 30 18:24:45 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
from datetime import datetime

# Mock modules: when building the documentation using autodoc, Sphinx
# imports the entire Python code, which can in turn import other packages.
# When building the documentation on systems which don't have these
# external packages installed, autodoc will fail to import them, which
# causes the entire build to fail. A workaround is to use 'mocks': packages
# that pretend to be the external modules so they can be imported, but
# don't actually do anything. This is needed for building the documentation
# on e.g. ReadTheDocs.org
from mock import MagicMock

MOCK_MODULES = [
    'matplotlib',
    'matplotlib.legend_handler',
    'matplotlib.pyplot',
    'matplotlib.collections',
    'matplotlib.axes',
    'iminuit',
    'numdifftools',
    'numpy',
    'ROOT',
    'scipy',
    'scipy.linalg',
    'scipy.misc',
    'scipy.optimize',
    'scipy.special',
    'scipy.stats',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
sys.modules['iminuit'].__version__ = "1.0.0"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('ext'))

import kafe2

print('kafe2 version:', kafe2.__version__)

# -- General configuration ------------------------------------------------

# style sheet customizations
def setup(app):
    app.add_css_file("style.css")

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.4'  # needed for imgmath extension

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.inheritance_diagram',
    'bootstrap_collapsible',
]

todo_include_todos = True
autodoc_member_order = 'bysource'

# set up intersphinx
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'iminuit': ('https://iminuit.readthedocs.io/en/latest', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'kafe2'
copyright = f'2019-{datetime.now().year}, J. Gäßler, C. Verstege, D. Savoiu and G. Quast'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = kafe2.__version__
# The full version, including alpha/beta/rc tags.
release = kafe2.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# Merge class and __init__ docstrings into one
autoclass_content = 'both'

# set inheritance graphs attributes
inheritance_graph_attrs = dict(rankdir="TB", size='""')

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'nature'

# use own modified RTD theme
#html_theme = 'rtd_theme_mod'
#html_theme = 'guzzle_sphinx_theme'
html_theme = 'bootstrap'

# some themes require further customization
if html_theme == 'bootstrap':
    import sphinx_bootstrap_theme
    html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
    html_theme_options = {
        'bootswatch_theme': "sandstone",
    }
elif html_theme == 'guzzle_sphinx_theme':
    import guzzle_sphinx_theme
    html_theme_path = guzzle_sphinx_theme.html_theme_path()

    # Register the theme as an extension to generate a sitemap.xml
    extensions.append("guzzle_sphinx_theme")

    # Guzzle theme options (see theme.conf for more information)
    html_theme_options = {
        # Set the name of the project to appear in the sidebar
        "project_nav_name": "kafe2",
    }
else:
    # Theme options are theme-specific and customize the look and feel of a theme
    # further.  For a list of options available for each theme, see the
    # documentation.
    html_theme_options = {
        #"collapse_navigation": False
    }

    # Add any paths that contain custom themes here, relative to this directory.
    html_theme_path = ['_themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/img/icon_kafe2.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/img/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'kafedoc'


# -- Options for LaTeX output ---------------------------------------------

latex_engine = 'xelatex'
latex_use_xindy = False

# For some reason building PDF document won't work without explicitly defining fonts.
latex_elements = {
    'fontpkg': r'''
\setmainfont{FreeSerif}
\setsansfont{FreeSerif}
\setmonofont{FreeMono}
''',

  # The paper size ('letterpaper' or 'a4paper').
  'papersize': 'a4paper',

  # The font size ('10pt', '11pt' or '12pt').
  'pointsize': '11pt',

  # Additional stuff for the LaTeX preamble.
  'preamble': r'''
\usepackage{enumitem}\setlistdepth{48}
\usepackage[titles]{tocloft}
\usepackage{bm}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class], toctree_only).
latex_documents = [
  ('index', 'kafe2.tex', u'kafe2 Documentation',
   u'J. Gäßler, C. Verstege, D. Savoiu, G. Quast', 'manual', False),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
latex_show_pagerefs = True

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('kafe2', 'kafe2', u'kafe2 Documentation',
     [u'D. Savoiu, G. Quast'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('kafe2', 'kafe2', u'kafe2 Documentation',
   u'J. Gäßler, C. Verstege, D. Savoiu, G. Quast', 'kafe2', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

# LaTeX-style references:
numfig = True
