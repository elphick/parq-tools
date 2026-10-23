Installation
============

.. toctree::
   :maxdepth: 2
   :caption: Installation:


The package is pip installable.

..  code-block::

    pip install parq-tools

If you want the extras (for profiling and visualisation) you'll install like this with pip.

.. code-block::

    pip install parq-tools[profiling,blockmodel]

For a uv-managed project:

..  code-block::

    uv add parq-tools

or with extras...

..  code-block::

    uv add "parq-tools[profiling,blockmodel]"

Development from source
-----------------------

Install uv, clone this repository, then run:

.. code-block:: powershell

    uv sync --locked --all-extras
    uv run --locked --all-extras pytest

The repository pins Python 3.13 for local development. uv downloads a compatible
interpreter when needed and creates a local ``.venv``. On Windows, select
``.venv\Scripts\python.exe`` as the interpreter in your IDE. Existing Poetry
environments are not modified. The package continues to support Python
3.11 through 3.13.

To update dependencies, edit ``pyproject.toml``, run ``uv lock``, and commit both
``pyproject.toml`` and ``uv.lock``. CI uses ``uv sync --locked --all-extras``
to keep installs reproducible.
