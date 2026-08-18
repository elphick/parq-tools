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

Or, if poetry is more your flavour.

..  code-block::

    poetry add parq-tools

or with extras...

..  code-block::

    poetry add "parq-tools[profiling,blockmodel]"
