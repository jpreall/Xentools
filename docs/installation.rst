Installation
============

Development Install
-------------------

The recommended current install path is an editable install from the GitHub
repository:

.. code-block:: bash

   git clone https://github.com/jpreall/Xentools.git
   cd Xentools
   pip install -e .

If you are working from an existing checkout, update it first:

.. code-block:: bash

   cd /path/to/Xentools
   git pull
   pip install -e .

Documentation Dependencies
--------------------------

To build these docs locally, install the documentation extra:

.. code-block:: bash

   pip install -e ".[docs]"

Then build the HTML docs:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Test Dependencies
-----------------

To run the test suite:

.. code-block:: bash

   pip install -e ".[test]"
   pytest

Dependency Notes
----------------

xentools currently depends on the scientific Python stack used for Xenium-style
analysis, including ``numpy``, ``pandas``, ``scipy``, ``anndata``, ``scanpy``,
``geopandas``, ``shapely``, ``zarr``, ``pyarrow``, ``tifffile``, and
``matplotlib``.

The package is tested against ``zarr>=2.18.3`` and includes compatibility code
for multiple Xenium output layouts.
