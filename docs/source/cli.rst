Command-Line Interface
======================

extract-data
------------

.. code-block:: bash

   poetry run extract-data --config config/config.yaml

Common selective extraction and audit examples:

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --ids 87 89
   poetry run extract-data --config config/config.yaml --codeid RHRHUG004-1 --test 6MWT
   poetry run extract-data --config config/config.yaml --ids 87 89 --check-only
   poetry run extract-data --config config/config.yaml --test 6MWT --missing-only
   poetry run extract-data --config config/config.yaml --list-hdf5-keys

Options added for extraction workflow management:

- ``--ids``: extract or audit only the specified CSV row ids
- ``--codeid``: extract or audit only the specified patient identifiers
- ``--missing-only``: skip rows already complete in HDF5 (both ``Left`` and ``Right``)
- ``--check-only``: compare the selected CSV rows against the HDF5 and report status
- ``--list-hdf5-keys``: print current HDF5 keys and exit

The ``--check-only`` mode reports whether each selected row is:

- ``complete``: both ``Left`` and ``Right`` keys exist
- ``left_only``: only the ``Left`` key exists
- ``right_only``: only the ``Right`` key exists
- ``missing``: neither foot is present

HDF5 trial keys are now self-contained and derived from the CSV start time in UTC:

.. code-block:: text

   p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right

analyze-gait
------------

.. code-block:: bash

   poetry run analyze-gait --config config/config.yaml
