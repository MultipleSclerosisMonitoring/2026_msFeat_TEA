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

Typical targeted workflow:

1. Check whether the selected CSV rows are already present in HDF5:

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --ids 87 89 --check-only

2. Extract only the desired cases:

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --ids 87 89

3. Inspect the available HDF5 keys if needed:

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --list-hdf5-keys

4. Analyze a specific extracted foot without editing the config file:

.. code-block:: bash

   poetry run analyze-gait --config config/config.yaml \
     --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right

analyze-gait
------------

.. code-block:: bash

   poetry run analyze-gait --config config/config.yaml

You can also override the configured HDF5 key directly from the command line:

.. code-block:: bash

   poetry run analyze-gait --config config/config.yaml \
     --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right

This is especially useful when comparing multiple patients, tests or feet
without editing ``config/config.yaml`` between runs.
