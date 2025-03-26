Architecture
============

.. code-block:: text

    [DB or Synthetic Data] 
          └─> DataLoader 
                └─> Session Data
                └─> Timeseries Data
                └─> PPF
                └─> Protocol Similarity

                  └─> DataProcessor (weights, alpha)
                        └─> Score matrix: Patient x Protocol

                            └─> CDSS (n, days, protocols/day)
                                  └─> Final 7-day recommendations per patient