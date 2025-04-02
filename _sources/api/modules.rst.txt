CDSS
====

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

:py:mod:`ai_cdss`:

.. currentmodule:: ai_cdss

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class_no_inherited_members.rst

   DataLoader
   DataProcessor

:py:mod:`ai_cdss.cdss`:

.. currentmodule:: ai_cdss.cdss

.. autosummary::
    :toctree: ../generated/
    :template: autosummary/class_no_inherited_members.rst

    CDSS
