Data Interfaces
===============

:py:mod:`ai_cdss.models`:

This module defines the main data validation interfaces for the Clinical Decision Support System (CDSS).  
Each class represents a structured Pandera schema used to validate key data inputs and outputs in the pipeline, including:

- **Session-level data** (e.g., patient demographics, protocol metadata, adherence, performance)
- **Time series records** (e.g., difficulty modulator and performance estimates across a session)
- **Patient Protocol Fit scores** between patients and protocols.
- **Protocol Similarity scores** pairwise protocol similarity based on protocol attributes.
- **Recommendation outputs** (e.g., protocol assignment scores and therapeutic parameters)

The schemas enforce strict typing and constraints to ensure data consistency throughout the system.  
All schemas inherit from `pandera.DataFrameModel` and are used both for runtime validation and structured documentation.

.. currentmodule:: ai_cdss.models

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class_no_inherited_members_pandera.rst
   :nosignatures:

   SessionSchema
   TimeseriesSchema
   PPFSchema
   PCMSchema

   ScoringSchema