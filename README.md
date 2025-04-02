
# ai-cdss: Clinical Decision Support System

<div class="column" align="middle">
    <a href="https://dabadav.github.io/ai-cdss/index.html">
        <img src="https://img.shields.io/badge/Docs-online-green?label=Documentation" alt="Docs"/>
    </a>
    <a href="https://github.com/dabadav/ai-cdss/actions/workflows/test.yml">
        <img src="https://github.com/dabadav/ai-cdss/actions/workflows/test.yml/badge.svg" alt="CI - Test"/>
    </a>
    <a href="https://github.com/dabadav/ai-cdss/releases/latest">
        <img src="https://img.shields.io/github/v/release/dabadav/ai-cdss?label=GitHub%20Release" alt="GitHub Release"/>
    </a>
</div>

## Documentation

[`Documentation`](https://dabadav.github.io/ai-cdss/index.html) for the ai-cdss encompasses installation instructions, tutorials, examples and an API reference.

## Installation

To install the lastest stable version of the ai-cdss, use pip in a terminal:

```bash
pip install ai_cdss-0.1.0.tar.gz
```

## Dependencies

This project requires the following Python packages:

- `rgs_interface` — Custom interface for DB reading and table merging.
  
  ```bash
  pip install git+https://github.com/dabadav/rgs-interface.git@v0.2.2
  ```
- `pandas[parquet]` (>=2.2.3,<3.0.0) — Pandas with Parquet support for efficient I/O.
- `pandera` (>=0.23.1,<0.24.0) — Data validation for Pandas.
- `gower` (>=0.1.2,<0.2.0) — Gower distance metric implementation for mixed-type data.
