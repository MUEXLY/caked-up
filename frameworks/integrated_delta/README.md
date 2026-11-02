# Integrated Delta Calibration (IB)

This directory contains the local predecessor framework copied into the
CAKED-UP repository so both methods can be selected and run from the same
application. Its Python sources, config template, plotting utilities, and
validation helper originated in the local `integratedDelta_calibration`
repository.

The only execution-interface change is in `main.py`: it accepts an optional
config path argument, defaulting to `config.json`, so the GUI can pass a
run-specific config without overwriting this framework's template. Data paths
and output paths are supplied by the GUI's run config. The source project's
datasets and historical result folders are not copied here.
