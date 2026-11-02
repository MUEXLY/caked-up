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

## Physical constraints

`calibration_settings.physical_constraints` defines monotonic prediction rules.
The GUI has a dedicated constraints page. A config row can look like:

```json
"physical_constraints": {
  "enabled": true,
  "mode": "soft",
  "strength": 1.0,
  "relationships": [
    {
      "predictor_type": "appdomain",
      "predictor": "x",
      "response_type": "observable",
      "response": "response",
      "direction": "decreasing"
    }
  ]
}
```

`predictor_type` is `appdomain` or `inputparam`; `response_type` can also be
`appdomain`, `inputparam`, or `observable`. Hard mode rejects violating
candidate states. Soft mode adds a quadratic log-posterior penalty, with larger
positive `strength` applying a stronger penalty. Observable responses use
emulator derivatives; predictor-to-predictor rules use values across adjacent
observation locations ordered by the selected predictor. The same settings
apply in IBFU, including its GP, lasso, fused-lasso, spike-slab, and horseshoe
kappa priors.
