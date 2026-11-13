"""State-level monotonicity and endpoint constraints for the emulator."""

import numpy as np
from scipy.optimize import minimize


def compile_directional_constraints(
    config,
    domain_labels,
    parameter_labels,
    observable_labels,
    parameter_transforms=None,
    observable_transforms=None,
):
    """Resolve configured constraint labels and physical endpoint values."""
    config = config or {}
    if not config.get("enabled", False):
        return {
            "enabled": False, "mode": "soft", "strength": 0.0,
            "rules": [], "boundary_rules": [], "parameter_sign_rules": [],
        }

    mode = str(config.get("mode", "soft")).lower()
    if mode not in {"soft", "hard"}:
        raise ValueError("physical_constraints.mode must be 'soft' or 'hard'")
    strength = float(config.get("strength", 1.0))
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("physical_constraints.strength must be finite and non-negative")

    label_sets = {
        "appdomain": [str(label) for label in domain_labels],
        "inputparam": [str(label) for label in parameter_labels],
        "observable": [str(label) for label in observable_labels],
    }

    def resolve_label(label, kind):
        """Accept both GUI labels and raw dataframe column prefixes."""
        name = str(label)
        if name in label_sets[kind]:
            return name
        prefixes = {
            "appdomain": ("x_",),
            "inputparam": ("theta_", "param_"),
            "observable": ("y_", "zeta_"),
        }[kind]
        candidate = name
        while candidate not in label_sets[kind]:
            prefix = next((item for item in prefixes if candidate.startswith(item)), None)
            if prefix is None:
                break
            candidate = candidate[len(prefix):]
        return candidate if candidate in label_sets[kind] else None

    rules = []
    for number, rule in enumerate(config.get("relationships", []), start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Physical constraint row {number} must be an object")
        predictor_type = str(rule.get("predictor_type", "")).lower()
        if predictor_type not in {"appdomain", "inputparam"}:
            raise ValueError(
                f"Physical constraint row {number}: predictor_type must be "
                "'appdomain' or 'inputparam'"
            )
        predictor_name = str(rule.get("predictor", ""))
        response_type = str(rule.get("response_type", "observable")).lower()
        response_name = str(rule.get("response", rule.get("observable", "")))
        canonical_predictor = resolve_label(predictor_name, predictor_type)
        if canonical_predictor is None:
            raise ValueError(
                f"Physical constraint row {number} references unknown "
                f"{predictor_type} variable {predictor_name!r}; available: "
                f"{label_sets[predictor_type]}"
            )
        if response_type not in {"appdomain", "inputparam", "observable"}:
            raise ValueError(
                f"Physical constraint row {number}: response_type must be "
                "'appdomain', 'inputparam', or 'observable'"
            )
        canonical_response = resolve_label(response_name, response_type)
        if canonical_response is None:
            raise ValueError(
                f"Physical constraint row {number} references unknown "
                f"{response_type} response {response_name!r}; available: "
                f"{label_sets[response_type]}"
            )
        if predictor_type == response_type and canonical_predictor == canonical_response:
            raise ValueError(
                f"Physical constraint row {number} cannot relate a variable to itself"
            )
        direction = str(rule.get("direction", "decreasing")).lower()
        if direction not in {"increasing", "decreasing"}:
            raise ValueError(
                f"Physical constraint row {number} direction must be "
                "'increasing' or 'decreasing'"
            )
        rules.append({
            "predictor_type": predictor_type,
            "predictor_index": label_sets[predictor_type].index(canonical_predictor),
            "response_type": response_type,
            "response_index": label_sets[response_type].index(canonical_response),
            "direction": direction,
        })

    parameter_transforms = parameter_transforms or {}
    observable_transforms = observable_transforms or {}
    parameter_sign_rules = []
    seen_sign_parameters = set()
    for number, rule in enumerate(config.get("parameter_sign_constraints", []), start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Parameter sign constraint row {number} must be an object")
        parameter_name = str(rule.get("parameter", ""))
        canonical_parameter = resolve_label(parameter_name, "inputparam")
        if canonical_parameter is None:
            raise ValueError(
                f"Parameter sign constraint row {number} references unknown input "
                f"parameter {parameter_name!r}; available: {label_sets['inputparam']}"
            )
        if canonical_parameter in seen_sign_parameters:
            raise ValueError(
                f"Input parameter {canonical_parameter!r} has more than one sign constraint"
            )
        seen_sign_parameters.add(canonical_parameter)
        sign = str(rule.get("sign", "")).lower()
        if sign not in {"positive", "negative"}:
            raise ValueError(
                f"Parameter sign constraint row {number} sign must be "
                "'positive' or 'negative'"
            )
        transform = parameter_transforms.get(canonical_parameter)
        if transform is None:
            raise ValueError(
                f"No normalization information is available for parameter "
                f"sign constraint {parameter_name!r}"
            )
        offset, scale = float(transform["offset"]), float(transform["scale"])
        if not np.isfinite(offset) or not np.isfinite(scale) or scale == 0:
            raise ValueError(
                f"Parameter {parameter_name!r} has invalid normalization information"
            )
        parameter_sign_rules.append({
            "parameter_index": label_sets["inputparam"].index(canonical_parameter),
            "parameter_name": canonical_parameter,
            "sign": sign,
            "offset": offset,
            "scale": scale,
        })

    boundary_rules = []
    for number, rule in enumerate(config.get("boundary_conditions", []), start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Boundary constraint row {number} must be an object")
        domain_name = str(rule.get("domain", ""))
        canonical_domain = resolve_label(domain_name, "appdomain")
        if canonical_domain is None:
            raise ValueError(
                f"Boundary constraint row {number} references unknown application "
                f"domain {domain_name!r}; available: {label_sets['appdomain']}"
            )
        endpoint = str(rule.get("endpoint", "")).lower()
        endpoint = {"min": "minimum", "max": "maximum"}.get(endpoint, endpoint)
        if endpoint not in {"minimum", "maximum"}:
            raise ValueError(
                f"Boundary constraint row {number} endpoint must be 'minimum' or 'maximum'"
            )
        target_type = str(rule.get("target_type", "")).lower()
        if target_type not in {"inputparam", "observable"}:
            raise ValueError(
                f"Boundary constraint row {number} target_type must be "
                "'inputparam' or 'observable'"
            )
        target_name = str(rule.get("target", ""))
        canonical_target = resolve_label(target_name, target_type)
        if canonical_target is None:
            raise ValueError(
                f"Boundary constraint row {number} references unknown {target_type} "
                f"target {target_name!r}; available: {label_sets[target_type]}"
            )
        try:
            target_value = float(rule["value"])
            tolerance = float(rule.get("tolerance", 0.01))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Boundary constraint row {number} needs numeric value and tolerance"
            ) from exc
        if not np.isfinite(target_value) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                f"Boundary constraint row {number} value must be finite and "
                "tolerance must be finite and greater than zero"
            )
        transform = (
            parameter_transforms if target_type == "inputparam"
            else observable_transforms
        ).get(canonical_target)
        if transform is None:
            raise ValueError(
                f"No normalization information is available for boundary target "
                f"{target_name!r}"
            )
        offset, scale = float(transform["offset"]), float(transform["scale"])
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(
                f"Boundary target {target_name!r} has a zero or invalid normalization scale"
            )
        boundary_rules.append({
            "domain_index": label_sets["appdomain"].index(canonical_domain),
            "endpoint": endpoint,
            "target_type": target_type,
            "target_index": label_sets[target_type].index(canonical_target),
            "target_name": canonical_target,
            "target_value": (target_value - offset) / scale,
            "tolerance": tolerance / abs(scale),
        })
    if config.get("enabled", False) and not rules and not boundary_rules and not parameter_sign_rules:
        raise ValueError("Physical constraints are enabled but no rows are configured")
    if mode == "soft" and (rules or boundary_rules) and strength <= 0:
        raise ValueError("Soft physical constraints require strength greater than zero")
    return {
        "enabled": True, "mode": mode, "strength": strength,
        "rules": rules, "boundary_rules": boundary_rules,
        "parameter_sign_rules": parameter_sign_rules,
    }


def _predict_full(gp_eta, x, theta, delta_eta=None):
    values = np.asarray(gp_eta.predict(np.hstack((x, theta))), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if delta_eta is not None:
        values = values + np.asarray(delta_eta, dtype=float)[:, None]
    return values


def _interpolate_trajectory(x_obs, theta_star, delta_eta, domain_index):
    """Interpolate the sampled calibrated state over one application domain."""
    coordinate = x_obs[:, domain_index]
    order = np.argsort(coordinate, kind="stable")
    ordered_x = x_obs[order]
    ordered_theta = theta_star[order]
    ordered_delta = np.asarray(delta_eta, dtype=float)[order]
    unique_x, inverse = np.unique(ordered_x[:, domain_index], return_inverse=True)
    if len(unique_x) == len(ordered_x):
        x_knots, theta_knots, delta_knots = ordered_x, ordered_theta, ordered_delta
    else:
        counts = np.bincount(inverse).astype(float)
        x_knots = np.zeros((len(unique_x), x_obs.shape[1]))
        theta_knots = np.zeros((len(unique_x), theta_star.shape[1]))
        delta_knots = np.zeros(len(unique_x))
        for column in range(x_obs.shape[1]):
            x_knots[:, column] = np.bincount(
                inverse, weights=ordered_x[:, column]
            ) / counts
        for column in range(theta_star.shape[1]):
            theta_knots[:, column] = np.bincount(
                inverse, weights=ordered_theta[:, column]
            ) / counts
        delta_knots = np.bincount(inverse, weights=ordered_delta) / counts

    grid_size = max(len(unique_x), (len(unique_x) - 1) * 16 + 1)
    grid_coordinate = np.linspace(unique_x[0], unique_x[-1], grid_size)
    x_grid = np.column_stack([
        np.interp(grid_coordinate, unique_x, x_knots[:, column])
        for column in range(x_obs.shape[1])
    ])
    theta_grid = np.column_stack([
        np.interp(grid_coordinate, unique_x, theta_knots[:, column])
        for column in range(theta_star.shape[1])
    ])
    delta_grid = np.interp(grid_coordinate, unique_x, delta_knots)
    return grid_coordinate, x_grid, theta_grid, delta_grid


def _boundary_values(gp_eta, x_obs, theta_star, rule, delta_eta=None):
    coordinate = x_obs[:, rule["domain_index"]]
    extreme = np.min(coordinate) if rule["endpoint"] == "minimum" else np.max(coordinate)
    selected = np.isclose(coordinate, extreme, rtol=0.0, atol=1e-10)
    if rule["target_type"] == "inputparam":
        return theta_star[selected, rule["target_index"]]
    inputs = np.hstack((x_obs[selected], theta_star[selected]))
    predictions = np.asarray(gp_eta.predict(inputs), dtype=float)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if delta_eta is not None:
        predictions = predictions + np.asarray(delta_eta)[selected, None]
    return predictions[:, rule["target_index"]]


def monotonicity_log_penalty(
    gp_eta,
    x_obs,
    theta_star,
    constraints,
    delta_eta=None,
    finite_difference_step=1e-4,
    hard_tolerance=1e-8,
):
    """Score monotonic and endpoint constraints on the calibrated prediction.

    For application-domain to observable rules, the sampled theta and
    delta_eta fields are interpolated between observation locations and the
    full prediction eta(x, theta(x)) + delta_eta(x) is checked on a dense grid.
    Inputs are the same normalized coordinates used by the fitted emulator.
    The soft penalty is a log-prior contribution; hard mode returns ``-inf``
    as soon as a rule is violated.
    """
    if not constraints or not constraints["enabled"]:
        return 0.0
    x_obs = np.asarray(x_obs, dtype=float)
    theta_star = np.asarray(theta_star, dtype=float)
    if theta_star.ndim == 1:
        theta_star = np.broadcast_to(theta_star, (len(x_obs), len(theta_star)))
    if theta_star.shape != (len(x_obs), theta_star.shape[-1]):
        raise ValueError("theta_star must contain one calibrated parameter vector per observation")
    if delta_eta is None:
        delta_eta = np.zeros(len(x_obs), dtype=float)
    else:
        delta_eta = np.asarray(delta_eta, dtype=float)
        if delta_eta.shape != (len(x_obs),):
            raise ValueError("delta_eta must contain one value per observation")

    penalty = 0.0
    for rule in constraints["rules"]:
        if rule["response_type"] != "observable":
            if rule["predictor_type"] == "appdomain":
                predictor_values = x_obs[:, rule["predictor_index"]]
            else:
                predictor_values = theta_star[:, rule["predictor_index"]]
            if rule["response_type"] == "appdomain":
                response_values = x_obs[:, rule["response_index"]]
            else:
                response_values = theta_star[:, rule["response_index"]]

            order = np.argsort(predictor_values, kind="stable")
            predictor_sorted = predictor_values[order]
            response_sorted = response_values[order]
            predictor_steps = np.diff(predictor_sorted)
            response_steps = np.diff(response_sorted)
            usable = predictor_steps > 1e-12
            if not np.any(usable):
                continue
            slopes = response_steps[usable] / predictor_steps[usable]
            signed_slopes = (
                slopes
                if rule["direction"] == "increasing"
                else -slopes
            )
            violations = np.maximum(-signed_slopes, 0.0)
            if constraints["mode"] == "hard":
                if np.any(violations > hard_tolerance):
                    return -np.inf
            else:
                response_scale = max(
                    float(np.std(response_values)), 1e-8
                )
                predictor_scale = max(
                    float(np.ptp(predictor_values)), 1e-8
                )
                normalized_violations = violations * predictor_scale / response_scale
                penalty += constraints["strength"] * float(
                    np.mean(normalized_violations**2)
                )
            continue

        plus_x = x_obs.copy()
        minus_x = x_obs.copy()
        plus_theta = theta_star.copy()
        minus_theta = theta_star.copy()
        if rule["predictor_type"] == "appdomain":
            coordinate = rule["predictor_index"]
            plus_x[:, coordinate] += finite_difference_step
            minus_x[:, coordinate] -= finite_difference_step
        else:
            coordinate = rule["predictor_index"]
            plus_theta[:, coordinate] += finite_difference_step
            minus_theta[:, coordinate] -= finite_difference_step

        plus_inputs = np.hstack((plus_x, plus_theta))
        minus_inputs = np.hstack((minus_x, minus_theta))
        response = rule["response_index"]
        plus_mean = np.asarray(gp_eta.predict(plus_inputs), dtype=float)
        minus_mean = np.asarray(gp_eta.predict(minus_inputs), dtype=float)
        if plus_mean.ndim == 1:
            plus_mean = plus_mean[:, None]
            minus_mean = minus_mean[:, None]
        if response >= plus_mean.shape[1]:
            raise ValueError(
                f"Configured observable index {response} is unavailable from the emulator"
            )
        if rule["predictor_type"] == "appdomain":
            domain_index = rule["predictor_index"]
            grid, x_grid, theta_grid, delta_grid = _interpolate_trajectory(
                x_obs, theta_star, delta_eta, domain_index
            )
            full_mean = _predict_full(gp_eta, x_grid, theta_grid, delta_grid)
            response_steps = np.diff(full_mean[:, response])
            predictor_steps = np.diff(grid)
            derivative = response_steps / predictor_steps
        else:
            # For a parameter-predictor rule, perturb that calibrated
            # parameter while holding x and the other parameters fixed.
            derivative = (
                plus_mean[:, response] - minus_mean[:, response]
            ) / (2.0 * finite_difference_step)
        signed_derivative = derivative if rule["direction"] == "increasing" else -derivative
        violations = np.maximum(-signed_derivative, 0.0)
        if constraints["mode"] == "hard":
            if np.any(violations > hard_tolerance):
                return -np.inf
        else:
            penalty += constraints["strength"] * float(np.mean(violations**2))
    for rule in constraints.get("boundary_rules", []):
        values = _boundary_values(gp_eta, x_obs, theta_star, rule, delta_eta)
        scaled_error = (values - rule["target_value"]) / rule["tolerance"]
        if constraints["mode"] == "hard":
            if np.any(np.abs(scaled_error) > 1.0):
                return -np.inf
        else:
            penalty += constraints["strength"] * float(np.mean(scaled_error**2))
    for rule in constraints.get("parameter_sign_rules", []):
        physical_values = (
            theta_star[:, rule["parameter_index"]] * rule["scale"]
            + rule["offset"]
        )
        signed_values = (
            physical_values if rule["sign"] == "positive" else -physical_values
        )
        violations = np.maximum(-signed_values, 0.0)
        if constraints["mode"] == "hard":
            if np.any(violations > hard_tolerance * abs(rule["scale"])):
                return -np.inf
        else:
            normalized_violations = violations / max(abs(rule["scale"]), 1e-12)
            penalty += constraints["strength"] * float(
                np.mean(normalized_violations**2)
            )
    return -penalty


def monotonicity_margins(gp_eta, x_obs, theta_star, constraints, delta_eta=None):
    """Return per-rule derivative/slope margins; non-negative is feasible."""
    if not constraints or not constraints["enabled"]:
        return []
    x_obs = np.asarray(x_obs, dtype=float)
    theta_star = np.asarray(theta_star, dtype=float)
    if theta_star.ndim == 1:
        theta_star = np.broadcast_to(theta_star, (len(x_obs), len(theta_star)))
    if delta_eta is None:
        delta_eta = np.zeros(len(x_obs), dtype=float)
    else:
        delta_eta = np.asarray(delta_eta, dtype=float)
    margins = []
    step = 1e-4
    for rule in constraints["rules"]:
        if rule["response_type"] != "observable":
            if rule["predictor_type"] == "appdomain":
                predictor_values = x_obs[:, rule["predictor_index"]]
            else:
                predictor_values = theta_star[:, rule["predictor_index"]]
            if rule["response_type"] == "appdomain":
                response_values = x_obs[:, rule["response_index"]]
            else:
                response_values = theta_star[:, rule["response_index"]]
            order = np.argsort(predictor_values, kind="stable")
            predictor_steps = np.diff(predictor_values[order])
            response_steps = np.diff(response_values[order])
            usable = predictor_steps > 1e-12
            slopes = np.zeros_like(predictor_steps)
            slopes[usable] = response_steps[usable] / predictor_steps[usable]
            signed_slopes = (
                slopes
                if rule["direction"] == "increasing"
                else -slopes
            )
            margins.append(np.where(usable, signed_slopes, 1.0))
            continue

        plus_x = x_obs.copy()
        minus_x = x_obs.copy()
        plus_theta = theta_star.copy()
        minus_theta = theta_star.copy()
        if rule["predictor_type"] == "appdomain":
            coordinate = rule["predictor_index"]
            plus_x[:, coordinate] += step
            minus_x[:, coordinate] -= step
        else:
            coordinate = rule["predictor_index"]
            plus_theta[:, coordinate] += step
            minus_theta[:, coordinate] -= step
        plus_mean = np.asarray(
            gp_eta.predict(np.hstack((plus_x, plus_theta))), dtype=float
        )
        minus_mean = np.asarray(
            gp_eta.predict(np.hstack((minus_x, minus_theta))), dtype=float
        )
        if plus_mean.ndim == 1:
            plus_mean = plus_mean[:, None]
            minus_mean = minus_mean[:, None]
        response = rule["response_index"]
        if rule["predictor_type"] == "appdomain":
            grid, x_grid, theta_grid, delta_grid = _interpolate_trajectory(
                x_obs, theta_star, delta_eta, rule["predictor_index"]
            )
            full_mean = _predict_full(gp_eta, x_grid, theta_grid, delta_grid)
            derivative = np.diff(full_mean[:, response]) / np.diff(grid)
        else:
            derivative = (plus_mean[:, response] - minus_mean[:, response]) / (2 * step)
        margins.append(
            derivative if rule["direction"] == "increasing" else -derivative
        )
    for rule in constraints.get("boundary_rules", []):
        values = _boundary_values(gp_eta, x_obs, theta_star, rule, delta_eta)
        margins.append(rule["tolerance"] - np.abs(values - rule["target_value"]))
    for rule in constraints.get("parameter_sign_rules", []):
        physical_values = (
            theta_star[:, rule["parameter_index"]] * rule["scale"]
            + rule["offset"]
        )
        signed_values = (
            physical_values if rule["sign"] == "positive" else -physical_values
        )
        margins.append(signed_values)
    return margins


def initialize_hard_constraint_state(
    initial_fields,
    theta_star_from_fields,
    gp_eta,
    x_obs,
    constraints,
    delta_eta=None,
    *,
    random_seed=1729,
    n_starts=5,
    max_iterations=500,
):
    """Find a feasible starting state before hard-constrained MCMC begins.

    A hard-constrained sampler cannot move out of an infeasible initial state
    when its proposal scale is small. This deterministic multistart search
    minimizes monotonicity violations, then returns a feasible state with a
    preference for smaller corrections.
    """
    fields = np.asarray(initial_fields, dtype=float)
    if constraints["mode"] != "hard":
        return fields

    def is_feasible(candidate):
        score = monotonicity_log_penalty(
            gp_eta,
            x_obs,
            theta_star_from_fields(candidate),
            constraints,
            delta_eta,
        )
        return np.isfinite(score)

    if is_feasible(fields):
        return fields

    soft_constraints = dict(constraints)
    soft_constraints["mode"] = "soft"
    soft_constraints["strength"] = 1.0
    field_shape = fields.shape

    def objective(flat):
        candidate = np.asarray(flat, dtype=float).reshape(field_shape)
        score = monotonicity_log_penalty(
            gp_eta,
            x_obs,
            theta_star_from_fields(candidate),
            soft_constraints,
            delta_eta,
        )
        violation = -float(score)
        return 1e4 * violation + 1e-6 * float(np.sum(candidate**2))

    rng = np.random.default_rng(random_seed)
    starts = [fields.ravel()]
    starts.extend(
        (fields + rng.normal(0.0, scale, size=field_shape)).ravel()
        for scale in (0.1, 0.5, 1.0, 2.0)[: max(0, n_starts - 1)]
    )
    feasible_candidates = []
    best_violation = np.inf

    def margins(flat):
        candidate = np.asarray(flat, dtype=float).reshape(field_shape)
        return np.concatenate(monotonicity_margins(
            gp_eta,
            x_obs,
            theta_star_from_fields(candidate),
            constraints,
            delta_eta,
        ))

    constrained_result = minimize(
        lambda flat: 0.5 * float(np.dot(flat, flat)),
        starts[0],
        method="SLSQP",
        constraints=[{
            "type": "ineq",
            "fun": lambda flat: margins(flat) + 1e-8,
        }],
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    constrained_candidate = constrained_result.x.reshape(field_shape)
    if is_feasible(constrained_candidate):
        feasible_candidates.append(constrained_candidate)

    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        candidate = result.x.reshape(field_shape)
        best_violation = min(
            best_violation,
            float(objective(result.x) - 1e-6 * np.sum(candidate**2)) / 1e4,
        )
        if is_feasible(candidate):
            feasible_candidates.append(candidate)

    if not feasible_candidates:
        raise RuntimeError(
            "Could not find a feasible initial state for the configured hard "
            "physical constraints. The constraint set may be incompatible with "
            "the emulator over the observation locations; try soft mode or "
            "review the selected variables and directions. "
            f"Best scaled violation score was {best_violation:.6g}."
        )
    return min(feasible_candidates, key=lambda candidate: float(np.sum(candidate**2)))


