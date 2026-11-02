"""State-level monotonicity constraints for the calibration emulator."""

import numpy as np
from scipy.optimize import minimize


def compile_directional_constraints(
    config,
    domain_labels,
    parameter_labels,
    observable_labels,
):
    """Resolve configured predictor/response names to model column indices."""
    config = config or {}
    if not config.get("enabled", False):
        return {"enabled": False, "mode": "soft", "strength": 0.0, "rules": []}

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
        if predictor_name not in label_sets[predictor_type]:
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
        if response_name not in label_sets[response_type]:
            raise ValueError(
                f"Physical constraint row {number} references unknown "
                f"{response_type} response {response_name!r}; available: "
                f"{label_sets[response_type]}"
            )
        if predictor_type == response_type and predictor_name == response_name:
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
            "predictor_index": label_sets[predictor_type].index(predictor_name),
            "response_type": response_type,
            "response_index": label_sets[response_type].index(response_name),
            "direction": direction,
        })
    if config.get("enabled", False) and not rules:
        raise ValueError("Physical constraints are enabled but no rows are configured")
    if mode == "soft" and rules and strength <= 0:
        raise ValueError("Soft physical constraints require strength greater than zero")
    return {"enabled": True, "mode": mode, "strength": strength, "rules": rules}


def monotonicity_log_penalty(
    gp_eta,
    x_obs,
    theta_star,
    constraints,
    finite_difference_step=1e-4,
    hard_tolerance=1e-8,
):
    """Score local monotonic relationships at the observation states.

    Inputs are the same normalized coordinates used by the fitted emulator.
    Positive affine normalization preserves derivative direction. The soft
    penalty is a log-prior contribution; hard mode returns ``-inf`` as soon as
    a rule is violated.
    """
    if not constraints or not constraints["enabled"]:
        return 0.0
    x_obs = np.asarray(x_obs, dtype=float)
    theta_star = np.asarray(theta_star, dtype=float)
    if theta_star.ndim == 1:
        theta_star = np.broadcast_to(theta_star, (len(x_obs), len(theta_star)))
    if theta_star.shape != (len(x_obs), theta_star.shape[-1]):
        raise ValueError("theta_star must contain one calibrated parameter vector per observation")

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
    return -penalty


def monotonicity_margins(gp_eta, x_obs, theta_star, constraints):
    """Return per-rule derivative/slope margins; non-negative is feasible."""
    if not constraints or not constraints["enabled"]:
        return []
    x_obs = np.asarray(x_obs, dtype=float)
    theta_star = np.asarray(theta_star, dtype=float)
    if theta_star.ndim == 1:
        theta_star = np.broadcast_to(theta_star, (len(x_obs), len(theta_star)))
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
        derivative = (plus_mean[:, response] - minus_mean[:, response]) / (2 * step)
        margins.append(
            derivative if rule["direction"] == "increasing" else -derivative
        )
    return margins


def initialize_hard_constraint_state(
    initial_fields,
    theta_star_from_fields,
    gp_eta,
    x_obs,
    constraints,
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


