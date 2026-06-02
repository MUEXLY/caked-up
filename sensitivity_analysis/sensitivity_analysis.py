import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Parsing + loading
# --------------------------------------------------

def parse_dirname(dirname):
    pattern = r"kvar([0-9.]+)_kell([0-9.]+)_evar([0-9.]+)_eell([0-9.]+)"
    m = re.match(pattern, dirname)
    if not m:
        return None
    return {
        "kvar": float(m.group(1)),
        "kell": float(m.group(2)),
        "evar": float(m.group(3)),
        "eell": float(m.group(4)),
        "name": dirname
    }


def load_results(base_path, dirname):
    path = os.path.join(base_path, dirname, "results_physical.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def extract_x(results):
    for k in results:
        if k.startswith("x_obs"):
            return results[k]
    raise ValueError("x not found")


def extract_kappas(results):
    k = 0
    out = []
    while f"kappa_{k}_mean" in results:
        out.append(k)
        k += 1
    return out


# --------------------------------------------------
# Plot helpers
# --------------------------------------------------

def compute_global_limits(results_list, ref_results):
    all_vals = []

    ref_delta = ref_results["delta_eta_mean"]

    for res in results_list:
        all_vals.append(res["delta_eta_mean"] - ref_delta)

        for k in extract_kappas(res):
            all_vals.append(
                res[f"kappa_{k}_mean"] - ref_results[f"kappa_{k}_mean"]
            )

    all_vals = np.concatenate(all_vals)
    lim = np.max(np.abs(all_vals))
    return -lim, lim


def plot_panel(ax, res, ref_res, ylim):
    x = extract_x(res)

    lines = []
    labels = []

    # --- δ_eta ---
    delta = res["delta_eta_mean"] - ref_res["delta_eta_mean"]
    line_delta, = ax.plot(x, delta, linewidth=2)
    lines.append(line_delta)
    labels.append(r"$\delta_\eta(x)$")

    # --- κ overlays ---
    kappas = extract_kappas(res)
    for k in kappas:
        kdiff = res[f"kappa_{k}_mean"] - ref_res[f"kappa_{k}_mean"]
        line_k, = ax.plot(
            x,
            kdiff,
            linestyle="--",
            alpha=0.8
        )
        lines.append(line_k)
        labels.append(rf"$\kappa_{{{k+1}}}(x)$")

    ax.axhline(0, linestyle="--", color="black", linewidth=0.8)

    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(alpha=0.2)

    return lines, labels


# --------------------------------------------------
# Main figure builder
# --------------------------------------------------

def create_2D_sensitivity_figure(base_path, reference_case, prior_prefix, legend=True):

    var_key = f"{prior_prefix}var"
    ell_key = f"{prior_prefix}ell"

    # --- parse dirs ---
    dirs = [
        parse_dirname(d) for d in os.listdir(base_path)
        if parse_dirname(d)
    ]

    # --- filter ---
    filtered = []
    for d in dirs:
        if all(
            abs(d[k] - v) < 1e-8
            for k, v in reference_case.items()
            if k not in [var_key, ell_key]
        ):
            filtered.append(d)

    # --- unique values ---
    var_vals = sorted(set(d[var_key] for d in filtered))
    ell_vals = sorted(set(d[ell_key] for d in filtered))

    # --- load results ---
    results_map = {}
    for d in filtered:
        key = (d[ell_key], d[var_key])
        results_map[key] = load_results(base_path, d["name"])

    ref_key = (reference_case[ell_key], reference_case[var_key])
    ref_res = results_map[ref_key]

    ylim = compute_global_limits(list(results_map.values()), ref_res)

    # --------------------------------------------------
    # FIGURE
    # --------------------------------------------------

    fig, axes = plt.subplots(
    len(ell_vals),
    len(var_vals),
    figsize=(3 * len(var_vals), 3 * len(ell_vals)),
    constrained_layout=False
)

    # reserve room for titles/subtitles
    fig.subplots_adjust(
        top=0.83,
        left=0.12,
        bottom=0.12,
        right=0.88,
        wspace=0.03,
        hspace=0.05
    )

    if len(ell_vals) == 1:
        axes = axes.reshape(1, -1)

    legend_lines = None
    legend_labels = None

    for i, ell in enumerate(ell_vals):
        for j, var in enumerate(var_vals):

            ax = axes[i, j]
            res = results_map.get((ell, var), None)

            if res is None:
                ax.axis("off")
                continue

            lines, labels = plot_panel(ax, res, ref_res, ylim)

            if legend_lines is None:
                legend_lines = lines
                legend_labels = labels

            is_ref = (ell == ref_key[0]) and (var == ref_key[1])

            # --- edge labeling ---
            if i == 0:
                ax.set_title(rf"$\sigma={var}$", fontsize=20)

            if j == 0:
                ax.set_ylabel(rf"$\ell={ell}$", fontsize=20)

            # --- highlight reference row/col ---
            if ell == ref_key[0] or var == ref_key[1]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            # --- reference annotation ---
            if is_ref:
                ax.text(
                    0.5, 0.5,
                    "Reference",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=20,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )

    #--------------------------------------------------
    # Global axis labels
    # --------------------------------------------------

    if prior_prefix == "e":

        fig.supxlabel(
            r"Increasing $\sigma_{\delta_\eta}$",
            fontsize=28
        )

        fig.supylabel(
            r"Increasing $\ell_{\delta_\eta}$",
            fontsize=28
        )

    elif prior_prefix == "k":

        fig.supxlabel(
            r"Increasing $\sigma_{\kappa}$",
            fontsize=28
        )

        fig.supylabel(
            r"Increasing $\ell_{\kappa}$",
            fontsize=28
        )

    # --------------------------------------------------
    # Suptitle hierarchy
    # --------------------------------------------------

    title_map = {
        "k": r"Sensitivity of discrepancy posteriors to $\phi_{\kappa}$",
        "e": r"Sensitivity of discrepancy posteriors to $\phi_{\delta_\eta}$"
    }

    fig.suptitle(
        title_map[prior_prefix],
        fontsize=28,
        y=0.96
    )

    # smaller subtitle
    fig.text(
        0.5,
        0.88,
        "Deviation from reference",
        ha="center",
        fontsize=20
    )

    # --- single legend ---
    if legend:
        fig.legend(
            legend_lines,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=20,
            frameon=False
        )

        # plt.show()
    plt.savefig(f"{base_path}/sensitivity_{prior_prefix}.pdf", bbox_inches="tight")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # read config
    with open("sensitivity_analysis/config_sensitivity.json", "r") as f:
        config = json.load(f)

    base_path = config["base_path"]
    reference_case = config["reference_case"]
    legend = config["legend"]


    # κ prior
    create_2D_sensitivity_figure(
        base_path,
        reference_case,
        prior_prefix="k",
        legend=legend
    )

    # δ_eta prior
    create_2D_sensitivity_figure(
        base_path,
        reference_case,
        prior_prefix="e",
        legend=legend
    )


if __name__ == "__main__":
    main()
