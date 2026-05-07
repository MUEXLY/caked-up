import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# -----------------------------
# Formatting helpers
# -----------------------------
def format_prior(prior_dict):
    """Compact formatting for prior dictionaries."""
    try:
        ell = prior_dict.get("ell", "NA")
        var = prior_dict.get("var", "NA")
        return f"(ℓ={ell}, σ²={var})"
    except Exception:
        return "N/A"

def load_loss(results_dir):
    """Load cross-validation loss if available."""
    loss_path = os.path.join(results_dir, "cross_validation_losses.json")

    if not os.path.exists(loss_path):
        return None

    try:
        with open(loss_path, "r") as f:
            data = json.load(f)
        return data.get("net_loss", None)
    except Exception:
        return None

def extract_label(config, config_dir, text_keys, loss=None):
    """Extract formatted label text from config."""
    lines = [config_dir]

    for key_tuple in text_keys:
        val = config
        try:
            for k in key_tuple:
                val = val[k]

            if key_tuple[-1] == "kappa_prior_vals":
                lines.append(f"κ: {format_prior(val)}")
            elif key_tuple[-1] == "delta_eta_prior_vals":
                lines.append(f"η: {format_prior(val)}")
            else:
                lines.append(f"{key_tuple[-1]}: {val}")

        except KeyError:
            lines.append(f"{key_tuple[-1]}: N/A")

    if loss is not None:
        lines.append(f"Loss: {loss:.3f}")
    else:
        lines.append("Loss: N/A")

    return "\n".join(lines)


# -----------------------------
# Core plotting function
# -----------------------------
def create_figure_grid(
    config_dirs,
    base_directory,
    figure_rel_path,
    config_filename,
    text_keys,
    ncols=4,
    figsize=(12, 10),
    dpi=100,
    output_path="grid.png"
):
    """Create a single grid image from a subset of config directories."""

    # -----------------------------
# First pass: collect valid entries
# -----------------------------
    losses = []
    valid_entries = []

    for config_dir in config_dirs:
        full_dir = os.path.join(base_directory, config_dir)

        fig_path = os.path.join(full_dir, figure_rel_path)
        config_path = os.path.join(full_dir, config_filename)

        if not os.path.exists(fig_path) or not os.path.exists(config_path):
            continue

        loss = load_loss(full_dir)

        valid_entries.append((config_dir, full_dir, fig_path, config_path, loss))
        losses.append(loss if loss is not None else float("inf"))

    # -----------------------------
    # NOW compute grid size
    # -----------------------------
    n_images = len(valid_entries)

    if n_images == 0:
        print("No valid images to plot.")
        return

    nrows = math.ceil(n_images / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Robust flatten
    if isinstance(axes, (list, tuple)):
        axes = list(axes)
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Initialize plot index
    plot_idx = 0

    # Determine best (lowest loss)
    best_idx = None
    if len(losses) > 0:
        best_idx = int(min(range(len(losses)), key=lambda i: losses[i]))

    # -----------------------------
    # Plotting loop (ALWAYS runs)
    # -----------------------------
    for i, (config_dir, full_dir, fig_path, config_path, loss) in enumerate(valid_entries):
        try:
            if plot_idx >= len(axes):
                print("Warning: more images than axes, skipping remaining.")
                break

            img = mpimg.imread(fig_path)[::2, ::2]

            with open(config_path, "r") as f:
                config = json.load(f)

            label_text = extract_label(config, config_dir, text_keys, loss=loss)

            ax = axes[plot_idx]
            ax.imshow(img)
            ax.axis("off")

            ax.set_title(label_text, fontsize=6)

            # Highlight best
            if best_idx is not None and i == best_idx:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)

            plot_idx += 1

        except Exception as e:
            print(f"Error processing {config_dir}: {e}")
            continue

    # Turn off unused axes
    for j in range(plot_idx, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Saved grid figure to {output_path}")


# -----------------------------
# Chunking wrapper
# -----------------------------
def create_chunked_grids(
    base_directory,
    figure_rel_path="figures/discrepancy_diagnostics_realspace.png",
    config_filename="used_config.json",
    text_keys=[
        ("calibration_settings", "kappa_prior_vals"),
        ("calibration_settings", "delta_eta_prior_vals"),
    ],
    chunk_size=16,
    ncols=4,
    figsize=(12, 10),
    dpi=100,
):
    """Split configs into multiple grids for readability and memory safety."""

    config_dirs = sorted([
    d for d in os.listdir(base_directory)
    if os.path.isdir(os.path.join(base_directory, d))])

    config_dirs = sorted(
    config_dirs,
    key=lambda d: load_loss(os.path.join(base_directory, d)) or float("inf"))

    if len(config_dirs) == 0:
        print("No config directories found.")
        return

    print(f"Found {len(config_dirs)} configs")

    # OPTIONAL: sort by metric (customize later)
    # config_dirs = sorted(config_dirs, key=lambda d: some_metric(d))

    for i in range(0, len(config_dirs), chunk_size):
        subset = config_dirs[i:i + chunk_size]

        output_path = os.path.join(
            base_directory,
            f"grid_{i // chunk_size:03d}.png"
        )

        print(f"Processing chunk {i // chunk_size}: {len(subset)} configs")

        create_figure_grid(
            config_dirs=subset,
            base_directory=base_directory,
            figure_rel_path=figure_rel_path,
            config_filename=config_filename,
            text_keys=text_keys,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            output_path=output_path
        )


# -----------------------------
# Entry point
# -----------------------------
def main():
    base_directory = "./results/sweep/"

    create_chunked_grids(
        base_directory=base_directory,
        figure_rel_path="figures/posterior_predictive_check_physical.png",
        config_filename="used_config.json",
        text_keys=[
            ("calibration_settings", "kappa_prior_vals"),
            ("calibration_settings", "delta_eta_prior_vals"),
        ],
        chunk_size=16,     # 🔑 key parameter
        ncols=4,
        figsize=(12, 10),
        dpi=100
    )


if __name__ == "__main__":
    main()