"""Desktop workflow GUI for the CAKED-UP and integrated-delta calibrators."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import numpy as np
import csv
import pandas as pd
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_REPOSITORIES = {
    "CAKED-UP": APP_ROOT,
    "IB Framework": APP_ROOT / "frameworks" / "integrated_delta",
}
METHOD_LOGOS = {
    "CAKED-UP": APP_ROOT / "logos" / "ibfu_logo.png",
    "IB Framework": APP_ROOT / "logos" / "ib_logo.png",
}
COLORS = {
    "window": "#191b1f",
    "header": "#24272b",
    "panel": "#2c3035",
    "panel_alt": "#353a40",
    "border": "#4b5158",
    "text": "#f2f3f4",
    "muted": "#c0c5cb",
    "button": "#454b52",
    "button_active": "#555c63",
    "success": "#a8d6ae",
    "warning": "#e9c77e",
    "danger": "#ef9890",
    "log": "#101215",
    "button_inactive": "#34393f",
    "button_selected": "#555c63",
    "button_disabled": "#3d4248",
    "text_disabled": "#9ba1a7",
}
NAV_ITEMS = [
    ("import", "1  IMPORT DATA /\n    MODEL"),
    ("settings", "2  SETTINGS"),
    ("run", "3  RUN / MONITOR"),
    ("analyze", "4  ANALYZE"),
]


def _get_input_paths(config: dict, method: str) -> tuple[str, str]:
    settings = config.get("input_settings", {})
    if method == "CAKED-UP":
        return settings.get("model_data_path", ""), settings.get("obs_data_path", "")
    return settings.get("input_model_path", ""), settings.get("input_observations_path", "")


def _set_input_paths(
    config: dict,
    method: str,
    model_path: str,
    observation_path: str,
    repository: Path,
) -> None:
    settings = config.setdefault("input_settings", {})
    model_key, observation_key = (
        ("model_data_path", "obs_data_path")
        if method == "CAKED-UP"
        else ("input_model_path", "input_observations_path")
    )
    def absolute(path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = repository / candidate
        return candidate.resolve()

    settings[model_key] = str(absolute(model_path)) + os.sep
    settings[observation_key] = str(absolute(observation_path)) + os.sep


def _set_output_paths(config: dict, method: str, results_dir: Path) -> None:
    figures_dir = results_dir / "figures"
    output = config.setdefault("output_settings", {})
    output["live_monitor"] = True
    output["live_monitor_interval"] = 50
    output["live_monitor_path"] = str(results_dir.parent / "live_prediction.json")
    figure_options = output.setdefault("figure_options", {})
    figure_options.setdefault("hyperprior_density_animation", True)
    figure_options["live_hyperparameter_updates"] = False
    if method == "CAKED-UP":
        figure_options.setdefault("calibration_prediction_animation", True)
        output["results_path"] = str(results_dir)
        output["figures_path"] = str(figures_dir)
    else:
        output["figures_path"] = str(figures_dir)
        results_options = output.setdefault("results_options", {})
        results_options["results_path"] = str(results_dir)


def _resolve_runtime_paths(config: dict, method: str, repository: Path, config_source: Path) -> None:
    """Make input/config references stable after writing a run-local config."""
    input_settings = config.setdefault("input_settings", {})
    input_keys = (
        ("model_data_path", "obs_data_path")
        if method == "CAKED-UP"
        else ("input_model_path", "input_observations_path")
    )
    for key in input_keys:
        value = input_settings.get(key)
        if value and not Path(value).expanduser().is_absolute():
            input_settings[key] = str((repository / value).resolve()) + os.sep

    for key in ("plot_settings_path",):
        value = config.get(key)
        if value and not Path(value).expanduser().is_absolute():
            config[key] = str((config_source.parent / value).resolve())
    plot_settings = config.get("plot_settings")
    if isinstance(plot_settings, str) and not Path(plot_settings).expanduser().is_absolute():
        config["plot_settings"] = str((config_source.parent / plot_settings).resolve())

    cross_validation = config.get("cross_validation_settings", {})
    holdout_paths = cross_validation.get("holdout_data_paths", {})
    for key, value in holdout_paths.items():
        if value and not Path(value).expanduser().is_absolute():
            holdout_paths[key] = str((repository / value).resolve())

    validation = config.get("validation_options", config.get("output_settings", {}).get("validation_options", {}))
    external_validation = validation.get("external_validation", {})
    csv_path = external_validation.get("csv_path")
    if csv_path and not Path(csv_path).expanduser().is_absolute():
        external_validation["csv_path"] = str((config_source.parent / csv_path).resolve())


TABLE_FILES = {
    "model_domain": ("appDomain.txt", "Simulator domain"),
    "model_parameters": ("thetaVals.txt", "Simulator parameters"),
    "model_outputs": ("modelPredictions.txt", "Simulator outputs"),
    "observation_domain": ("appDomain.txt", "Observation domain"),
    "observations": ("observationData.txt", "Observations"),
}
MODEL_TABLE_KEYS = ("model_domain", "model_parameters", "model_outputs")
OBSERVATION_TABLE_KEYS = ("observation_domain", "observations")


def _read_input_table(path: str | Path) -> pd.DataFrame:
    """Read a headered table without requiring users to specify its delimiter."""
    source = Path(path).expanduser()
    if source.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source)
    else:
        if source.suffix.lower() in {".tsv", ".tab"}:
            delimiter = "\t"
        else:
            with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
                sample = stream.read(8192)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
            except csv.Error:
                delimiter = r"\s+"
        frame = pd.read_csv(
            source,
            sep=delimiter,
            engine="python",
            encoding="utf-8-sig",
            decimal="," if delimiter == ";" else ".",
        )

    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{source.name} contains no data columns.")
    if any(not str(column).strip() or str(column).startswith("Unnamed:") for column in frame.columns):
        raise ValueError(f"{source.name} needs a header row naming each column.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{source.name} contains duplicate column names.")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{source.name} contains missing or non-finite values.")
    return frame


def _validate_input_tables(tables: dict[str, str]) -> dict[str, pd.DataFrame]:
    missing = [key for key in TABLE_FILES if not tables.get(key)]
    if missing:
        labels = ", ".join(TABLE_FILES[key][1] for key in missing)
        raise ValueError(f"Select all five input tables. Missing: {labels}.")
    frames = {key: _read_input_table(tables[key]) for key in TABLE_FILES}

    model_rows = [len(frames[key]) for key in MODEL_TABLE_KEYS]
    if len(set(model_rows)) != 1:
        raise ValueError("Simulator domain, parameter, and output tables must have the same number of rows.")
    observation_rows = [len(frames[key]) for key in OBSERVATION_TABLE_KEYS]
    if len(set(observation_rows)) != 1:
        raise ValueError("Observation domain and observation values must have the same number of rows.")
    if frames["model_domain"].shape[1] != frames["observation_domain"].shape[1]:
        raise ValueError("Simulator and observation domain tables must have the same number of columns.")
    if frames["model_outputs"].shape[1] != frames["observations"].shape[1]:
        raise ValueError("Simulator output and observation tables must have the same number of columns.")
    return frames


def _stage_input_tables(config: dict, method: str, tables: dict[str, str], run_dir: Path) -> None:
    frames = _validate_input_tables(tables)
    model_dir = run_dir / "data" / "model"
    observation_dir = run_dir / "data" / "observations"
    model_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "model_domain": model_dir / "appDomain.txt",
        "model_parameters": model_dir / "thetaVals.txt",
        "model_outputs": model_dir / "modelPredictions.txt",
        "observation_domain": observation_dir / "appDomain.txt",
        "observations": observation_dir / "observationData.txt",
    }
    for key, frame in frames.items():
        frame.to_csv(output_names[key], sep="\t", index=False, float_format="%.17g")
    _set_input_paths(config, method, str(model_dir), str(observation_dir), run_dir)
    config.setdefault("input_settings", {})["input_delimiter"] = "\t"


def _add_tooltip(widget: tk.Widget, text: str) -> None:
    state = {"window": None, "job": None}

    def show(_event=None) -> None:
        if state["job"]:
            widget.after_cancel(state["job"])
        state["job"] = widget.after(450, display)

    def display() -> None:
        if not widget.winfo_exists():
            return
        x = widget.winfo_rootx() + 12
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        window = tk.Toplevel(widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        window.configure(bg=COLORS["border"])
        tk.Label(
            window,
            text=text,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            justify="left",
            wraplength=300,
            padx=10,
            pady=7,
            font=("Avenir Next", 9),
        ).pack(padx=1, pady=1)
        state["window"] = window

    def hide(_event=None) -> None:
        if state["job"]:
            try:
                widget.after_cancel(state["job"])
            except tk.TclError:
                pass
            state["job"] = None
        if state["window"] is not None:
            state["window"].destroy()
            state["window"] = None

    widget.bind("<Enter>", show, add="+")
    widget.bind("<Leave>", hide, add="+")


class _ColorButton(tk.Label):
    """A button-like label whose dark colors render consistently on macOS Tk."""

    def __init__(self, parent: tk.Widget, text: str, command, *, state: str = "normal"):
        self.command = command
        self.enabled = state != "disabled"
        self.normal_background = COLORS["button_inactive"]
        super().__init__(
            parent,
            text=text,
            bg=COLORS["button_disabled"] if not self.enabled else self.normal_background,
            fg=COLORS["text_disabled"] if not self.enabled else COLORS["text"],
            activebackground=COLORS["button_selected"],
            activeforeground=COLORS["text"],
            font=("Avenir Next", 10, "bold"),
            padx=13,
            pady=8,
            cursor="arrow" if not self.enabled else "hand2",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            anchor="center",
        )
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<ButtonPress-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")

    def _on_enter(self, _event=None) -> None:
        if self.enabled:
            self.configure(bg=COLORS["button_selected"])

    def _on_leave(self, _event=None) -> None:
        self.configure(bg=self.normal_background if self.enabled else COLORS["button_disabled"])

    def _on_press(self, _event=None) -> None:
        if self.enabled:
            self.configure(bg=COLORS["button_active"])

    def _on_release(self, event=None) -> None:
        if not self.enabled:
            return
        self.configure(bg=COLORS["button_selected"] if event is None or (0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()) else self.normal_background)
        if event is None or (0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()):
            self.command()

    def configure(self, cnf=None, **kwargs):
        state = kwargs.pop("state", None)
        result = super().configure(cnf, **kwargs)
        if state is not None:
            self.enabled = state != "disabled"
            super().configure(
                bg=self.normal_background if self.enabled else COLORS["button_disabled"],
                fg=COLORS["text"] if self.enabled else COLORS["text_disabled"],
                cursor="hand2" if self.enabled else "arrow",
            )
        return result

    config = configure


def _make_button(
    parent: tk.Widget,
    text: str,
    command,
    *,
    state: str = "normal",
    tooltip: str | None = None,
) -> _ColorButton:
    button = _ColorButton(parent, text, command, state=state)
    if tooltip:
        _add_tooltip(button, tooltip)
    return button


class CalibrationGUI(tk.Tk):
    """Small, local-first interface around the two existing Python frameworks."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Calibration Studio")
        self.geometry("1320x850")
        self.minsize(1050, 700)
        self.configure(bg=COLORS["window"])

        self.method = "CAKED-UP"
        self.repositories = dict(DEFAULT_REPOSITORIES)
        self.config_paths = {name: root / "config.json" for name, root in self.repositories.items()}
        self.configs: dict[str, dict] = {}
        self.plot_settings_by_method: dict[str, dict] = {}
        self.import_tables: dict[str, dict[str, str]] = {}
        self.import_table_paths: dict[str, dict[str, str]] = {}
        self.page = "import"
        self.current_results_dir: Path | None = None
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.log_history: list[str] = []
        self.image_reference = None
        self.gif_players: dict[tk.Widget, dict] = {}
        self._save_settings_callback = None
        self._build_shell()
        self._load_config(self.method, show_error=False)
        self._render_page()

    def _build_shell(self) -> None:
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(
            "Dark.TButton",
            font=("Avenir Next", 11),
            padding=(12, 8),
            background=COLORS["button_inactive"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            focuscolor=COLORS["button_inactive"],
            relief="flat",
        )
        self.style.configure(
            "Unavailable.TButton",
            font=("Avenir Next", 11),
            padding=(12, 8),
            background=COLORS["panel_alt"],
            foreground=COLORS["text_disabled"],
            bordercolor=COLORS["border"],
            focuscolor=COLORS["panel_alt"],
            relief="flat",
        )
        self.style.map(
            "Dark.TButton",
            background=[
                ("pressed", COLORS["button_selected"]),
                ("active", COLORS["button_active"]),
                ("focus", COLORS["button_inactive"]),
            ],
            foreground=[("pressed", COLORS["text"]), ("active", COLORS["text"])],
        )
        self.style.map(
            "Unavailable.TButton",
            background=[("disabled", COLORS["button_disabled"]), ("active", COLORS["button_disabled"]), ("focus", COLORS["button_disabled"])],
            foreground=[("disabled", COLORS["text_disabled"])],
        )
        for style_name, normal, active in (
            ("Method.TButton", COLORS["button_inactive"], COLORS["button_selected"]),
            ("SelectedMethod.TButton", COLORS["button_selected"], COLORS["button_selected"]),
            ("Nav.TButton", COLORS["header"], COLORS["button_inactive"]),
            ("SelectedNav.TButton", COLORS["button_selected"], COLORS["button_selected"]),
        ):
            self.style.configure(
                style_name,
                font=("Avenir Next", 10, "bold"),
                padding=(20, 19) if style_name in {"Nav.TButton", "SelectedNav.TButton"} else (18, 8),
                anchor="w" if style_name in {"Nav.TButton", "SelectedNav.TButton"} else "center",
                background=normal,
                foreground=COLORS["text"],
                bordercolor=COLORS["border"],
                focuscolor=normal,
                relief="flat",
            )
            self.style.map(
                style_name,
                background=[("pressed", active), ("active", active), ("focus", normal)],
                foreground=[("pressed", COLORS["text"]), ("active", COLORS["text"])],
            )
        self.style.configure("TLabel", font=("Avenir Next", 11), background=COLORS["panel"], foreground=COLORS["text"])
        self.style.configure("TEntry", padding=7, font=("Avenir Next", 11), fieldbackground=COLORS["panel_alt"], foreground=COLORS["text"])
        self.style.configure(
            "Dark.TCombobox",
            padding=6,
            fieldbackground=COLORS["panel_alt"],
            background=COLORS["button_inactive"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
        )
        self.style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", COLORS["panel_alt"]), ("focus", COLORS["panel_alt"])],
            foreground=[("readonly", COLORS["text"])],
            selectbackground=[("readonly", COLORS["button_selected"])],
            selectforeground=[("readonly", COLORS["text"])],
            background=[("active", COLORS["button_selected"]), ("!disabled", COLORS["button_inactive"])],
        )

        header = tk.Frame(self, bg=COLORS["header"], height=112, highlightbackground=COLORS["border"], highlightthickness=1)
        header.pack(fill="x", padx=20, pady=(18, 0))
        header.pack_propagate(False)

        brand = tk.Frame(header, bg="#000000", width=230, height=88)
        brand.pack(side="left", padx=18, pady=14)
        brand.pack_propagate(False)
        self.brand_image = None
        self.brand_image_label = tk.Label(brand, bg="#000000")
        self.brand_image_label.pack(fill="both", expand=True)
        self._set_method_logo()

        selector = tk.Frame(header, bg=COLORS["header"])
        selector.pack(side="left", padx=22, pady=16)
        tk.Label(selector, text="CALIBRATION METHOD", bg=COLORS["header"], fg=COLORS["muted"], font=("Avenir Next", 9, "bold")).pack(anchor="w", pady=(0, 7))
        switch = tk.Frame(selector, bg=COLORS["border"], padx=3, pady=3)
        switch.pack(anchor="w")
        self.method_buttons = {}
        for method, label in (("IB Framework", "IB"), ("CAKED-UP", "IBFU")):
            button = ttk.Button(
                switch,
                text=label,
                command=lambda value=method: self.set_method(value),
                style="Method.TButton",
                cursor="hand2",
            )
            button.pack(side="left")
            self.method_buttons[method] = button

        self.status_label = tk.Label(header, text="READY", bg=COLORS["header"], fg=COLORS["success"], font=("Avenir Next", 10, "bold"))
        self.status_label.pack(side="right", padx=24)

        body = tk.Frame(self, bg=COLORS["window"], highlightbackground=COLORS["border"], highlightthickness=1)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        self.sidebar = tk.Frame(body, bg=COLORS["header"], width=245)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="WORKFLOW", bg=COLORS["header"], fg=COLORS["muted"], font=("Avenir Next", 9, "bold")).pack(anchor="w", padx=20, pady=(20, 12))
        self.nav_buttons = {}
        for key, label in NAV_ITEMS:
            button = ttk.Button(
                self.sidebar,
                text=label,
                command=lambda value=key: self.show_page(value),
                style="Nav.TButton",
                cursor="hand2",
            )
            button.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[key] = button

        self.content = tk.Frame(body, bg=COLORS["window"], padx=28, pady=25)
        self.content.pack(side="left", fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_method_logo(self) -> None:
        logo_path = METHOD_LOGOS[self.method]
        try:
            logo = Image.open(logo_path).convert("RGB")
            logo.thumbnail((218, 84), Image.Resampling.LANCZOS)
            self.brand_image = ImageTk.PhotoImage(logo)
            self.brand_image_label.configure(image=self.brand_image)
        except (OSError, tk.TclError):
            self.brand_image = None
            self.brand_image_label.configure(
                image="",
                text=self.method.upper(),
                fg=COLORS["text"],
                font=("Avenir Next", 15, "bold"),
            )

    def set_method(self, method: str) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Calibration running", "Wait for the current calibration to finish before switching frameworks.")
            return
        if method == self.method or not self._confirm_leave_settings():
            return
        self.method = method
        if method not in self.configs:
            self._load_config(method, show_error=False)
        self._set_method_logo()
        self._render_page()

    def show_page(self, page: str) -> None:
        if page == self.page or not self._confirm_leave_settings():
            return
        self.page = page
        self._render_page()

    def _settings_have_unsaved_changes(self) -> bool:
        if self.page != "settings" or not hasattr(self, "settings_values"):
            return False
        for path, kind, variable in self.settings_values:
            try:
                raw = variable.get()
                if kind == "bool":
                    value = bool(raw)
                elif kind == "int":
                    value = int(raw)
                elif kind == "float":
                    value = float(raw)
                elif kind == "json":
                    value = json.loads(raw)
                elif kind == "none":
                    value = None if not raw.strip() else raw
                else:
                    value = raw
                original = self.config
                for part in path:
                    original = original[part]
                if value != original:
                    return True
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                return True
        return False

    def _confirm_leave_settings(self) -> bool:
        if not self._settings_have_unsaved_changes():
            return True
        choice = messagebox.askyesnocancel(
            "Unsaved settings",
            "You have unsaved changes. Save them before leaving this window?\n\n"
            "Yes: save a working config\nNo: discard the changes\nCancel: stay here",
            parent=self,
        )
        if choice is None:
            return False
        if choice:
            return bool(self._save_settings_callback and self._save_settings_callback())
        return True

    def _on_close(self) -> None:
        if self._confirm_leave_settings():
            self.destroy()

    def _load_config(self, method: str, show_error: bool = True) -> bool:
        path = self.config_paths[method]
        try:
            with path.open("r", encoding="utf-8") as stream:
                self.configs[method] = json.load(stream)
            config = self.configs[method]
            plot_settings = config.get("plot_settings", {})
            plot_path = config.get("plot_settings_path")
            if isinstance(plot_settings, str):
                plot_path = plot_settings
                plot_settings = {}
            if plot_path and not plot_settings:
                candidate = Path(plot_path).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if candidate.is_file():
                    with candidate.open("r", encoding="utf-8") as stream:
                        plot_settings = json.load(stream)
            if method == "IB Framework" and not plot_settings:
                plot_settings = config.get("label_settings", {})
            self.plot_settings_by_method[method] = json.loads(json.dumps(plot_settings))
            if method == "CAKED-UP" and plot_settings:
                config["plot_settings"] = json.loads(json.dumps(plot_settings))
            return True
        except Exception as exc:
            if show_error:
                messagebox.showerror("Could not load config", f"{path}\n\n{exc}")
            self.configs[method] = {}
            return False

    @property
    def config(self) -> dict:
        return self.configs.setdefault(self.method, {})

    @property
    def label_settings_key(self) -> str:
        return "plot_settings" if self.method == "CAKED-UP" else "label_settings"

    def _heading(self, title: str, subtitle: str) -> None:
        tk.Label(self.content, text=title, bg=COLORS["window"], fg=COLORS["text"], font=("Avenir Next", 23, "bold")).pack(anchor="w")
        tk.Label(self.content, text=subtitle, bg=COLORS["window"], fg=COLORS["muted"], font=("Avenir Next", 11)).pack(anchor="w", pady=(5, 20))

    def _field(
        self,
        parent: tk.Widget,
        label: str,
        value: str,
        browse: str | None = None,
        directory: bool = False,
        filetypes=None,
    ) -> tk.Entry:
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill="x", pady=7)
        tk.Label(row, text=label, width=22, anchor="w", bg=COLORS["panel"], fg=COLORS["text"], font=("Avenir Next", 10, "bold")).pack(side="left")
        entry = tk.Entry(row, relief="solid", borderwidth=1, font=("Avenir Next", 10), bg=COLORS["panel_alt"], fg=COLORS["text"], insertbackground=COLORS["text"])
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        if browse:
            def choose() -> None:
                selected = filedialog.askdirectory() if directory else filedialog.askopenfilename(filetypes=filetypes or [("JSON config", "*.json"), ("All files", "*.*")])
                if selected:
                    entry.delete(0, "end")
                    entry.insert(0, selected)
                    entry.focus_set()
            _make_button(row, browse, choose).pack(side="left", padx=(8, 0))
        return entry

    def _card(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, padx=18, pady=14)
        frame.pack(fill="x", pady=(0, 14))
        return frame

    def _render_page(self) -> None:
        for widget in list(self.gif_players):
            self._stop_gif_animation(widget)
        for child in self.content.winfo_children():
            child.destroy()
        for method, button in self.method_buttons.items():
            active = method == self.method
            button.configure(
                style="SelectedMethod.TButton" if active else "Method.TButton",
            )
        for key, button in self.nav_buttons.items():
            button.configure(
                style="SelectedNav.TButton" if key == self.page else "Nav.TButton",
            )
        {
            "import": self._render_import,
            "settings": self._render_settings,
            "run": self._render_run,
            "analyze": self._render_analyze,
        }[self.page]()

    def _render_import(self) -> None:
        self._heading("Import data and model", "Select five headered tables. CSV, TSV, whitespace text, and XLSX are supported; separators are detected automatically.")
        root = self._card(self.content)
        root_entry = self._field(root, "Framework repository", str(self.repositories[self.method]), "Browse", directory=True)
        config_entry = self._field(root, "Configuration file", str(self.config_paths[self.method]), "Choose")

        model_data = self._card(self.content)
        model_path, observation_path = _get_input_paths(self.config, self.method)
        model_dir = Path(model_path).expanduser() if model_path else Path()
        observation_dir = Path(observation_path).expanduser() if observation_path else Path()
        repository = self.repositories[self.method].expanduser().resolve()
        if model_path and not model_dir.is_absolute():
            model_dir = repository / model_dir
        if observation_path and not observation_dir.is_absolute():
            observation_dir = repository / observation_dir
        table_filetypes = [
            ("Delimited tables", "*.csv *.tsv *.tab *.txt *.dat"),
            ("Excel workbooks", "*.xlsx"),
            ("All files", "*.*"),
        ]
        table_entries = {}
        saved_paths = self.import_table_paths.get(self.method, {})
        for key in MODEL_TABLE_KEYS:
            filename, label = TABLE_FILES[key]
            table_entries[key] = self._field(
                model_data,
                label,
                saved_paths.get(key, str(model_dir / filename) if model_path else ""),
                "Choose",
                filetypes=table_filetypes,
            )

        observation_data = self._card(self.content)
        for key in OBSERVATION_TABLE_KEYS:
            filename, label = TABLE_FILES[key]
            table_entries[key] = self._field(
                observation_data,
                label,
                saved_paths.get(key, str(observation_dir / filename) if observation_path else ""),
                "Choose",
                filetypes=table_filetypes,
            )

        message = tk.Label(self.content, text="Files must have a header row and numeric data. The GUI validates row/column compatibility and converts tables to a consistent run-local format. No delimiter setting is needed.", bg=COLORS["window"], fg=COLORS["muted"], wraplength=850, justify="left")
        message.pack(anchor="w", pady=(0, 15))

        controls = tk.Frame(self.content, bg=COLORS["window"])
        controls.pack(anchor="w")

        def remember_table_paths(_event=None) -> None:
            self.import_table_paths[self.method] = {
                key: entry.get().strip() for key, entry in table_entries.items()
            }

        for entry in table_entries.values():
            entry.bind("<KeyRelease>", remember_table_paths, add="+")
            entry.bind("<FocusOut>", remember_table_paths, add="+")

        def load_config() -> None:
            repo = Path(root_entry.get()).expanduser()
            selected = Path(config_entry.get()).expanduser()
            if not selected.is_absolute():
                selected = repo / selected
            self.repositories[self.method] = repo
            self.config_paths[self.method] = selected
            if self._load_config(self.method):
                self._render_page()
                self.status_label.configure(text="CONFIG LOADED", fg=COLORS["success"])

        def apply_data() -> None:
            if not self.config:
                messagebox.showwarning("Load a config", "Load a config file before applying data paths.")
                return
            repository = self.repositories[self.method].expanduser().resolve()
            selected_tables = {}
            for key, entry in table_entries.items():
                value = Path(entry.get().strip()).expanduser()
                if not value.is_absolute():
                    value = repository / value
                selected_tables[key] = str(value)
            self.import_table_paths[self.method] = {
                key: entry.get().strip() for key, entry in table_entries.items()
            }
            try:
                frames = _validate_input_tables(selected_tables)
            except Exception as exc:
                self.import_tables.pop(self.method, None)
                messagebox.showerror("Could not read input tables", str(exc))
                return
            self.import_tables[self.method] = selected_tables
            summary = "\n".join(
                f"{TABLE_FILES[key][1]}: {len(frame)} rows × {frame.shape[1]} columns"
                for key, frame in frames.items()
            )
            self.status_label.configure(text="DATA TABLES READY", fg=COLORS["success"])
            messagebox.showinfo("Input tables ready", f"Detected formats and validated dimensions:\n\n{summary}\n\nThe GUI will normalize these files for the selected framework when the run starts.")

        _make_button(controls, "Load config", load_config).pack(side="left", padx=(0, 10))
        _make_button(controls, "Validate and prepare tables", apply_data).pack(side="left")

    def _render_settings(self) -> None:
        self._heading(
            "Calibration settings",
            "Toggle true/false options, enter numbers and text directly, then save a working copy.",
        )
        source = self.config_paths[self.method]
        tk.Label(
            self.content,
            text=f"Source: {source}",
            bg=COLORS["window"],
            fg=COLORS["muted"],
            anchor="w",
            wraplength=950,
        ).pack(fill="x", pady=(0, 9))

        style = ttk.Style(self)
        style.configure("Dark.TNotebook", background=COLORS["window"], borderwidth=0)
        style.configure(
            "Dark.TNotebook.Tab",
            background=COLORS["button"],
            foreground=COLORS["text"],
            padding=(13, 8),
            font=("Avenir Next", 10, "bold"),
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", COLORS["button_selected"]), ("active", COLORS["button_inactive"])],
            foreground=[("selected", COLORS["text"])],
        )
        notebook = ttk.Notebook(self.content, style="Dark.TNotebook")
        notebook.pack(fill="both", expand=True)
        self.settings_values: list[tuple[tuple[str, ...], str, tk.Variable]] = []

        def write_value(root: dict, path: tuple[str, ...], value) -> None:
            target = root
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        def make_scrollable_tab(title: str):
            tab = tk.Frame(notebook, bg=COLORS["window"])
            notebook.add(tab, text=title.replace("_", " ").title())
            shell = tk.Frame(tab, bg=COLORS["window"])
            shell.pack(fill="both", expand=True, pady=(8, 0))
            canvas = tk.Canvas(shell, bg=COLORS["window"], highlightthickness=0)
            scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            inner = tk.Frame(canvas, bg=COLORS["window"])
            window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
            inner.bind(
                "<Configure>",
                lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.bind(
                "<Configure>",
                lambda event: canvas.itemconfigure(window_id, width=event.width),
            )
            return inner

        def render_value(parent: tk.Widget, key: str, value, path: tuple[str, ...]) -> None:
            if isinstance(value, dict):
                section = tk.LabelFrame(
                    parent,
                    text=key.replace("_", " "),
                    bg=COLORS["panel"],
                    fg=COLORS["text"],
                    padx=12,
                    pady=8,
                    font=("Avenir Next", 10, "bold"),
                    highlightbackground=COLORS["border"],
                    highlightthickness=1,
                )
                section.pack(fill="x", padx=8, pady=7)
                for child_key, child_value in value.items():
                    render_value(section, child_key, child_value, path + (child_key,))
                return

            if path[-1] in {"input_delimiter", "input_header"}:
                return

            row = tk.Frame(parent, bg=COLORS["panel"])
            row.pack(fill="x", padx=8, pady=4)
            tk.Label(
                row,
                text=key.replace("_", " "),
                width=29,
                anchor="w",
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=("Avenir Next", 10),
            ).pack(side="left", padx=(5, 10))

            if isinstance(value, bool):
                variable = tk.BooleanVar(value=value)
                control = tk.Checkbutton(
                    row,
                    text="True" if value else "False",
                    variable=variable,
                    command=lambda var=variable, label=None: None,
                    onvalue=True,
                    offvalue=False,
                    bg=COLORS["panel"],
                    fg=COLORS["text"],
                    activebackground=COLORS["panel"],
                    activeforeground=COLORS["text"],
                    selectcolor=COLORS["panel_alt"],
                    font=("Avenir Next", 10),
                )
                control.configure(command=lambda var=variable, widget=control: widget.configure(text="True" if var.get() else "False"))
                control.pack(side="left", anchor="w")
                kind = "bool"
            elif isinstance(value, str) and (choices := self._settings_enum_options(key, path)):
                variable = tk.StringVar(value=value)
                control = ttk.Combobox(
                    row,
                    textvariable=variable,
                    values=choices,
                    state="readonly",
                    style="Dark.TCombobox",
                    width=24,
                )
                control.pack(side="left", anchor="w")
                kind = "str"
            else:
                if isinstance(value, list):
                    initial = json.dumps(value)
                    kind = "json"
                else:
                    initial = "" if value is None else str(value)
                    kind = "none" if value is None else type(value).__name__
                variable = tk.StringVar(value=initial)
                entry = tk.Entry(
                    row,
                    textvariable=variable,
                    relief="flat",
                    bg=COLORS["panel_alt"],
                    fg=COLORS["text"],
                    insertbackground=COLORS["text"],
                    font=("Menlo", 10),
                )
                entry.pack(side="left", fill="x", expand=True, ipady=6)
            self.settings_values.append((path, kind, variable))

        labels_key = self.label_settings_key
        for section_name, section_value in self.config.items():
            if section_name in {"plot_settings_path", "input_settings", labels_key}:
                continue
            tab_content = make_scrollable_tab(section_name)
            render_value(tab_content, section_name, section_value, (section_name,))

        labels = self.config.get(labels_key, self.plot_settings_by_method.get(self.method, {}))
        tab_content = make_scrollable_tab("plot_labels")
        if labels:
            render_value(tab_content, labels_key, labels, (labels_key,))
        else:
            tk.Label(
                tab_content,
                text="No label settings are defined in this config.",
                bg=COLORS["window"],
                fg=COLORS["muted"],
            ).pack(anchor="w", padx=12, pady=12)

        def save_copy() -> bool:
            updated = json.loads(json.dumps(self.config))
            try:
                for path, kind, variable in self.settings_values:
                    raw = variable.get()
                    if kind == "bool":
                        value = bool(raw)
                    elif kind == "int":
                        value = int(raw)
                    elif kind == "float":
                        value = float(raw)
                    elif kind == "json":
                        value = json.loads(raw)
                    elif kind == "none":
                        value = None if not raw.strip() else raw
                    else:
                        value = raw
                    write_value(updated, path, value)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                messagebox.showerror("Invalid setting", f"Check the highlighted field values and JSON arrays.\n\n{exc}")
                return False

            if labels_key in updated:
                self.plot_settings_by_method[self.method] = json.loads(json.dumps(updated[labels_key]))
            updated.pop("plot_settings_path", None)
            self.configs[self.method] = updated
            destination = APP_ROOT / "gui_runs" / "working_configs" / (
                "caked_up.json" if self.method == "CAKED-UP" else "ib_framework.json"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(updated, indent=2), encoding="utf-8")
            messagebox.showinfo("Working copy saved", f"Saved without changing the source config:\n{destination}")
            self.status_label.configure(text="WORKING CONFIG SAVED", fg=COLORS["success"])
            return True

        self._save_settings_callback = save_copy
        _make_button(self.content, "Save working config", save_copy).pack(anchor="w", pady=(12, 0))

    def _settings_prior_options(self, path: tuple[str, ...]) -> set[str]:
        """Return configured kappa prior blocks available beside a selector."""
        node = self.config
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                return set()
            node = node[part]
        if not isinstance(node, dict):
            return set()
        return {name for name in ("gp", "lasso", "fused_lasso", "spike_slab", "horseshoe") if isinstance(node.get(name), dict)}

    def _settings_enum_options(self, key: str, path: tuple[str, ...]) -> list[str]:
        known_options = {
            "theta_initialization": ["fixed", "compositional"],
            "calibration_prediction_animation_space": ["physical", "normalized"],
            "known_delta_form": ["power_law", "trig_funct", "linear", "polynomial"],
        }
        if key == "shrinkage_prior":
            available = self._settings_prior_options(path)
            return [item for item in ("gp", "lasso", "fused_lasso", "spike_slab", "horseshoe") if item in available]
        if key == "method":
            try:
                kappa = self.config["calibration_settings"]["shrinkage_models"]["kappa"]
                available = self._settings_prior_options(("calibration_settings", "shrinkage_models", "kappa", "shrinkage_prior"))
                available.update(
                    group.get("method")
                    for group in kappa.get("kappa_groups", [])
                    if isinstance(group, dict) and isinstance(group.get("method"), str)
                )
                return sorted(available)
            except (KeyError, TypeError):
                return []
        if key == "known_theta_form":
            params = self.config.get("cross_validation_settings", {}).get("known_theta_form_params", {})
            options = list(params) if isinstance(params, dict) else []
            if options:
                return options
        if key == "known_delta_form":
            params = self.config.get("cross_validation_settings", {}).get("known_delta_form_params", {})
            options = list(params) if isinstance(params, dict) else []
            if options:
                return options
        return known_options.get(key, [])

    def _build_run_config(self, run_dir: Path) -> tuple[Path, Path, list[str], Path]:
        config = json.loads(json.dumps(self.config))
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        _set_output_paths(config, self.method, results_dir)
        repo = self.repositories[self.method].expanduser().resolve()
        config_source = self.config_paths[self.method].expanduser().resolve()
        _resolve_runtime_paths(config, self.method, repo, config_source)
        if self.method == "CAKED-UP" and self.plot_settings_by_method.get(self.method):
            config["plot_settings"] = json.loads(json.dumps(self.plot_settings_by_method[self.method]))
            config.pop("plot_settings_path", None)
        selected_tables = self.import_tables.get(self.method)
        if not selected_tables:
            raise ValueError("Validate and apply all five input tables before starting a run.")
        _stage_input_tables(config, self.method, selected_tables, run_dir)

        if self.method == "CAKED-UP":
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            command = [sys.executable, "-u", str(repo / "main.py"), str(config_path)]
            return config_path, results_dir, command, repo

        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", str(repo / "main.py"), str(config_path)]
        return config_path, results_dir, command, repo

    def _render_run(self) -> None:
        self._heading("Run and monitor", "Track calibration predictions as the sampler runs.")
        info = self._card(self.content)
        repo = self.repositories[self.method]
        tk.Label(info, text=f"Framework: {self.method}\nRepository: {repo}\nConfig: {self.config_paths[self.method]}", bg=COLORS["panel"], fg=COLORS["text"], justify="left", anchor="w").pack(fill="x")
        actions = tk.Frame(self.content, bg=COLORS["window"])
        actions.pack(fill="x", pady=(0, 12))
        has_validated_data = self.method in self.import_tables
        run_tooltip = (
            "Import and validate the five simulator and observation tables first."
            if not has_validated_data
            else "Start the selected framework with the validated, normalized tables."
        )
        self.run_button = _make_button(
            actions,
            "Start calibration",
            self.start_run,
            state=("disabled" if self.process and self.process.poll() is None else "normal")
            if has_validated_data else "disabled",
            tooltip=run_tooltip,
        )
        self.run_button.pack(side="left")
        _make_button(actions, "Clear log", self.clear_log).pack(side="left", padx=10)
        self.run_status = tk.Label(actions, text="Idle", bg=COLORS["window"], fg=COLORS["muted"], font=("Avenir Next", 10, "bold"))
        self.run_status.pack(side="left", padx=12)
        monitor = tk.Frame(self.content, bg=COLORS["window"])
        monitor.pack(fill="both", expand=True)
        log_frame = tk.Frame(monitor, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, width=520)
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", bg=COLORS["log"], fg=COLORS["text"], insertbackground=COLORS["text"], font=("Menlo", 9), padx=12, pady=12)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        if self.log_history:
            self.log_text.configure(state="normal")
            self.log_text.insert("1.0", "".join(self.log_history))
            self.log_text.configure(state="disabled")
            self.log_text.see("end")
        if self.process and self.process.poll() is None:
            self.run_status.configure(text="Calibration is running", fg=COLORS["warning"])
            self.status_label.configure(text="CALIBRATION RUNNING", fg=COLORS["warning"])
        elif self.process and self.process.returncode == 0:
            self.run_status.configure(text="Completed successfully", fg=COLORS["success"])
            self.status_label.configure(text="RUN COMPLETE", fg=COLORS["success"])

        side = tk.Frame(monitor, bg=COLORS["window"], width=440)
        side.pack(side="left", fill="both")
        side.pack_propagate(False)
        calibration_card = tk.Frame(side, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, padx=10, pady=8)
        calibration_card.pack(fill="both", expand=True, pady=(0, 10))
        tk.Label(calibration_card, text="LIVE CALIBRATION PREDICTION", bg=COLORS["panel"], fg=COLORS["text"], font=("Avenir Next", 9, "bold")).pack(anchor="w", padx=5, pady=(0, 5))
        self.live_prediction_status = tk.Label(calibration_card, text="Waiting for sampler update", bg=COLORS["panel"], fg=COLORS["muted"], font=("Avenir Next", 8))
        self.live_prediction_status.pack(anchor="w", padx=5)
        self.live_canvas = tk.Canvas(calibration_card, height=440, bg=COLORS["panel_alt"], highlightthickness=0)
        self.live_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.last_live_prediction_mtime = 0.0
        if self.current_results_dir:
            self.after(400, self._poll_live_monitor)

    def start_run(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not self.config:
            messagebox.showwarning("No config loaded", "Load a valid config from Import Data / Model first.")
            return
        if self.method not in self.import_tables:
            messagebox.showwarning("Input tables not ready", "Select and validate all five model and observation tables on Import Data / Model first.")
            return
        repo = self.repositories[self.method].expanduser().resolve()
        if not (repo / "main.py").is_file():
            messagebox.showerror("Framework not found", f"Could not find main.py in {repo}")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        run_dir = APP_ROOT / "gui_runs" / self.method.lower().replace(" ", "_") / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            config_path, results_dir, command, cwd = self._build_run_config(run_dir)
            self.current_results_dir = results_dir
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Could not start calibration", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.run_status.configure(text=f"Running · {run_dir.name}", fg=COLORS["warning"])
        self.status_label.configure(text="CALIBRATION RUNNING", fg=COLORS["warning"])
        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()
        self.after(120, self._poll_log)
        self.after(500, self._poll_live_monitor)
        self._append_log(f"Run folder: {run_dir}\nConfig: {config_path}\nCommand: {' '.join(command)}\n\n")

    def _poll_live_monitor(self) -> None:
        if not hasattr(self, "live_canvas") or not self.live_canvas.winfo_exists():
            return
        if self.current_results_dir:
            snapshot = self.current_results_dir.parent / "live_prediction.json"
            if snapshot.is_file():
                try:
                    modified = snapshot.stat().st_mtime
                    if modified > self.last_live_prediction_mtime:
                        payload = json.loads(snapshot.read_text(encoding="utf-8"))
                        self._draw_live_prediction(payload)
                        self.last_live_prediction_mtime = modified
                except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
                    pass
        if self.process and self.process.poll() is None:
            self.after(700, self._poll_live_monitor)

    def _draw_live_prediction(self, payload: dict) -> None:
        canvas = self.live_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 360)
        height = max(canvas.winfo_height(), 220)
        observed = np.asarray(payload["observed"], dtype=float)
        predicted = np.asarray(payload["predicted"], dtype=float)
        uncertainty = np.asarray(payload["uncertainty"], dtype=float)
        x_values = np.asarray(payload["x"], dtype=float)
        count = min(len(observed), len(predicted), len(uncertainty), len(x_values))
        observed, predicted, uncertainty, x_values = (
            values[:count] for values in (observed, predicted, uncertainty, x_values)
        )
        if count == 0:
            return

        lower = predicted - 2 * uncertainty
        upper = predicted + 2 * uncertainty
        y_min = float(min(np.min(observed), np.min(lower)))
        y_max = float(max(np.max(observed), np.max(upper)))
        if y_max <= y_min:
            y_min -= 1.0
            y_max += 1.0
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        if x_max <= x_min:
            x_min -= 0.5
            x_max += 0.5
        left, right, top, bottom = 42, width - 14, 14, height - 28

        def px(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def py(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        canvas.create_line(left, top, left, bottom, fill=COLORS["muted"])
        canvas.create_line(left, bottom, right, bottom, fill=COLORS["muted"])
        order = np.argsort(x_values)
        band_points = [(px(x_values[i]), py(upper[i])) for i in order]
        band_points += [(px(x_values[i]), py(lower[i])) for i in order[::-1]]
        if len(band_points) >= 4:
            canvas.create_polygon(band_points, fill=COLORS["button"], outline="", stipple="gray50")
        line_points = [coordinate for i in order for coordinate in (px(x_values[i]), py(predicted[i]))]
        if len(line_points) >= 4:
            canvas.create_line(*line_points, fill="#e1e4e7", width=2, smooth=False)
        for x_value, y_value in zip(x_values, observed):
            x_pixel, y_pixel = px(x_value), py(y_value)
            canvas.create_oval(x_pixel - 3, y_pixel - 3, x_pixel + 3, y_pixel + 3, fill="#9da4aa", outline="")
        canvas.create_text(7, top + 2, text=f"{y_max:.2g}", fill=COLORS["muted"], anchor="w", font=("Avenir Next", 8))
        canvas.create_text(7, bottom, text=f"{y_min:.2g}", fill=COLORS["muted"], anchor="w", font=("Avenir Next", 8))
        canvas.create_text((left + right) / 2, height - 7, text="Domain / observation index", fill=COLORS["muted"], font=("Avenir Next", 8))
        canvas.create_text(right - 2, top + 2, text="— prediction   • observed   band ±2 SD", fill=COLORS["text"], anchor="ne", font=("Avenir Next", 8))
        self.live_prediction_status.configure(
            text=f"Iteration {payload['iteration']:,} / {payload['total_iterations']:,}"
        )

    def _stop_gif_animation(self, label: tk.Widget) -> None:
        state = getattr(self, "gif_players", {}).pop(label, None)
        if state and state.get("after_id"):
            try:
                label.after_cancel(state["after_id"])
            except tk.TclError:
                pass

    def _start_gif_animation(self, path: Path, label: tk.Label, max_size: tuple[int, int]) -> None:
        self._stop_gif_animation(label)
        try:
            with Image.open(path) as source:
                frame_count = int(getattr(source, "n_frames", 1))
            if frame_count < 1:
                return
            state = {
                "path": path,
                "max_size": max_size,
                "frame_count": frame_count,
                "index": 0,
                "after_id": None,
            }
            self.gif_players[label] = state
            self._advance_gif_frame(label, state)
        except (OSError, tk.TclError):
            label.configure(image="", text=f"Could not display animation: {path.name}")

    def _advance_gif_frame(self, label: tk.Label, state: dict) -> None:
        try:
            if not label.winfo_exists():
                return
            if "path" in state:
                with Image.open(state["path"]) as source:
                    source.seek(state["index"])
                    image = source.convert("RGB")
                    image.thumbnail(state["max_size"], Image.Resampling.LANCZOS)
                    duration = max(40, int(source.info.get("duration", 100)))
                    photo = ImageTk.PhotoImage(image)
                state["photo"] = photo
                state["index"] = (state["index"] + 1) % state["frame_count"]
            else:
                photo, duration = state["frames"][state["index"]]
                state["index"] = (state["index"] + 1) % len(state["frames"])
            label.configure(image=photo, text="")
            label.image = photo
            state["after_id"] = label.after(duration, self._advance_gif_frame, label, state)
        except (OSError, tk.TclError):
            return

    def _read_process_output(self, process: subprocess.Popen) -> None:
        if process.stdout:
            for line in process.stdout:
                self.log_queue.put(line)
        process.wait()
        self.log_queue.put(None)

    def _poll_log(self) -> None:
        finished = False
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
            else:
                self._append_log(line)
        if finished and self.process:
            return_code = self.process.returncode
            if hasattr(self, "run_button") and self.run_button.winfo_exists():
                self.run_button.configure(state="normal")
            if return_code == 0:
                if hasattr(self, "run_status") and self.run_status.winfo_exists():
                    self.run_status.configure(text="Completed successfully", fg=COLORS["success"])
                self.status_label.configure(text="RUN COMPLETE", fg=COLORS["success"])
                messagebox.showinfo("Calibration complete", "The calibration finished. Open Analyze to browse its figures and result files.")
            else:
                if hasattr(self, "run_status") and self.run_status.winfo_exists():
                    self.run_status.configure(text=f"Exited with status {return_code}", fg=COLORS["danger"])
                self.status_label.configure(text="RUN FAILED", fg=COLORS["danger"])
            return
        self.after(120, self._poll_log)

    def _append_log(self, text: str) -> None:
        self.log_history.append(text)
        self.log_history = self.log_history[-5000:]
        if not hasattr(self, "log_text"):
            return
        try:
            if not self.log_text.winfo_exists():
                return
        except tk.TclError:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        if hasattr(self, "log_text"):
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.configure(state="disabled")

    def _render_analyze(self) -> None:
        self._heading("Analyze results", "Select a figure or result file to preview it.")
        row = tk.Frame(self.content, bg=COLORS["window"])
        row.pack(fill="x", pady=(0, 13))
        default_dir = str(self.current_results_dir or "")
        self.results_entry = self._field(row, "Results directory", default_dir, "Browse", directory=True)
        _make_button(row, "Refresh", self.refresh_results).pack(anchor="w", pady=(8, 0))

        panel = tk.Frame(self.content, bg=COLORS["window"])
        panel.pack(fill="both", expand=True)
        list_frame = tk.Frame(panel, bg=COLORS["panel"], width=280, highlightbackground=COLORS["border"], highlightthickness=1)
        list_frame.pack(side="left", fill="y", padx=(0, 14))
        list_frame.pack_propagate(False)
        tk.Label(list_frame, text="FIGURES / FILES", bg=COLORS["panel"], fg=COLORS["text"], font=("Avenir Next", 9, "bold")).pack(anchor="w", padx=12, pady=10)
        self.result_list = tk.Listbox(
            list_frame,
            relief="flat",
            font=("Avenir Next", 10),
            activestyle="dotbox",
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            selectbackground=COLORS["button_active"],
            selectforeground=COLORS["text"],
        )
        self.result_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.result_list.bind("<<ListboxSelect>>", self._show_selected_result)
        preview = tk.Frame(panel, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        preview.pack(side="left", fill="both", expand=True)
        self.preview_image_frame = tk.Frame(preview, bg=COLORS["panel"])
        self.preview_image_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.preview_canvas = tk.Canvas(
            self.preview_image_frame,
            bg=COLORS["panel"],
            highlightthickness=0,
            xscrollcommand=lambda first, last: self.preview_xscroll.set(first, last),
            yscrollcommand=lambda first, last: self.preview_yscroll.set(first, last),
        )
        self.preview_xscroll = ttk.Scrollbar(self.preview_image_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_yscroll = ttk.Scrollbar(self.preview_image_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_yscroll.grid(row=0, column=1, sticky="ns")
        self.preview_xscroll.grid(row=1, column=0, sticky="ew")
        self.preview_image_frame.rowconfigure(0, weight=1)
        self.preview_image_frame.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            self.preview_canvas,
            text="Select a figure or result file",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
        )
        self.preview_canvas_window = self.preview_canvas.create_window((0, 0), window=self.preview_label, anchor="nw")
        self.preview_label.bind(
            "<Configure>",
            lambda _event: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")),
        )

        self.preview_text_frame = tk.Frame(preview, bg=COLORS["panel"])
        self.preview_text = tk.Text(
            self.preview_text_frame,
            wrap="none",
            state="disabled",
            font=("Menlo", 9),
            padx=10,
            pady=10,
            height=12,
            bg=COLORS["panel_alt"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
        )
        text_y = ttk.Scrollbar(self.preview_text_frame, orient="vertical", command=self.preview_text.yview)
        text_x = ttk.Scrollbar(self.preview_text_frame, orient="horizontal", command=self.preview_text.xview)
        self.preview_text.configure(yscrollcommand=text_y.set, xscrollcommand=text_x.set)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        text_y.grid(row=0, column=1, sticky="ns")
        text_x.grid(row=1, column=0, sticky="ew")
        self.preview_text_frame.rowconfigure(0, weight=1)
        self.preview_text_frame.columnconfigure(0, weight=1)
        self.artifact_info = tk.Label(preview, text="", bg=COLORS["panel"], fg=COLORS["muted"], anchor="w", justify="left", wraplength=700)
        self.artifact_info.pack(fill="x", padx=12, pady=(0, 10))
        self.result_paths: list[Path] = []
        if default_dir:
            self.refresh_results()

    def refresh_results(self) -> None:
        if not hasattr(self, "result_list"):
            return
        root = Path(self.results_entry.get()).expanduser()
        self.result_list.delete(0, "end")
        self.result_paths = []
        self._stop_gif_animation(self.preview_label)
        self.preview_label.configure(image="", text="Select a figure or result file")
        self.preview_text_frame.pack_forget()
        self.preview_image_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.artifact_info.configure(text="")
        if not root.is_dir():
            return
        supported = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".csv", ".json", ".npz", ".txt"}
        self.result_paths = [
            path for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in supported
        ]
        for path in self.result_paths:
            self.result_list.insert("end", str(path.relative_to(root)))

    def _show_selected_result(self, _event=None) -> None:
        selection = self.result_list.curselection()
        if not selection:
            return
        path = self.result_paths[selection[0]]
        self.image_reference = None
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
            try:
                self.preview_text_frame.pack_forget()
                self.preview_image_frame.pack(fill="both", expand=True, padx=5, pady=5)
                if path.suffix.lower() == ".gif":
                    self._start_gif_animation(path, self.preview_label, (850, 590))
                else:
                    self._stop_gif_animation(self.preview_label)
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((850, 590), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                    self.image_reference = photo
                    self.preview_label.configure(image=photo, text="")
                self.artifact_info.configure(text=str(path))
            except Exception as exc:
                self.preview_label.configure(image="", text=f"Could not preview image:\n{exc}")
        else:
            self._stop_gif_animation(self.preview_label)
            self.preview_image_frame.pack_forget()
            self.preview_text_frame.pack(fill="both", expand=True, padx=5, pady=5)
            try:
                if path.suffix.lower() == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        lines = []
                        for name in archive.files:
                            values = archive[name]
                            detail = f"{name}: shape={values.shape}, dtype={values.dtype}"
                            if np.issubdtype(values.dtype, np.number) and values.size:
                                detail += f", min={np.nanmin(values):.6g}, max={np.nanmax(values):.6g}"
                            lines.append(detail)
                        content = "\n".join(lines) or "No arrays found."
                elif path.suffix.lower() == ".json":
                    content = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
                else:
                    content = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:120])
                self.preview_text.configure(state="normal")
                self.preview_text.delete("1.0", "end")
                self.preview_text.insert("1.0", content[:40000])
                self.preview_text.configure(state="disabled")
            except Exception as exc:
                self.preview_text.configure(state="normal")
                self.preview_text.delete("1.0", "end")
                self.preview_text.insert("1.0", f"Could not read artifact:\n{exc}")
                self.preview_text.configure(state="disabled")
            self.artifact_info.configure(text=str(path))


if __name__ == "__main__":
    app = CalibrationGUI()
    app.mainloop()
