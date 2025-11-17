import shutil
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from materials_solver import (
    DebyeControls,
    DoubleLorentzControls,
    compute_debye_ceps,
    compute_debye_lorentz_ceps,
    fit_debye,
    fit_double_lorentz,
    generate_dbe_report,
    generate_dlm_report,
    read_material_file,
)


def _extract_parameter_mapping(path: Path, marker: str) -> Optional[Dict[str, str]]:
    """Returns a header/value mapping for the block identified by marker."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for idx, line in enumerate(lines):
        if marker in line:
            header_index = _find_table_header_index(lines, idx + 1)
            if header_index is None:
                return None
            value_index = _find_numeric_line_index(lines, header_index + 1)
            if value_index is None:
                return None
            header_tokens = [token.upper() for token in lines[header_index].split()]
            data_tokens = lines[value_index].split()
            if len(data_tokens) < len(header_tokens):
                return None
            return dict(zip(header_tokens, data_tokens))
    return None


def _find_table_header_index(lines: List[str], start: int) -> Optional[int]:
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if stripped[0] in "=<>":
            continue
        if stripped.upper().startswith("TO FIX"):
            continue
        tokens = stripped.split()
        if tokens and all(_token_has_letter(token) for token in tokens):
            return idx
    return None


def _find_numeric_line_index(lines: List[str], start: int) -> Optional[int]:
    for idx in range(start, len(lines)):
        tokens = lines[idx].split()
        if not tokens:
            continue
        if all(_is_numeric_token(token) for token in tokens):
            return idx
    return None


def _token_has_letter(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def _is_numeric_token(token: str) -> bool:
    cleaned = token.rstrip(",;")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def parse_dbe_file(path: Path) -> Optional[Dict[str, str]]:
    marker = "LEAST SQUARE FIT TO DEBYE PARAMETERS:"
    return _extract_parameter_mapping(path, marker)


def parse_2lm_file(path: Path) -> Optional[Dict[str, str]]:
    markers = (
        "LEAST SQUARE FIT TO DOUBLE LORENTZIAN PARAMETERS:",
        "LEAST SQUARE FIT TO DEBYE-LORENTZIAN PARAMETERS:",
    )
    for marker in markers:
        parsed = _extract_parameter_mapping(path, marker)
        if parsed:
            return parsed
    return None


@dataclass(frozen=True)
class TabConfig:
    key: str
    notebook_label: str
    file_extension: str
    parser: Callable[[Path], Optional[Dict[str, str]]]
    value_headers: Tuple[str, ...]
    small_headers: Tuple[str, ...]
    plot_color: str
    polynomial_headers: Tuple[str, ...]
    ini_mapping: Dict[str, Any]
    value_aliases: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    monotonic_constraints: Dict[str, str] = field(default_factory=dict)

    @property
    def stem_header(self) -> str:
        return f"STEM OF THE {self.file_extension.upper()} FILE"

    @property
    def large_headers(self) -> Tuple[str, ...]:
        return ("ALPHA",) + self.value_headers + (self.stem_header,)


@dataclass
class ColumnSelection:
    tab_key: str
    header: str
    large_index: int


@dataclass
class TabState:
    config: TabConfig
    frame: ttk.Frame
    figure: Figure
    canvas: FigureCanvasTkAgg
    axes: Any
    large_tree: ttk.Treeview
    small_tree: ttk.Treeview
    summary_label: ttk.Label
    summary_tree: ttk.Treeview
    small_rows: Dict[str, str]
    column_index: Dict[str, int]
    large_overlays: List[tk.Widget]
    small_overlays: List[tk.Widget]
    ini_label_var: tk.StringVar
    ini_tree: ttk.Treeview


@dataclass
class ParameterBlock:
    path: Path
    headers: List[str]
    values: List[str]
    header_index: int
    value_index: int
    lines: List[str]


@dataclass
class ParameterEditorRow:
    header: str
    mode_var: tk.StringVar
    value_entry: ttk.Entry
    original_value: str
    preview_label: ttk.Label


@dataclass
class AutoConvergeSettings:
    target_rms: float
    max_iterations: int
    convergence_factor: float
    min_improvement: float


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<FocusOut>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(padx=4, pady=2)

    def _hide(self, _event=None) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


TAB_CONFIGS: Tuple[TabConfig, ...] = (
    TabConfig(
        key="EPS",
        notebook_label="EPS_DEBYE",
        file_extension=".DBE",
        parser=parse_dbe_file,
        value_headers=("FRES", "DEPS", "EPSV", "GAMMA", "SIGE", "RMS"),
        small_headers=("FRES", "DEPS", "EPSV", "GAMMA", "SIGE"),
        plot_color="#d9534f",
        polynomial_headers=("FRES", "DEPS", "EPSV", "GAMMA", "SIGE"),
        ini_mapping={
            "FRES": "FRES0",
            "DEPS": "DEPS0",
            "EPSV": "EPSV0",
            "GAMMA": "GAMM0",
            "SIGE": "SIGE0",
        },
        monotonic_constraints={
            "FRES": "auto",
            "DEPS": "auto",
            "EPSV": "auto",
            "GAMMA": "auto",
            "SIGE": "auto",
        },
    ),
    TabConfig(
        key="MUS",
        notebook_label="MUS_DL",
        file_extension=".2LM",
        parser=parse_2lm_file,
        value_headers=("FR_M1", "DMUR1", "GAMM1", "FR_M2", "DMUR2",
                       "GAMM2", "MURV", "EPSV", "RMS"),
        small_headers=("FR_M1", "DMUR1", "GAMM1", "FR_M2", "DMUR2",
                       "GAMM2", "MURV", "EPSV"),
        plot_color="#0275d8",
        polynomial_headers=("FR_M1", "DMUR1", "GAMM1", "FR_M2", "DMUR2",
                            "GAMM2", "MURV", "EPSV"),
        ini_mapping={
            "FR_M1": ("FR_M01", "FR_M0D"),
            "DMUR1": ("DMUR01", "DMUR0D"),
            "GAMM1": ("GAMM01", "GAMM0D"),
            "FR_M2": ("FR_M02", "FR_M0L"),
            "DMUR2": ("DMUR02", "DMUR0L", "4MUR02"),
            "GAMM2": ("GAMM02", "GAMM0L"),
            "MURV": "MURV0",
            "EPSV": "SIGM0",
        },
        value_aliases={"EPSV": ("SIGM",)},
        monotonic_constraints={
            "FR_M1": "auto",
            "DMUR1": "auto",
            "GAMM1": "auto",
            "FR_M2": "auto",
            "DMUR2": "auto",
            "GAMM2": "auto",
            "MURV": "auto",
            "EPSV": "auto",
        },
    ),
)


class MaterialGui:
    CONTROL_OPTIONS = (
        ("estimate", "Let computer estimate (0)"),
        ("fix", "Keep parameter fixed (>0)"),
        ("negative", "Use negative initial guess (<0)"),
    )
    INI_TREND_DEVIATION_TAG = "ini-trend-deviation"
    INI_TREND_COLOR = "#fdecea"
    INI_TREND_MOVING_RADIUS = 1
    INI_TREND_ARROW_MAP = {"up": "↑", "down": "↓", "flat": "→"}
    INI_ADJUSTMENT_PRESETS: Tuple[Tuple[str, float], ...] = (
        ("-10%", -0.10),
        ("-5%", -0.05),
        ("+5%", 0.05),
        ("+10%", 0.10),
    )

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Material Parameter Viewer")
        self.root.geometry("1100x650")
        self.status_var = tk.StringVar(value="No file loaded.")
        self.tabs: Dict[str, TabState] = {}
        self.selection: Optional[ColumnSelection] = None
        self.loaded_alpha_pairs: List[Tuple[str, str]] = []
        self.loaded_base_path: Optional[Path] = None
        self.loaded_lst_name: Optional[str] = None
        self.loaded_lst_path: Optional[Path] = None
        self.alpha0: Optional[float] = None
        self.idbkode: Optional[int] = None
        self.tab_extensions: Dict[str, Optional[str]] = {"EPS": ".DBE", "MUS": ".2LM"}
        self.lsf_data: Dict[str, Dict[str, Dict[str, str]]] = {"EPS": {}, "MUS": {}}
        self.loaded_lsf_path: Optional[Path] = None
        self.alpha_to_files: Dict[str, Dict[str, str]] = {"EPS": {}, "MUS": {}}
        self.alpha_to_stem: Dict[str, str] = {}
        self.polynomial_models: Dict[str, Dict[str, Dict[str, Any]]] = {"EPS": {}, "MUS": {}}
        self.parameter_blocks: Dict[str, Dict[str, ParameterBlock]] = {"EPS": {}, "MUS": {}}
        self.alpha_file_paths: Dict[str, Dict[str, Path]] = {"EPS": {}, "MUS": {}}
        self.ini_inline_widgets: Dict[str, Optional[tk.Widget]] = {"EPS": None, "MUS": None}
        self.loaded_lsf_path: Optional[Path] = None
        self.lsf_data: Dict[str, Dict[str, Dict[str, str]]] = {"EPS": {}, "MUS": {}}
        self.snapshot_dir: Optional[Path] = None
        self.snapshot_map: Dict[Path, Path] = {}
        self.diagnostics_listbox: Optional[tk.Listbox] = None
        self.diagnostics_quantity_var: Optional[tk.StringVar] = None
        self.diagnostics_status_var: Optional[tk.StringVar] = None
        self.diagnostics_axes_real = None
        self.diagnostics_axes_imag = None
        self.diagnostics_canvas: Optional[FigureCanvasTkAgg] = None
        self.notebook: Optional[ttk.Notebook] = None
        self.tab_frame_to_key: Dict[tk.Widget, str] = {}
        self._tooltips: List[Tooltip] = []
        self._build_style()
        self._build_header()
        self._build_notebook()

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("default")
        except tk.TclError:
            pass
        style.configure("Status.TLabel", anchor="w")
        base_bg = style.lookup("Treeview", "background") or "#ffffff"
        base_fg = style.lookup("Treeview", "foreground") or "#000000"
        style.map(
            "Treeview",
            background=[("selected", base_bg)],
            foreground=[("selected", base_fg)],
        )

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        if not text:
            return
        tooltip = Tooltip(widget, text)
        self._tooltips.append(tooltip)

    def _build_header(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=10, pady=5)
        load_btn = ttk.Button(header, text="Load .LST File", command=self.load_lst_file)
        load_btn.pack(side=tk.LEFT)
        self._add_tooltip(load_btn, "Load an .LST file to populate the parameter tables.")
        edit_btn = ttk.Button(header, text="Edit .LST", command=self.open_lst_editor)
        edit_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(edit_btn, "Open a dialog to edit alpha/stem entries in the current .LST file.")
        init_btn = ttk.Button(header, text="Generate Initial Fits", command=self.generate_initial_fits)
        init_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(init_btn, "Create DBE/2LM files from raw ROP/EMU measurements for all entries.")
        run_btn = ttk.Button(header, text="Run Fits", command=self.run_selected_tab_fits)
        run_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(run_btn, "Re-run fits for the currently selected tab using current INI seeds.")
        idb_btn = ttk.Button(header, text="Generate IDB", command=self.generate_idb_file)
        idb_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(idb_btn, "Build an IDB file from the latest polynomial models.")
        restore_btn = ttk.Button(header, text="Restore Originals", command=self.restore_original_files)
        restore_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(restore_btn, "Copy files back from the last snapshot created when loading data.")
        auto_btn = ttk.Button(header, text="Auto Converge", command=self.auto_converge)
        auto_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._add_tooltip(auto_btn, "Iteratively adjust the selected column's INI seeds to reach a target RMS.")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT, padx=12)

    def _build_notebook(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        for config in TAB_CONFIGS:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=config.notebook_label)
            state = self._build_tab(frame, config)
            self.tabs[config.key] = state
            self.tab_frame_to_key[frame] = config.key
        self._build_diagnostics_tab(notebook)

    def _build_tab(self, container: ttk.Frame, config: TabConfig) -> TabState:
        outer = ttk.Panedwindow(container, orient=tk.VERTICAL)
        outer.pack(fill="both", expand=True, padx=5, pady=5)

        top_split = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        bottom_split = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        outer.add(top_split, weight=1)
        outer.add(bottom_split, weight=2)

        # Plot
        plot_frame = ttk.Frame(top_split)
        top_split.add(plot_frame, weight=2)
        figure = Figure(figsize=(4, 3), dpi=100)
        axes = figure.add_subplot(111)
        axes.set_title("Select a column to plot")
        axes.set_xlabel("Alpha")
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Small table + controls
        control_frame = ttk.Frame(top_split)
        top_split.add(control_frame, weight=3)
        small_columns = ("ROW",) + config.small_headers
        small_tree = ttk.Treeview(
            control_frame,
            columns=small_columns,
            show="headings",
            selectmode="browse",
            height=4,
        )
        for column in small_columns:
            heading = "" if column == "ROW" else column
            anchor = "center"
            small_tree.heading(column, text=heading)
            small_tree.column(column, width=90 if column != "ROW" else 60, anchor=anchor, stretch=True)
        small_tree.pack(fill="both", expand=True, pady=(0, 8))
        small_rows = {
            "PWR": small_tree.insert("", "end", values=("PWR",) + ("0",) * len(config.small_headers)),
            "RMS": small_tree.insert("", "end", values=("RMS",) + ("",) * len(config.small_headers)),
        }
        small_tree.bind("<Button-1>", lambda event, key=config.key: self.handle_column_click(event, key))

        button_row = ttk.Frame(control_frame)
        button_row.pack(fill="x", pady=(0, 4))
        inc_btn = ttk.Button(
            button_row,
            text="Increase PWR",
            command=lambda key=config.key: self.adjust_power(key, 1),
        )
        inc_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._add_tooltip(inc_btn, "Raise the polynomial fit order for the highlighted column.")
        dec_btn = ttk.Button(
            button_row,
            text="Decrease PWR",
            command=lambda key=config.key: self.adjust_power(key, -1),
        )
        dec_btn.pack(side=tk.LEFT)
        self._add_tooltip(dec_btn, "Lower the polynomial fit order for the highlighted column.")
        ini_section = ttk.LabelFrame(control_frame, text="INI Parameter Controls")
        ini_section.pack(fill="both", expand=False, pady=(4, 0))
        ini_label_var = tk.StringVar(value="Select a column to edit its INI parameters.")
        ini_label = ttk.Label(ini_section, textvariable=ini_label_var, anchor="w")
        ini_label.pack(fill="x", padx=4, pady=(4, 2))
        preset_frame = ttk.Frame(ini_section)
        preset_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(preset_frame, text="Guided adjustments:").pack(side=tk.LEFT)
        for preset_label, preset_value in self.INI_ADJUSTMENT_PRESETS:
            btn = ttk.Button(
                preset_frame,
                text=preset_label,
                command=lambda v=preset_value, lbl=preset_label, key=config.key: self.apply_ini_preset(key, v, lbl),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self._add_tooltip(btn, f"Apply a {preset_label} change to the selected INI row.")
        update_btn = ttk.Button(
            preset_frame,
            text="Update Parameter",
            command=lambda key=config.key: self.update_selected_parameter(key),
        )
        update_btn.pack(side=tk.RIGHT)
        self._add_tooltip(
            update_btn,
            "Seed monotonic guesses (if enabled) and re-run fits for this column's files.",
        )
        ini_tree = ttk.Treeview(
            ini_section,
            columns=("FILE", "ALPHA", "PARAM", "MODE", "VALUE", "FINAL", "TREND"),
            show="headings",
            selectmode="browse",
            height=len(config.small_headers) * 2,
        )
        ini_tree.heading("FILE", text="File")
        ini_tree.heading("ALPHA", text="Alpha")
        ini_tree.heading("PARAM", text="Parameter")
        ini_tree.heading("MODE", text="Mode")
        ini_tree.heading("VALUE", text="Magnitude")
        ini_tree.heading("FINAL", text="Applied")
        ini_tree.heading("TREND", text="Trend")
        ini_tree.column("FILE", width=140, anchor="w")
        ini_tree.column("ALPHA", width=80, anchor="center")
        ini_tree.column("PARAM", width=110, anchor="center")
        ini_tree.column("MODE", width=160, anchor="center")
        ini_tree.column("VALUE", width=100, anchor="center")
        ini_tree.column("FINAL", width=100, anchor="center")
        ini_tree.column("TREND", width=70, anchor="center")
        ini_tree.pack(fill="both", expand=False, padx=4, pady=(0, 4))
        ini_tree.tag_configure(self.INI_TREND_DEVIATION_TAG, background=self.INI_TREND_COLOR)
        ini_tree.bind("<Double-1>", lambda event, key=config.key: self.start_ini_edit(key, event))

        # Large table
        large_frame = ttk.Frame(bottom_split)
        bottom_split.add(large_frame, weight=3)
        large_tree_container = ttk.Frame(large_frame)
        large_tree_container.pack(fill="both", expand=True)
        large_tree_scroll_x = ttk.Scrollbar(large_tree_container, orient="horizontal")
        large_tree_scroll_x.pack(fill="x", side=tk.BOTTOM)
        large_tree = ttk.Treeview(
            large_tree_container,
            columns=config.large_headers,
            show="headings",
            selectmode="browse",
            xscrollcommand=large_tree_scroll_x.set,
        )
        large_tree_scroll_x.config(command=large_tree.xview)
        for header in config.large_headers:
            large_tree.heading(header, text=header)
            large_tree.column(header, width=120, anchor="center", stretch=True)
        large_tree.pack(fill="both", expand=True)
        large_tree.bind(
            "<Configure>",
            lambda event, tree=large_tree: self._autosize_tree_columns(tree, min_width=90),
        )
        large_tree.bind("<Button-1>", lambda event, key=config.key: self.handle_column_click(event, key))

        # Summary
        summary_frame = ttk.Frame(bottom_split)
        bottom_split.add(summary_frame, weight=2)
        summary_tree_container = ttk.Frame(summary_frame)
        summary_tree_container.pack(fill="both", expand=True)
        summary_tree_scroll_x = ttk.Scrollbar(summary_tree_container, orient="horizontal")
        summary_tree_scroll_x.pack(fill="x", side=tk.BOTTOM)
        summary_label = ttk.Label(
            summary_frame,
            text='LSF / IDB / Δ VALUES OF "COLUMN" AT SPECIFIED ALPHAS',
            anchor="center",
        )
        summary_label.pack(fill="x")
        summary_tree = ttk.Treeview(
            summary_tree_container,
            columns=("ALPHA", "LSF", "IDB", "DELTA"),
            show="headings",
            height=8,
            xscrollcommand=summary_tree_scroll_x.set,
        )
        summary_tree_scroll_x.config(command=summary_tree.xview)
        for column in ("ALPHA", "LSF", "IDB", "DELTA"):
            summary_tree.heading(column, text=column)
            summary_tree.column(column, width=110, anchor="center")
        summary_tree.pack(fill="both", expand=True)
        summary_tree.bind(
            "<Configure>",
            lambda event, tree=summary_tree: self._autosize_tree_columns(tree, min_width=80),
        )

        column_index = {header: idx for idx, header in enumerate(config.large_headers)}
        return TabState(
            config=config,
            frame=container,
            figure=figure,
            canvas=canvas,
            axes=axes,
            large_tree=large_tree,
            small_tree=small_tree,
            summary_label=summary_label,
            summary_tree=summary_tree,
            small_rows=small_rows,
            column_index=column_index,
            large_overlays=[],
            small_overlays=[],
            ini_label_var=ini_label_var,
            ini_tree=ini_tree,
        )

    def _build_diagnostics_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="IDB vs ROP")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        side_panel = ttk.Frame(frame)
        side_panel.grid(row=0, column=0, sticky="nsw", padx=8, pady=8)
        ttk.Label(side_panel, text="Select Alpha:").pack(anchor="w")
        self.diagnostics_listbox = tk.Listbox(side_panel, height=12, exportselection=False)
        self.diagnostics_listbox.pack(fill="y", expand=True)
        self.diagnostics_listbox.bind("<<ListboxSelect>>", lambda _e: self.update_diagnostics_plot())
        self.diagnostics_quantity_var = tk.StringVar(value="EPS")
        ttk.Label(side_panel, text="Quantity:").pack(anchor="w", pady=(8, 0))
        for text, value in (("CEPS", "EPS"), ("CMU", "MUS")):
            ttk.Radiobutton(
                side_panel,
                text=text,
                value=value,
                variable=self.diagnostics_quantity_var,
                command=self.update_diagnostics_plot,
            ).pack(anchor="w")
        self.diagnostics_status_var = tk.StringVar(value="Load an LST file to view diagnostics.")
        ttk.Label(side_panel, textvariable=self.diagnostics_status_var, wraplength=180).pack(fill="x", pady=(8, 0))

        plot_frame = ttk.Frame(frame)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.diagnostics_figure = Figure(figsize=(5, 4), dpi=100)
        self.diagnostics_axes_real = self.diagnostics_figure.add_subplot(211)
        self.diagnostics_axes_imag = self.diagnostics_figure.add_subplot(212)
        self.diagnostics_axes_real.set_ylabel("Real")
        self.diagnostics_axes_imag.set_ylabel("Imag")
        self.diagnostics_axes_imag.set_xlabel("Frequency (GHz)")
        self.diagnostics_canvas = FigureCanvasTkAgg(self.diagnostics_figure, master=plot_frame)
        self.diagnostics_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.refresh_diagnostics_alpha_list()

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def load_lst_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select LST File",
            filetypes=[("LST Files", "*.LST"), ("All Files", "*.*")],
        )
        if not file_path:
            self.set_status("No file loaded.")
            return
        self._load_lst_from_path(Path(file_path))

    def _load_lst_from_path(self, lst_path: Path) -> None:
        alpha_pairs, idbkode = self.parse_lst_file(lst_path)
        if not alpha_pairs:
            self.set_status(f"{lst_path.name}: no valid entries found.")
            return
        self.idbkode = idbkode
        self.alpha_to_stem = {self.normalize_alpha(alpha): stem for alpha, stem in alpha_pairs}
        missing = self.populate_tables(alpha_pairs, lst_path.parent)
        missing_summary = ""
        if missing:
            missing_summary = f"; missing {len(missing)} file(s)"
        self.set_status(f"Loaded {lst_path.name} ({len(alpha_pairs)} alphas){missing_summary}")
        self.loaded_alpha_pairs = alpha_pairs
        self.loaded_base_path = lst_path.parent
        self.loaded_lst_name = lst_path.name
        self.loaded_lst_path = lst_path
        alpha_values: List[float] = []
        for alpha, _ in alpha_pairs:
            try:
                alpha_values.append(float(alpha))
            except (ValueError, TypeError):
                continue
        if alpha_values:
            self.alpha0 = sum(alpha_values) / len(alpha_values)
        else:
            self.alpha0 = None
        self.update_expected_extensions()
        self.alpha_to_files = {"EPS": {}, "MUS": {}}
        self.load_lsf_data(lst_path)
        self.update_plots()
        self.update_summary_tables()
        self.update_highlights()
        self.update_polynomial_models()
        for key in self.tabs:
            self.refresh_ini_table(key)
        self.refresh_diagnostics_alpha_list()
        self.create_file_snapshot()

    def parse_lst_file(self, path: Path) -> Tuple[List[Tuple[str, str]], Optional[int]]:
        try:
            lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        except OSError as exc:
            self.set_status(f"Unable to read {path.name}: {exc}")
            return [], None
        data_lines = [line for line in lines if not line.startswith("!")]
        if not data_lines:
            return [], None
        try:
            header_tokens = data_lines[0].split()
            count = int(header_tokens[0])
            idbk = int(header_tokens[1]) if len(header_tokens) > 1 else None
        except ValueError:
            return [], None
        expected = count * 2
        entries = data_lines[1:1 + expected]
        if len(entries) < expected:
            return [], None
        pairs: List[Tuple[str, str]] = []
        for idx in range(count):
            alpha = entries[2 * idx].split()[0]
            stem = entries[2 * idx + 1]
            pairs.append((alpha, stem))
        return pairs, idbk

    def _write_lst_file(self, alpha_pairs: List[Tuple[str, str]], idbk: Optional[int], path: Path) -> None:
        lines: List[str] = []
        header_idbk = idbk if idbk is not None else 0
        lines.append(f"{len(alpha_pairs)} {header_idbk}")
        for alpha, stem in alpha_pairs:
            lines.append("")
            lines.append(str(alpha).strip())
            lines.append(str(stem).strip())
        content = "\n".join(lines).strip() + "\n"
        path.write_text(content)

    def open_lst_editor(self) -> None:
        if not self.loaded_lst_path:
            messagebox.showerror("Edit LST", "Load an .LST file before editing.")
            return
        editor = tk.Toplevel(self.root)
        editor.title("Edit LST Entries")
        editor.geometry("400x420")
        entries = list(self.loaded_alpha_pairs)

        tree = ttk.Treeview(editor, columns=("ALPHA", "STEM"), show="headings", selectmode="browse")
        tree.heading("ALPHA", text="Alpha")
        tree.heading("STEM", text="Stem")
        tree.column("ALPHA", width=100, anchor="center")
        tree.column("STEM", width=220, anchor="w")
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        form = ttk.Frame(editor)
        form.pack(fill="x", padx=8)
        ttk.Label(form, text="Alpha:").grid(row=0, column=0, sticky="e")
        alpha_entry = ttk.Entry(form)
        alpha_entry.grid(row=0, column=1, sticky="we", padx=(4, 0))
        ttk.Label(form, text="Stem:").grid(row=1, column=0, sticky="e", pady=(4, 0))
        stem_entry = ttk.Entry(form)
        stem_entry.grid(row=1, column=1, sticky="we", padx=(4, 0), pady=(4, 0))
        form.columnconfigure(1, weight=1)

        def refresh_tree() -> None:
            tree.delete(*tree.get_children())
            for idx, (alpha, stem) in enumerate(entries):
                tree.insert("", "end", iid=str(idx), values=(alpha, stem))

        def clear_form() -> None:
            alpha_entry.delete(0, tk.END)
            stem_entry.delete(0, tk.END)

        def on_select(_event: tk.Event) -> None:
            selected = tree.selection()
            if not selected:
                return
            idx = int(selected[0])
            alpha, stem = entries[idx]
            clear_form()
            alpha_entry.insert(0, alpha)
            stem_entry.insert(0, stem)

        tree.bind("<<TreeviewSelect>>", on_select)

        button_row = ttk.Frame(editor)
        button_row.pack(fill="x", padx=8, pady=8)

        def add_entry() -> None:
            alpha = alpha_entry.get().strip()
            stem = stem_entry.get().strip()
            if not alpha or not stem:
                messagebox.showerror("Edit LST", "Alpha and stem are required.")
                return
            entries.append((alpha, stem))
            refresh_tree()
            clear_form()

        def update_entry() -> None:
            selected = tree.selection()
            if not selected:
                messagebox.showerror("Edit LST", "Select an entry to update.")
                return
            idx = int(selected[0])
            alpha = alpha_entry.get().strip()
            stem = stem_entry.get().strip()
            if not alpha or not stem:
                messagebox.showerror("Edit LST", "Alpha and stem are required.")
                return
            entries[idx] = (alpha, stem)
            refresh_tree()

        def delete_entry() -> None:
            selected = tree.selection()
            if not selected:
                messagebox.showerror("Edit LST", "Select an entry to delete.")
                return
            idx = int(selected[0])
            del entries[idx]
            refresh_tree()
            clear_form()

        ttk.Button(button_row, text="Add", command=add_entry).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Update", command=update_entry).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_row, text="Delete", command=delete_entry).pack(side=tk.LEFT)

        def save_changes() -> None:
            if not entries:
                messagebox.showerror("Edit LST", "At least one entry is required.")
                return
            try:
                self._write_lst_file(entries, self.idbkode, self.loaded_lst_path)
            except OSError as exc:
                messagebox.showerror("Edit LST", f"Unable to save LST: {exc}")
                return
            self._load_lst_from_path(self.loaded_lst_path)
            editor.destroy()

        ttk.Button(button_row, text="Save", command=save_changes).pack(side=tk.LEFT, padx=12)
        ttk.Button(button_row, text="Close", command=editor.destroy).pack(side=tk.LEFT)

        refresh_tree()

    def populate_tables(self, alpha_pairs: List[Tuple[str, str]], base_path: Path) -> List[str]:
        missing_files: List[str] = []
        for config in TAB_CONFIGS:
            state = self.tabs[config.key]
            state.large_tree.delete(*state.large_tree.get_children())
            self.alpha_to_files[config.key] = {}
            self.parameter_blocks[config.key] = {}
            self.alpha_file_paths[config.key] = {}
            extension = self.tab_extensions.get(config.key, config.file_extension)
            if not extension:
                continue
            for alpha, stem in alpha_pairs:
                file_path = self.resolve_stem(stem, base_path, extension)
                params: Dict[str, str] = {}
                if file_path.exists():
                    parsed = config.parser(file_path)
                    if parsed:
                        params = parsed
                        for target, aliases in config.value_aliases.items():
                            if target not in params:
                                for alias in aliases:
                                    if alias in params:
                                        params[target] = params[alias]
                                        break
                    else:
                        missing_files.append(file_path.name)
                else:
                    missing_files.append(file_path.name)
                display_name = file_path.name
                row = [alpha]
                for header in config.value_headers:
                    row.append(params.get(header, ""))
                row.append(display_name)
                alpha_key = self.normalize_alpha(alpha)
                self.alpha_to_files[config.key][alpha_key] = display_name
                self.alpha_file_paths[config.key][alpha_key] = file_path
                block = self.parse_parameter_block(file_path)
                if block:
                    self.parameter_blocks[config.key][alpha_key] = block
                state.large_tree.insert("", "end", values=row)
            self.refresh_ini_table(config.key)
        return missing_files

    def refresh_diagnostics_alpha_list(self) -> None:
        if not self.diagnostics_listbox:
            return
        self.diagnostics_listbox.delete(0, tk.END)
        if not self.loaded_alpha_pairs:
            self.diagnostics_status_var.set("Load an LST file to view diagnostics.")
            return
        for alpha, _ in self.loaded_alpha_pairs:
            self.diagnostics_listbox.insert(tk.END, alpha)
        self.diagnostics_status_var.set("Select an alpha and quantity to compare IDB vs ROP data.")

    def resolve_stem(self, stem: str, base_path: Path, extension: str) -> Path:
        candidate = Path(stem)
        suffix = extension if extension.startswith(".") else f".{extension}"
        if candidate.suffix.lower() != suffix.lower():
            candidate = candidate.with_suffix(suffix)
        if not candidate.is_absolute():
            candidate = base_path / candidate
        return candidate

    def handle_column_click(self, event: tk.Event, tab_key: str) -> None:
        tree = event.widget
        column_id = tree.identify_column(event.x)
        if not column_id or not column_id.startswith("#"):
            return "break"
        column_index = int(column_id[1:]) - 1
        columns = tree["columns"]
        if column_index >= len(columns):
            return "break"
        raw_header = tree.heading(columns[column_index], "text")
        header_text = self._normalize_header(raw_header)
        if (
            not header_text
            or header_text.upper().startswith("STEM")
            or header_text.upper() == "RMS"
            or header_text == "ROW"
        ):
            return "break"
        state = self.tabs[tab_key]
        large_index = state.column_index.get(header_text)
        if large_index is None:
            return "break"
        self.selection = ColumnSelection(tab_key=tab_key, header=header_text, large_index=large_index)
        self.update_highlights()
        self.update_plots()
        self.update_summary_tables()
        self.refresh_ini_table(tab_key)
        return "break"

    def refresh_ini_table(self, tab_key: str) -> None:
        state = self.tabs[tab_key]
        tree = state.ini_tree
        for entry in tree.get_children():
            tree.delete(entry)
        self.close_ini_editor(tab_key)
        selection = self.selection
        if not selection or selection.tab_key != tab_key:
            state.ini_label_var.set("Select a column in this tab to edit its INI parameter.")
            return
        param_header = state.config.ini_mapping.get(selection.header)
        if not param_header:
            state.ini_label_var.set(f"No INI parameter available for column {selection.header}.")
            return
        blocks = self.parameter_blocks.get(tab_key, {})
        if not blocks:
            state.ini_label_var.set("INI parameters unavailable for this tab.")
            return
        state.ini_label_var.set(f"{selection.header} controls for all files.")
        sorted_alphas = sorted(blocks.keys(), key=lambda a: self._safe_float(a))
        row_entries: List[Dict[str, Any]] = []
        for alpha_key in sorted_alphas:
            block = blocks[alpha_key]
            candidates = (param_header,) if isinstance(param_header, str) else tuple(param_header)
            header_name = next((header for header in candidates if header in block.headers), None)
            if not header_name:
                continue
            idx = block.headers.index(header_name)
            value = block.values[idx]
            mode = self._infer_control_mode(value)
            mode_label = self._control_label(mode)
            magnitude = self._entry_value_for_mode(value, mode)
            final_value = self._preview_value_for_mode(mode, magnitude, value) or value
            file_name = block.path.name if isinstance(block.path, Path) else str(block.path)
            row_entries.append(
                {
                    "file": file_name,
                    "alpha": alpha_key,
                    "param": header_name,
                    "mode": mode_label,
                    "magnitude": magnitude,
                    "final": final_value,
                    "numeric": self._try_parse_numeric(final_value),
                }
            )
        if not row_entries:
            return
        numeric_values = [entry["numeric"] for entry in row_entries]
        moving_average = self._moving_average_sequence(numeric_values, self.INI_TREND_MOVING_RADIUS)
        tolerance = self._trend_tolerance(numeric_values)
        for idx, entry in enumerate(row_entries):
            direction = self._moving_average_direction(moving_average, idx, tolerance)
            arrow = self.INI_TREND_ARROW_MAP.get(direction, "")
            tags = ()
            if self._is_trend_deviation(direction, numeric_values, idx, tolerance):
                tags = (self.INI_TREND_DEVIATION_TAG,)
            tree.insert(
                "",
                "end",
                values=(
                    entry["file"],
                    entry["alpha"],
                    entry["param"],
                    entry["mode"],
                    entry["magnitude"],
                    entry["final"],
                    arrow,
                ),
                tags=tags,
            )

    def start_ini_edit(self, tab_key: str, event: tk.Event) -> None:
        tree = self.tabs[tab_key].ini_tree
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        column = tree.identify_column(event.x)
        row_id = tree.identify_row(event.y)
        if not row_id:
            return
        alpha_key = tree.set(row_id, "ALPHA")
        parameter = tree.set(row_id, "PARAM")
        if column == "#4":
            self.start_ini_mode_editor(tab_key, tree, row_id, column, alpha_key, parameter)
        elif column == "#5":
            self.start_ini_value_editor(tab_key, tree, row_id, column, alpha_key, parameter)
        else:
            return

    def start_ini_mode_editor(
        self,
        tab_key: str,
        tree: ttk.Treeview,
        row_id: str,
        column: str,
        alpha_key: str,
        parameter: str,
    ) -> None:
        bbox = tree.bbox(row_id, column)
        if not bbox:
            return
        self.close_ini_editor(tab_key)
        combo = ttk.Combobox(tree, state="readonly", values=[label for _, label in self.CONTROL_OPTIONS])
        combo.set(tree.set(row_id, "MODE"))
        x, y, width, height = bbox
        combo.place(x=x, y=y, width=width, height=height)
        combo.focus()

        def commit(event=None) -> None:
            mode_label = combo.get()
            magnitude = tree.set(row_id, "VALUE")
            if self.commit_ini_change(tab_key, alpha_key, parameter, mode_label, magnitude):
                self.close_ini_editor(tab_key)
            else:
                combo.focus()

        combo.bind("<<ComboboxSelected>>", commit)
        combo.bind("<FocusOut>", lambda _e: self.close_ini_editor(tab_key))
        combo.bind("<Escape>", lambda _e: self.close_ini_editor(tab_key))
        self.ini_inline_widgets[tab_key] = combo

    def start_ini_value_editor(
        self,
        tab_key: str,
        tree: ttk.Treeview,
        row_id: str,
        column: str,
        alpha_key: str,
        parameter: str,
    ) -> None:
        bbox = tree.bbox(row_id, column)
        if not bbox:
            return
        current_value = tree.set(row_id, "VALUE")
        self.close_ini_editor(tab_key)
        entry = tk.Entry(tree)
        entry.insert(0, current_value)
        x, y, width, height = bbox
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus()

        def commit(event=None) -> None:
            new_value = entry.get().strip()
            mode_label = tree.set(row_id, "MODE")
            if self.commit_ini_change(tab_key, alpha_key, parameter, mode_label, new_value):
                self.close_ini_editor(tab_key)
            else:
                entry.focus()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", lambda _event: self.close_ini_editor(tab_key))
        entry.bind("<Escape>", lambda _event: self.close_ini_editor(tab_key))
        self.ini_inline_widgets[tab_key] = entry

    def close_ini_editor(self, tab_key: str) -> None:
        widget = self.ini_inline_widgets.get(tab_key)
        if widget is not None:
            widget.destroy()
        self.ini_inline_widgets[tab_key] = None

    def commit_ini_change(
        self,
        tab_key: str,
        alpha_key: str,
        parameter: str,
        mode_label: str,
        magnitude_text: str,
    ) -> bool:
        block = self.parameter_blocks.get(tab_key, {}).get(alpha_key)
        if not block:
            messagebox.showerror("INI Update", "Unable to locate parameter block for this selection.")
            return False
        try:
            index = block.headers.index(parameter)
        except ValueError:
            return False
        mode_code = self._mode_from_label(mode_label)
        magnitude = magnitude_text.strip()
        if mode_code != "estimate" and not magnitude:
            magnitude = "1"
        formatted = self._format_value_for_mode(parameter, mode_code, magnitude, block.values[index])
        if formatted is None:
            return False
        block.values[index] = formatted
        return self.apply_ini_changes(block, block.values)

    def apply_ini_preset(self, tab_key: str, percent: float, label: str) -> None:
        selection = self.selection
        if not selection or selection.tab_key != tab_key:
            messagebox.showinfo("Guided Adjustment", "Select a parameter column first.")
            return
        state = self.tabs[tab_key]
        tree = state.ini_tree
        self.close_ini_editor(tab_key)
        row_ids = tree.selection()
        if not row_ids:
            messagebox.showinfo("Guided Adjustment", "Select an INI row to adjust.")
            return
        row_id = row_ids[0]
        alpha_key = tree.set(row_id, "ALPHA")
        parameter = tree.set(row_id, "PARAM")
        mode_label = tree.set(row_id, "MODE")
        final_text = tree.set(row_id, "FINAL")
        if not alpha_key or not parameter:
            messagebox.showerror("Guided Adjustment", "Row is missing alpha or parameter information.")
            return
        final_value = self._try_parse_numeric(final_text)
        if final_value is None:
            messagebox.showerror("Guided Adjustment", "Unable to adjust non-numeric parameter values.")
            return
        mode_code = self._mode_from_label(mode_label)
        if mode_code == "estimate":
            messagebox.showinfo("Guided Adjustment", "Set this parameter to Fix or Negative before applying presets.")
            return
        change_factor = 1.0 + percent
        new_value = final_value * change_factor
        magnitude_value = abs(new_value)
        if magnitude_value <= 0:
            messagebox.showerror("Guided Adjustment", "Adjustment would zero-out this parameter. Choose a smaller change.")
            return
        magnitude_text = self._format_value(magnitude_value)
        if not magnitude_text.strip():
            magnitude_text = str(magnitude_value)
        if self.commit_ini_change(tab_key, alpha_key, parameter, mode_label, magnitude_text):
            self.set_status(f"Applied {label} preset to {parameter} at alpha {alpha_key}.")

    def adjust_power(self, tab_key: str, delta: int) -> None:
        if not self.selection or self.selection.tab_key != tab_key:
            return
        state = self.tabs[tab_key]
        column_id = self._column_id_for_header(state.small_tree, self.selection.header)
        row_id = state.small_rows.get("PWR")
        if not column_id or not row_id:
            return
        try:
            current = int(state.small_tree.set(row_id, column_id) or "0")
        except ValueError:
            current = 0
        new_value = max(0, current + delta)
        state.small_tree.set(row_id, column_id, str(new_value))
        self.update_polynomial_models()
        self.update_plots()
        self.update_summary_tables()
        self.update_highlights()

    def open_ini_editor(self) -> None:
        initialdir = str(self.loaded_base_path) if self.loaded_base_path else None
        file_path = filedialog.askopenfilename(
            title="Select Parameter File",
            filetypes=[("Parameter Files", "*.DBE *.2LM"), ("DBE Files", "*.DBE"), ("2LM Files", "*.2LM")],
            initialdir=initialdir,
        )
        if not file_path:
            return
        selected_path = Path(file_path)
        block = self.parse_parameter_block(selected_path)
        if not block:
            display_name = selected_path.name or str(selected_path)
            messagebox.showerror("Change INI", f'Could not find editable parameters in "{display_name}".')
            return
        self._show_ini_window(block)

    def parse_parameter_block(self, path: Path) -> Optional[ParameterBlock]:
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            self.set_status(f"Unable to read {path.name}: {exc}")
            return None
        marker_index = next((i for i, line in enumerate(lines) if "PARAMETER CONTROL" in line.upper()), None)
        if marker_index is None:
            return None
        header_index = self._find_parameter_header_index(lines, marker_index + 1)
        if header_index is None:
            return None
        value_index = self._find_parameter_value_index(lines, header_index + 1)
        if value_index is None:
            return None
        headers = lines[header_index].split()
        values = lines[value_index].split()
        if not headers or not values or len(values) < len(headers):
            return None
        return ParameterBlock(path=path, headers=headers, values=values, header_index=header_index, value_index=value_index, lines=lines)

    def _find_parameter_header_index(self, lines: List[str], start: int) -> Optional[int]:
        for idx in range(start, len(lines)):
            tokens = lines[idx].split()
            if not tokens:
                continue
            if all(self._token_has_digit(token) for token in tokens):
                return idx
        return None

    def _find_parameter_value_index(self, lines: List[str], start: int) -> Optional[int]:
        for idx in range(start, len(lines)):
            tokens = lines[idx].split()
            if not tokens:
                continue
            if all(self._is_numeric_token(token) for token in tokens):
                return idx
        return None

    @staticmethod
    def _token_has_digit(token: str) -> bool:
        return any(char.isdigit() for char in token)

    @staticmethod
    def _is_numeric_token(token: str) -> bool:
        cleaned = token.rstrip(",;")
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _control_label(self, mode: str) -> str:
        mapping = dict(self.CONTROL_OPTIONS)
        return mapping.get(mode, self.CONTROL_OPTIONS[0][1])

    def _mode_from_label(self, label: str) -> str:
        reverse = {text: code for code, text in self.CONTROL_OPTIONS}
        return reverse.get(label, "estimate")

    def _infer_control_mode(self, value_text: str) -> str:
        try:
            value = float(value_text)
        except (TypeError, ValueError):
            return "estimate"
        if abs(value) < 1e-12:
            return "estimate"
        if value < 0:
            return "negative"
        return "fix"

    def _entry_value_for_mode(self, value_text: str, mode: str) -> str:
        stripped = value_text.strip()
        if not stripped:
            return ""
        if mode == "estimate":
            return ""
        if stripped.startswith(("+", "-")):
            stripped = stripped[1:]
        return stripped

    def _on_control_mode_changed(
        self,
        entry: ttk.Entry,
        mode: str,
        original_value: str,
        preview_label: ttk.Label,
        mode_var: tk.StringVar,
    ) -> None:
        if mode == "estimate":
            entry.delete(0, tk.END)
            entry.state(["disabled"])
        else:
            if "disabled" in entry.state():
                default = self._entry_value_for_mode(original_value, mode) or "1"
                entry.delete(0, tk.END)
                entry.insert(0, default)
            entry.state(["!disabled"])
        self._update_preview_label(mode, entry.get().strip(), original_value, preview_label)

    def _update_preview_label(
        self,
        mode: str,
        entry_value: str,
        original_value: str,
        label: ttk.Label,
    ) -> None:
        preview = self._preview_value_for_mode(mode, entry_value, original_value)
        label.config(text=preview if preview is not None else "")

    def _preview_value_for_mode(
        self,
        mode: str,
        entry_value: str,
        original_value: str,
    ) -> Optional[str]:
        if mode == "estimate":
            return "0"
        text = entry_value.strip()
        if not text:
            text = self._entry_value_for_mode(original_value, mode) or ""
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric <= 0:
            return None
        normalized = text.lstrip("+")
        if mode == "negative":
            normalized = normalized.lstrip("-")
            if not normalized:
                normalized = "1"
            normalized = f"-{normalized}"
        return normalized

    def _format_value_for_mode(
        self,
        header: str,
        mode: str,
        entry_value: str,
        original_value: str,
    ) -> Optional[str]:
        preview = self._preview_value_for_mode(mode, entry_value, original_value)
        if preview is None:
            messagebox.showerror("Change INI", f"{header}: enter a valid positive number.")
            return None
        return preview

    def _show_ini_window(self, block: ParameterBlock) -> None:
        window = tk.Toplevel(self.root)
        window.title(f"Change INI – {block.path.name}")
        ttk.Label(
            window,
            text="0 = let computer estimate, >0 = keep parameter fixed, <0 = use negative value as initial guess.",
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 10))
        rows: List[ParameterEditorRow] = []
        for idx, header in enumerate(block.headers):
            value_text = block.values[idx] if idx < len(block.values) else ""
            ttk.Label(window, text=header).grid(row=idx + 1, column=0, sticky="e", padx=6, pady=3)
            mode = self._infer_control_mode(value_text)
            mode_var = tk.StringVar(value=self._control_label(mode))
            combo = ttk.Combobox(
                window,
                state="readonly",
                values=[label for _, label in self.CONTROL_OPTIONS],
                textvariable=mode_var,
                width=30,
            )
            combo.set(mode_var.get())
            combo.grid(row=idx + 1, column=1, sticky="w", padx=6, pady=3)
            entry = ttk.Entry(window, width=16)
            entry_value = self._entry_value_for_mode(value_text, mode)
            entry.insert(0, entry_value)
            if mode == "estimate":
                entry.state(["disabled"])
            preview = ttk.Label(window, text=value_text, width=16, anchor="w")
            preview.grid(row=idx + 1, column=3, sticky="w", padx=(6, 0))

            def combo_handler(evt, e=entry, var=mode_var, original=value_text, label=preview):
                self._on_control_mode_changed(e, self._mode_from_label(var.get()), original, label, var)

            combo.bind("<<ComboboxSelected>>", combo_handler)
            entry.bind(
                "<KeyRelease>",
                lambda _event, e=entry, var=mode_var, original=value_text, label=preview: self._update_preview_label(
                    self._mode_from_label(var.get()),
                    e.get().strip(),
                    original,
                    label,
                ),
            )
            entry.grid(row=idx + 1, column=2, sticky="w", padx=6, pady=3)
            rows.append(
                ParameterEditorRow(
                    header=header,
                    mode_var=mode_var,
                    value_entry=entry,
                    original_value=value_text,
                    preview_label=preview,
                )
            )

        def on_save() -> None:
            new_values: List[str] = []
            for row in rows:
                mode = self._mode_from_label(row.mode_var.get())
                value_text = row.value_entry.get().strip()
                formatted = self._format_value_for_mode(row.header, mode, value_text, row.original_value)
                if formatted is None:
                    return
                new_values.append(formatted)
            if self.apply_ini_changes(block, new_values):
                window.destroy()

        def on_cancel() -> None:
            window.destroy()

        button_frame = ttk.Frame(window)
        button_frame.grid(row=len(block.headers) + 1, column=0, columnspan=3, pady=(12, 6))
        ttk.Button(button_frame, text="Save", command=on_save).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=6)
        window.grab_set()
        window.transient(self.root)
        window.focus_set()

    def apply_ini_changes(self, block: ParameterBlock, values: List[str], *, reload_data: bool = True) -> bool:
        if len(values) != len(block.headers):
            messagebox.showerror("Change INI", "Parameter count mismatch.")
            return False
        lines = list(block.lines)
        template = lines[block.value_index] if block.value_index < len(lines) else ""
        new_line = self._format_ini_line(template, values)
        lines[block.value_index] = new_line
        try:
            block.path.write_text("\n".join(lines) + "\n")
        except OSError as exc:
            messagebox.showerror("Change INI", f"Unable to save changes: {exc}")
            return False
        block.lines = lines
        block.values = list(values)
        if reload_data:
            self.reload_after_ini_change(block.path)
        return True

    def _format_ini_line(self, template: str, values: List[str]) -> str:
        indent = len(template) - len(template.lstrip(" "))
        return (" " * max(indent, 0)) + "   ".join(values)

    def reload_after_ini_change(self, path: Path) -> None:
        if self.loaded_alpha_pairs and self.loaded_base_path:
            missing = self.populate_tables(self.loaded_alpha_pairs, self.loaded_base_path)
            status_msg = f"Updated INI in {path.name}; reloaded {self.loaded_lst_name or 'data'} (rerun LSF tool to refresh fit results)"
            if missing:
                status_msg += f"; missing {len(missing)} file(s)"
            self.set_status(status_msg)
            if self.loaded_lst_path:
                self.load_lsf_data(self.loaded_lst_path)
            self.update_expected_extensions()
            self.update_plots()
            self.update_summary_tables()
            self.update_polynomial_models()
            self.update_highlights()
            for key in self.tabs:
                self.refresh_ini_table(key)
        else:
            self.set_status(f"Updated INI in {path.name}. Load an LST file to view changes.")

    def load_lsf_data(self, lst_path: Path) -> None:
        self.lsf_data = {"EPS": {}, "MUS": {}}
        self.loaded_lsf_path = None
        candidate = self._find_lsf_path(lst_path)
        if not candidate:
            return
        try:
            lines = candidate.read_text().splitlines()
        except OSError as exc:
            self.set_status(f"Unable to read {candidate.name}: {exc}")
            return
        self.lsf_data = self.parse_lsf_lines(lines)
        self.loaded_lsf_path = candidate

    def _clear_snapshot_dir(self) -> None:
        if self.snapshot_dir and self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir, ignore_errors=True)
        self.snapshot_dir = None
        self.snapshot_map = {}

    def create_file_snapshot(self) -> None:
        if not self.loaded_alpha_pairs or not self.loaded_base_path:
            return
        self._clear_snapshot_dir()
        try:
            snapshot_root = Path(tempfile.mkdtemp(prefix="materials_snapshot_"))
        except OSError as exc:
            self.set_status(f"Unable to create snapshot: {exc}")
            return
        self.snapshot_dir = snapshot_root
        recorded_paths = set()
        for paths in self.alpha_file_paths.values():
            recorded_paths.update(paths.values())
        for original_path in recorded_paths:
            if not original_path.exists():
                continue
            try:
                relative_path = original_path.relative_to(self.loaded_base_path)
            except ValueError:
                relative_path = Path(original_path.name)
            destination = snapshot_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(original_path, destination)
            except OSError as exc:
                self.set_status(f"Snapshot failed for {original_path.name}: {exc}")
                continue
            self.snapshot_map[original_path] = destination

    def restore_original_files(self) -> None:
        if not self.snapshot_map:
            messagebox.showinfo("Restore Originals", "No snapshot available. Load an .LST file first.")
            return
        failures: List[str] = []
        for original_path, snapshot_path in self.snapshot_map.items():
            if not snapshot_path.exists():
                failures.append(f"Snapshot missing for {original_path.name}")
                continue
            try:
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot_path, original_path)
            except OSError as exc:
                failures.append(f"{original_path.name}: {exc}")
        if failures:
            messagebox.showerror("Restore Originals", "\n".join(failures[:10]) + ("\n..." if len(failures) > 10 else ""))
        else:
            self.set_status("Restored original parameter files.")
        if self.loaded_alpha_pairs and self.loaded_base_path:
            missing = self.populate_tables(self.loaded_alpha_pairs, self.loaded_base_path)
            if missing:
                messagebox.showwarning(
                    "Restore Originals",
                    "Unable to reload some files after restore:\n" + "\n".join(missing[:10]),
                )
            if self.loaded_lst_path:
                self.load_lsf_data(self.loaded_lst_path)
            self.update_plots()
            self.update_summary_tables()
            self.update_highlights()
            self.update_polynomial_models()
            for key in self.tabs:
                self.refresh_ini_table(key)

    def run_selected_tab_fits(self) -> None:
        if not self.notebook:
            return
        tab_id = self.notebook.select()
        if not tab_id:
            messagebox.showinfo("Run Fits", "Select a data tab before running fits.")
            return
        widget = self.root.nametowidget(tab_id)
        tab_key = self.tab_frame_to_key.get(widget)
        if not tab_key:
            messagebox.showinfo("Run Fits", "Select the EPS_DEBYE or MUS_DL tab, then try again.")
            return
        errors: List[str] = []
        successes = self.run_tab_fits(tab_key, silent=False, error_list=errors, reload=True, selected_pairs=None)
        if successes:
            self.set_status(f"[{tab_key}] Completed fits for {successes} file(s).")
        if errors:
            messagebox.showerror(
                "Run Fits",
                "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else ""),
            )

    def run_tab_fits(
        self,
        tab_key: str,
        silent: bool = False,
        error_list: Optional[List[str]] = None,
        reload: bool = True,
        selected_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> int:
        if not self.loaded_alpha_pairs or not self.loaded_base_path:
            if not silent:
                messagebox.showerror("Run Fits", "Load an .LST file before running fits.")
            return 0
        config = self.tabs[tab_key].config
        total = len(self.loaded_alpha_pairs)
        successes = 0
        local_errors: List[str] = []
        target_pairs = selected_pairs if selected_pairs else self.loaded_alpha_pairs
        for idx, (alpha_text, stem) in enumerate(target_pairs, start=1):
            alpha_key = self.normalize_alpha(alpha_text)
            measurement_path = self._measurement_path_for_stem(stem, self.loaded_base_path)
            if not measurement_path:
                local_errors.append(f"{stem}: measurement file (.ROP/.EMU) not found")
                continue
            try:
                rop_data = read_material_file(measurement_path)
            except Exception as exc:  # noqa: BLE001
                local_errors.append(f"{stem}: unable to read measurement data: {exc}")
                continue
            output_path = self.alpha_file_paths.get(tab_key, {}).get(alpha_key)
            if not output_path:
                extension = self.tab_extensions.get(tab_key, config.file_extension)
                output_path = self.resolve_stem(stem, self.loaded_base_path, extension)
            try:
                if tab_key == "EPS":
                    controls = self._build_debye_controls(alpha_key)
                    result = fit_debye(rop_data, controls)
                    report = generate_dbe_report(
                        rop_data,
                        controls,
                        result,
                        input_file=measurement_path,
                        output_file=output_path,
                    )
                else:
                    controls = self._build_double_lorentz_controls(alpha_key)
                    result = fit_double_lorentz(rop_data, "mu", controls)
                    report = generate_dlm_report(
                        rop_data,
                        "mu",
                        controls,
                        result,
                        input_file=measurement_path,
                        output_file=output_path,
                    )
                output_path.write_text(report)
                successes += 1
                if not silent:
                    self.set_status(
                        f"[{tab_key}] Fitted {stem} ({idx}/{total}); RMS={result.rms:.4f}"
                    )
                    self.root.update_idletasks()
                if result.iflag != 0:
                    local_errors.append(f"{stem}: solver reached iteration limit")
            except Exception as exc:  # noqa: BLE001
                local_errors.append(f"{stem}: {exc}")
                continue
        if successes and reload:
            missing = self.populate_tables(self.loaded_alpha_pairs, self.loaded_base_path)
            if missing:
                local_errors.extend(f"{name}: could not reload" for name in missing)
            if self.loaded_lst_path:
                self.load_lsf_data(self.loaded_lst_path)
            self.update_plots()
            self.update_summary_tables()
            self.update_highlights()
            self.update_polynomial_models()
            for key in self.tabs:
                self.refresh_ini_table(key)
            if not silent:
                self.set_status(f"[{tab_key}] Updated {successes} file(s).")
        if local_errors:
            if error_list is not None:
                error_list.extend(local_errors)
            elif not silent:
                messagebox.showerror(
                    "Run Fits",
                    "\n".join(local_errors[:10]) + ("\n..." if len(local_errors) > 10 else ""),
                )
        return successes

    def update_expected_extensions(self) -> None:
        ceps_ext = self.tab_extensions.get("EPS", ".DBE")
        cmu_ext = self.tab_extensions.get("MUS", ".2LM")
        if self.idbkode is None:
            self.tab_extensions["EPS"] = ceps_ext
            self.tab_extensions["MUS"] = cmu_ext
            return
        if self.idbkode < 10:
            mapping = {
                0: (".DBE", None),
                1: (".LNE", None),
                2: (".DBE", ".DBM"),
                3: (".LNE", ".DBM"),
                4: (".DBE", ".LNM"),
                5: (".LNE", ".LNM"),
            }
            ceps_ext, cmu_ext = mapping.get(self.idbkode, (ceps_ext, cmu_ext))
        else:
            ce_map = {
                0: None,
                1: ".DBE",
                2: ".LNE",
                3: ".2DE",
                4: ".2LE",
                5: ".3DE",
                6: ".DLE",
            }
            cm_map = {
                0: None,
                1: ".DBM",
                2: ".LNM",
                3: ".2DM",
                4: ".2LM",
                5: ".3DM",
                6: ".DLM",
            }
            idbke = self.idbkode // 10
            idbkm = self.idbkode % 10
            ceps_ext = ce_map.get(idbke, ceps_ext)
            cmu_ext = cm_map.get(idbkm, cmu_ext)
        self.tab_extensions["EPS"] = ceps_ext
        self.tab_extensions["MUS"] = cmu_ext

    def _find_lsf_path(self, lst_path: Path) -> Optional[Path]:
        candidates = [
            lst_path.with_suffix(".LSF"),
            lst_path.with_suffix(".lsf"),
        ]
        for stem in ("Material", "material"):
            candidates.append(lst_path.parent / f"{stem}.LSF")
            candidates.append(lst_path.parent / f"{stem}.lsf")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def parse_lsf_lines(self, lines: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
        data: Dict[str, Dict[str, Dict[str, str]]] = {"EPS": {}, "MUS": {}}
        current: Optional[str] = None
        reading = False
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("LSF CEPS DATA"):
                current = "EPS"
                reading = False
                continue
            if upper.startswith("LSF CMU DATA"):
                current = "MUS"
                reading = False
                continue
            if not current:
                continue
            if stripped.startswith("!"):
                if "ALPHA" in upper and "LSF" in upper:
                    reading = True
                continue
            if reading:
                if not stripped:
                    reading = False
                    continue
                if upper.startswith("LSF ") or upper.startswith("DLPAR") or upper.startswith("IX,"):
                    reading = False
                    continue
                tokens = stripped.split()
                if len(tokens) < 3:
                    continue
                try:
                    alpha_value = float(tokens[0])
                except ValueError:
                    reading = False
                    continue
                lsf_file = tokens[-1]
                values = tokens[1:-1]
                normalized_alpha = self.normalize_alpha(alpha_value)
                data[current][normalized_alpha] = {
                    "file": lsf_file,
                    "values": values,
                }
        return data

    def update_highlights(self) -> None:
        for state in self.tabs.values():
            self._clear_column_overlays(state.large_overlays)
            self._clear_column_overlays(state.small_overlays)
        if not self.selection:
            return
        selected_state = self.tabs[self.selection.tab_key]
        self._highlight_column_cells(selected_state.large_tree, self.selection.header, selected_state.large_overlays)
        self._highlight_column_cells(selected_state.small_tree, self.selection.header, selected_state.small_overlays)

    def _column_id_for_header(self, tree: ttk.Treeview, header_text: str) -> Optional[str]:
        for column_id in tree["columns"]:
            text = tree.heading(column_id, "text")
            if self._normalize_header(text) == header_text:
                return column_id
        return None

    def _clear_column_overlays(self, overlays: List[tk.Widget]) -> None:
        while overlays:
            overlay = overlays.pop()
            overlay.destroy()

    def _highlight_column_cells(
        self,
        tree: ttk.Treeview,
        header_text: str,
        overlays: List[tk.Widget],
    ) -> None:
        column_id = self._column_id_for_header(tree, header_text)
        if not column_id:
            return
        for item in tree.get_children():
            bbox = tree.bbox(item, column=column_id)
            if not bbox:
                continue
            x, y, width, height = bbox
            value = tree.set(item, column_id)
            overlay = tk.Label(
                tree,
                text=value,
                bg="#ffe28a",
                fg="#000000",
                borderwidth=0,
                font=("TkDefaultFont", 10),
                anchor="center",
            )
            overlay.place(x=x, y=y, width=width, height=height)
            overlay.bind(
                "<Button-1>",
                lambda event, t=tree, w=overlay: self._forward_tree_event(t, w, "<Button-1>", event),
            )
            overlay.bind(
                "<ButtonRelease-1>",
                lambda event, t=tree, w=overlay: self._forward_tree_event(t, w, "<ButtonRelease-1>", event),
            )
            overlays.append(overlay)

    def _forward_tree_event(self, tree: ttk.Treeview, widget: tk.Widget, sequence: str, event: tk.Event) -> None:
        tree.event_generate(
            sequence,
            x=event.x + widget.winfo_x(),
            y=event.y + widget.winfo_y(),
        )

    def update_diagnostics_plot(self) -> None:
        if not self.diagnostics_listbox:
            return
        selection = self.diagnostics_listbox.curselection()
        self.diagnostics_axes_real.clear()
        self.diagnostics_axes_imag.clear()
        self.diagnostics_axes_real.set_ylabel("Real")
        self.diagnostics_axes_imag.set_ylabel("Imag")
        self.diagnostics_axes_imag.set_xlabel("Frequency (GHz)")
        if not selection or not self.loaded_alpha_pairs or not self.loaded_base_path:
            self.diagnostics_axes_real.text(0.5, 0.5, "Select an entry", ha="center", va="center")
            self.diagnostics_axes_imag.text(0.5, 0.5, "", ha="center", va="center")
            self.diagnostics_canvas.draw()
            return
        alpha_text = self.diagnostics_listbox.get(selection[0])
        alpha_key = self.normalize_alpha(alpha_text)
        stem = self.alpha_to_stem.get(alpha_key)
        if not stem:
            self.diagnostics_status_var.set(f"No stem found for alpha {alpha_text}.")
            self.diagnostics_canvas.draw()
            return
        measurement_path = self._measurement_path_for_stem(stem, self.loaded_base_path)
        if not measurement_path:
            self.diagnostics_status_var.set(f"Measurement file missing for {stem}.")
            self.diagnostics_canvas.draw()
            return
        try:
            rop_data = read_material_file(measurement_path)
        except Exception as exc:  # noqa: BLE001
            self.diagnostics_status_var.set(f"Unable to read {stem}: {exc}")
            self.diagnostics_canvas.draw()
            return
        self.update_polynomial_models()
        quantity = self.diagnostics_quantity_var.get() if self.diagnostics_quantity_var else "EPS"
        alpha_float = self._safe_float(alpha_text)
        if quantity == "EPS":
            measured_real = rop_data.eps_real
            measured_imag = rop_data.eps_imag
        else:
            measured_real = rop_data.mu_real
            measured_imag = rop_data.mu_imag
        freq = rop_data.frequencies_ghz
        prediction = self._compute_idb_prediction(alpha_float, freq, quantity)
        if prediction is None:
            self.diagnostics_status_var.set(
                "Polynomial models not available for the selected parameters."
            )
            self.diagnostics_canvas.draw()
            return
        fitted_real, fitted_imag = prediction
        self.diagnostics_axes_real.plot(freq, measured_real, label="Measured", color="#555555")
        self.diagnostics_axes_real.plot(freq, fitted_real, label="IDB", color="#d9534f")
        self.diagnostics_axes_imag.plot(freq, measured_imag, label="Measured", color="#555555")
        self.diagnostics_axes_imag.plot(freq, fitted_imag, label="IDB", color="#0275d8")
        self.diagnostics_axes_real.legend(loc="best")
        self.diagnostics_axes_imag.legend(loc="best")
        self.diagnostics_axes_real.set_title(f"Alpha {alpha_text} ({quantity})")
        self.diagnostics_status_var.set(
            f"Comparing measurement vs IDB prediction for {stem} ({quantity})."
        )
        self.diagnostics_canvas.draw()

    def _compute_idb_prediction(
        self,
        alpha_value: float,
        frequencies: List[float],
        tab_key: str,
    ) -> Optional[Tuple[List[float], List[float]]]:
        if tab_key == "EPS":
            headers = ("FRES", "DEPS", "EPSV", "GAMMA", "SIGE")
            params = {}
            for header in headers:
                value = self.evaluate_polynomial("EPS", header, alpha_value)
                if value is None:
                    return None
                params[header] = value
            def compute(freq: float) -> complex:
                return compute_debye_ceps(
                    freq,
                    params["FRES"],
                    params["DEPS"],
                    params["EPSV"],
                    params["GAMMA"],
                    params["SIGE"],
                )
        else:
            headers = (
                "FR_M1",
                "DMUR1",
                "GAMM1",
                "FR_M2",
                "DMUR2",
                "GAMM2",
                "MURV",
                "EPSV",
            )
            params = {}
            for header in headers:
                value = self.evaluate_polynomial("MUS", header, alpha_value)
                if value is None:
                    return None
                params[header] = value
            def compute(freq: float) -> complex:
                return compute_debye_lorentz_ceps(
                    freq,
                    params["FR_M1"],
                    params["DMUR1"],
                    params["GAMM1"],
                    params["FR_M2"],
                    params["DMUR2"],
                    params["GAMM2"],
                    params["MURV"],
                    params["EPSV"],
                )
        real_vals = []
        imag_vals = []
        for freq in frequencies:
            value = compute(freq)
            real_vals.append(value.real)
            imag_vals.append(value.imag)
        return real_vals, imag_vals

    def update_plots(self) -> None:
        for state in self.tabs.values():
            self._update_plot_for_state(state)

    def _update_plot_for_state(self, state: TabState) -> None:
        state.axes.clear()
        state.axes.set_xlabel("Alpha")
        state.axes.set_ylabel("Value")
        if not self.selection:
            state.axes.set_title("Select a column to plot")
            state.canvas.draw()
            return
        column_index = state.column_index.get(self.selection.header)
        if column_index is None:
            state.axes.set_title(f"{self.selection.header} not available in {state.config.key}")
            state.canvas.draw()
            return
        x_vals: List[float] = []
        y_vals: List[float] = []
        y_fit_vals: List[Optional[float]] = []
        for item in state.large_tree.get_children():
            values = state.large_tree.item(item, "values")
            if not values:
                continue
            alpha_val = self._safe_float(values[0])
            x_vals.append(alpha_val)
            raw_value = values[column_index] if len(values) > column_index else ""
            y_vals.append(self._safe_float(raw_value))
            y_fit_vals.append(self.evaluate_polynomial(state.config.key, self.selection.header, alpha_val))
        state.axes.plot(x_vals, y_vals, marker="x", linestyle="None", color=state.config.plot_color, label="LSF Data")
        model = self.polynomial_models.get(state.config.key, {}).get(self.selection.header)
        if x_vals and model:
            alpha_min, alpha_max = min(x_vals), max(x_vals)
            sample_count = max(50, len(x_vals) * 10)
            alpha_line = np.linspace(alpha_min, alpha_max, sample_count)
            fit_line = [
                self.evaluate_polynomial(state.config.key, self.selection.header, float(alpha))
                for alpha in alpha_line
            ]
            state.axes.plot(alpha_line, fit_line, color="black", linewidth=1.5, label=f"Polynomial (PWR {model['power']})")
        rms_value = model["rms"] if model else None
        rms_text = f"{rms_value:.4g}" if rms_value is not None else "N/A"
        state.axes.set_title(f"{self.selection.header} vs. Alpha (PWR={model['power'] if model else 'N/A'}, RMS={rms_text})")
        state.axes.legend(loc="best")
        state.canvas.draw()

    def get_small_table_value(self, state: TabState, row_label: str, header_text: str) -> str:
        row_id = state.small_rows.get(row_label)
        column_id = self._column_id_for_header(state.small_tree, header_text)
        if not row_id or not column_id:
            return ""
        return state.small_tree.set(row_id, column_id) or ""

    def update_summary_tables(self) -> None:
        for state in self.tabs.values():
            self._update_summary_for_state(state)

    def auto_converge(self) -> None:
        if not self.loaded_alpha_pairs or not self.loaded_base_path:
            messagebox.showerror("Auto Converge", "Load an .LST file and associated measurement files first.")
            return
        if not self.selection:
            messagebox.showinfo("Auto Converge", "Select a column to auto-converge.")
            return
        tab_key = self.selection.tab_key
        settings = self._prompt_auto_converge_settings()
        if not settings:
            return
        self._perform_auto_converge(tab_key, self.selection.header, settings)

    def _prompt_auto_converge_settings(self) -> Optional[AutoConvergeSettings]:
        window = tk.Toplevel(self.root)
        window.title("Auto Converge Settings")
        ttk.Label(window, text="Target RMS").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Label(window, text="Max Iterations").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        ttk.Label(window, text="Convergence Factor (0-1)").grid(row=2, column=0, sticky="e", padx=6, pady=4)
        target_var = tk.StringVar(value="0.01")
        iterations_var = tk.StringVar(value="5")
        factor_var = tk.StringVar(value="0.35")
        ttk.Entry(window, textvariable=target_var, width=12).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Entry(window, textvariable=iterations_var, width=12).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Entry(window, textvariable=factor_var, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        result: Dict[str, Any] = {}

        def submit() -> None:
            try:
                target = float(target_var.get())
                max_iterations = int(iterations_var.get())
                factor = float(factor_var.get())
            except ValueError:
                messagebox.showerror("Auto Converge", "Enter valid numeric values for all fields.")
                return
            if target <= 0 or max_iterations <= 0:
                messagebox.showerror("Auto Converge", "Target RMS and iterations must be positive.")
                return
            factor = max(0.0, min(1.0, factor))
            min_improvement = max(target * 0.05, 1e-4)
            result["settings"] = AutoConvergeSettings(target, max_iterations, factor, min_improvement)
            window.destroy()

        def cancel() -> None:
            window.destroy()

        button_frame = ttk.Frame(window)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(6, 8))
        ttk.Button(button_frame, text="Start", command=submit).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=4)
        window.grab_set()
        self.root.wait_window(window)
        return result.get("settings")

    def _perform_auto_converge(self, tab_key: str, header: str, settings: AutoConvergeSettings) -> None:
        if not self._ensure_parameter_data(tab_key):
            return
        entries = self._gather_parameter_entries(tab_key, header)
        if not entries:
            messagebox.showerror("Auto Converge", f"No INI entries found for {header} in this tab.")
            return
        alpha_keys = [entry["alpha_key"] for entry in entries]
        rms_value = self._compute_rms_statistic(tab_key, alpha_keys)
        if rms_value is None:
            messagebox.showerror("Auto Converge", "Unable to determine RMS values for this parameter.")
            return
        if rms_value <= settings.target_rms:
            messagebox.showinfo(
                "Auto Converge",
                f"Current RMS ({rms_value:.4g}) already meets the target {settings.target_rms:.4g}.",
            )
            return
        prev_rms = rms_value
        iteration = 0
        while iteration < settings.max_iterations:
            iteration += 1
            if not self._blend_ini_values_toward_targets(entries, settings.convergence_factor):
                if iteration == 1:
                    messagebox.showinfo("Auto Converge", "Unable to adjust INI entries for this column.")
                break
            self._seed_monotonic_initial_guesses(tab_key, header)
            pairs = self._pairs_for_entries(tab_key, entries)
            if not pairs:
                messagebox.showerror("Auto Converge", "Unable to locate files for the selected parameter.")
                return
            errors: List[str] = []
            successes = self.run_tab_fits(tab_key, silent=False, error_list=errors, reload=True, selected_pairs=pairs)
            if errors:
                messagebox.showerror(
                    "Auto Converge",
                    "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else ""),
                )
                return
            if successes == 0:
                break
            entries = self._gather_parameter_entries(tab_key, header)
            alpha_keys = [entry["alpha_key"] for entry in entries]
            rms_value = self._compute_rms_statistic(tab_key, alpha_keys)
            if rms_value is None:
                break
            if rms_value <= settings.target_rms:
                self.set_status(
                    f"Auto-converged {header} to RMS {rms_value:.4g} in {iteration} iteration(s)."
                )
                messagebox.showinfo(
                    "Auto Converge",
                    f"Reached RMS {rms_value:.4g} after {iteration} iteration(s).",
                )
                return
            if prev_rms - rms_value < settings.min_improvement:
                break
            prev_rms = rms_value
        if rms_value is not None:
            messagebox.showinfo(
                "Auto Converge",
                f"Stopped after {iteration} iteration(s). Latest RMS: {rms_value:.4g} (target {settings.target_rms:.4g}).",
            )

    def _pairs_for_entries(self, tab_key: str, entries: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        alpha_map = self.alpha_to_files.get(tab_key, {})
        pairs: List[Tuple[str, str]] = []
        for entry in entries:
            display_name = entry.get("file_name") or alpha_map.get(entry["alpha_key"], "")
            if not display_name:
                continue
            pairs.append((entry["alpha_text"], display_name))
        return pairs

    def _blend_ini_values_toward_targets(self, entries: List[Dict[str, Any]], factor: float) -> bool:
        factor = max(0.0, min(1.0, factor))
        updated_blocks: Dict[Path, ParameterBlock] = {}
        changed = False
        for entry in entries:
            block = entry["block"]
            idx = entry["index"]
            target_value = entry.get("lsf_value")
            if target_value is None:
                continue
            current_text = block.values[idx]
            mode = self._infer_control_mode(current_text)
            current_numeric = self._try_parse_numeric(current_text)
            if mode == "estimate":
                mode = "negative"
                current_magnitude = abs(target_value)
            else:
                current_magnitude = abs(current_numeric) if current_numeric is not None else abs(target_value)
            target_magnitude = max(abs(target_value), 1e-9)
            new_magnitude = current_magnitude + (target_magnitude - current_magnitude) * factor
            new_magnitude = max(new_magnitude, 1e-9)
            formatted = f"{new_magnitude:.6g}"
            if mode == "negative":
                formatted = f"-{formatted.lstrip('+').lstrip('-')}"
            if formatted == current_text:
                continue
            block.values[idx] = formatted
            updated_blocks[block.path] = block
            changed = True
        if not changed:
            return False
        for block in updated_blocks.values():
            if not self.apply_ini_changes(block, block.values, reload_data=False):
                return False
        return True

    def _compute_rms_statistic(self, tab_key: str, alpha_keys: List[str]) -> Optional[float]:
        state = self.tabs.get(tab_key)
        if not state:
            return None
        rms_index = state.column_index.get("RMS")
        if rms_index is None:
            return None
        lookup = set(alpha_keys)
        values: List[float] = []
        for item in state.large_tree.get_children():
            row = state.large_tree.item(item, "values")
            if not row:
                continue
            alpha_key = self.normalize_alpha(row[0])
            if alpha_key not in lookup:
                continue
            if rms_index >= len(row):
                continue
            numeric = self._try_parse_numeric(row[rms_index])
            if numeric is not None:
                values.append(numeric)
        if not values:
            return None
        return max(values)

    def _ensure_parameter_data(self, tab_key: str) -> bool:
        blocks = self.parameter_blocks.get(tab_key, {})
        if any(blocks.values()):
            return True
        errors: List[str] = []
        successes = self.run_tab_fits(
            tab_key,
            silent=False,
            error_list=errors,
            reload=True,
            selected_pairs=None,
        )
        if successes:
            return True
        if errors:
            messagebox.showerror(
                "Auto Converge",
                "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else ""),
            )
        else:
            messagebox.showerror("Auto Converge", "Unable to generate parameter files for this tab.")
        return False

    @staticmethod
    def _autosize_tree_columns(tree: ttk.Treeview, min_width: int = 80) -> None:
        columns = tree["columns"]
        if not columns:
            return
        total_width = tree.winfo_width()
        if total_width <= 0:
            return
        width_per_column = max(min_width, total_width // len(columns))
        for column in columns:
            tree.column(column, width=width_per_column)

    def _update_summary_for_state(self, state: TabState) -> None:
        state.summary_tree.delete(*state.summary_tree.get_children())
        if not self.selection:
            state.summary_label.config(text='LSF / IDB / Δ VALUES OF "COLUMN" AT SPECIFIED ALPHAS')
            return
        column_index = state.column_index.get(self.selection.header)
        if column_index is None:
            state.summary_label.config(text=f"{self.selection.header} not found for {state.config.key}")
            return
        state.summary_label.config(
            text=f'LSF / IDB / Δ VALUES OF {self.selection.header} AT SPECIFIED ALPHAS'
        )
        models_for_tab = self.polynomial_models.get(state.config.key, {})
        for item in state.large_tree.get_children():
            values = state.large_tree.item(item, "values")
            if not values:
                continue
            alpha_text = values[0]
            alpha_val = self._safe_float(alpha_text)
            lsf_val: Optional[float] = None
            if len(values) > column_index:
                try:
                    lsf_val = float(values[column_index])
                except (TypeError, ValueError):
                    lsf_val = None
            idb_val = self.evaluate_polynomial(state.config.key, self.selection.header, alpha_val)
            delta_val = lsf_val - idb_val if (lsf_val is not None and idb_val is not None) else None
            state.summary_tree.insert(
                "",
                "end",
                values=(
                    self._format_value(alpha_val),
                    self._format_value(lsf_val),
                    self._format_value(idb_val),
                    self._format_value(delta_val),
                ),
            )

    def update_polynomial_models(self) -> None:
        for key, state in self.tabs.items():
            self.polynomial_models[key] = self._compute_polynomial_models(state)

    def _compute_polynomial_models(self, state: TabState) -> Dict[str, Dict[str, Any]]:
        models: Dict[str, Dict[str, Any]] = {}
        power_row_id = state.small_rows.get("PWR")
        rms_row_id = state.small_rows.get("RMS")
        if rms_row_id:
            for header in state.config.polynomial_headers:
                column_id = self._column_id_for_header(state.small_tree, header)
                if column_id:
                    state.small_tree.set(rms_row_id, column_id, "")
        alpha_vals: List[float] = []
        per_header_values: Dict[str, List[float]] = {header: [] for header in state.config.polynomial_headers}
        for item in state.large_tree.get_children():
            row = state.large_tree.item(item, "values")
            if not row:
                continue
            alpha_val = self._safe_float(row[0])
            alpha_vals.append(alpha_val)
            for header in state.config.polynomial_headers:
                col_idx = state.column_index.get(header)
                if col_idx is None or col_idx >= len(row):
                    continue
                per_header_values.setdefault(header, []).append(self._safe_float(row[col_idx]))
        if not alpha_vals or self.alpha0 is None:
            return models
        centered_alphas = np.array([val - self.alpha0 for val in alpha_vals], dtype=float)
        for header in state.config.polynomial_headers:
            values = per_header_values.get(header, [])
            if len(values) != len(alpha_vals) or not values:
                continue
            power = self._get_power_for_header(state, header, power_row_id)
            max_power = min(power, len(alpha_vals) - 1)
            coeffs, rms = self._fit_polynomial(centered_alphas, np.array(values, dtype=float), max_power)
            if coeffs is None:
                continue
            models[header] = {"power": max_power, "coeffs": coeffs, "rms": rms}
            if rms_row_id:
                column_id = self._column_id_for_header(state.small_tree, header)
                if column_id:
                    state.small_tree.set(rms_row_id, column_id, f"{rms:.4f}")
        return models

    def _get_power_for_header(self, state: TabState, header: str, row_id: Optional[str]) -> int:
        if not row_id:
            return 0
        column_id = self._column_id_for_header(state.small_tree, header)
        if not column_id:
            return 0
        try:
            return max(0, int(state.small_tree.set(row_id, column_id) or 0))
        except ValueError:
            return 0

    def _fit_polynomial(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        degree: int,
    ) -> Tuple[Optional[List[float]], float]:
        if degree < 0:
            return None, 0.0
        degree = min(degree, len(x_vals) - 1)
        if degree < 0:
            return None, 0.0
        vandermonde = np.vander(x_vals, N=degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(vandermonde, y_vals, rcond=None)
        residuals = y_vals - vandermonde.dot(coeffs)
        rms = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else 0.0
        return coeffs.tolist(), rms

    def evaluate_polynomial(self, tab_key: str, header: str, alpha_value: float) -> Optional[float]:
        if self.alpha0 is None:
            return None
        models_for_tab = self.polynomial_models.get(tab_key)
        if not models_for_tab:
            return None
        model = models_for_tab.get(header)
        if not model:
            return None
        x = alpha_value - self.alpha0
        result = 0.0
        for idx, coeff in enumerate(model["coeffs"]):
            result += coeff * (x ** idx)
        return result

    def generate_idb_file(self) -> None:
        if not self.loaded_lst_name or not self.loaded_base_path:
            messagebox.showerror("Generate IDB", "Load an .LST file before generating an IDB.")
            return
        if not self.loaded_alpha_pairs:
            messagebox.showerror("Generate IDB", "No alpha entries available.")
            return
        if self.alpha0 is None:
            messagebox.showerror("Generate IDB", "Unable to determine ALPHA0.")
            return
        if self.idbkode is None:
            messagebox.showerror("Generate IDB", "IDBKODE not found in the .LST file.")
            return
        self.update_polynomial_models()
        sections: List[Tuple[str, TabConfig]] = []
        for tab_key in ("EPS", "MUS"):
            extension = self.tab_extensions.get(tab_key)
            if extension:
                sections.append((tab_key, self.tabs[tab_key].config))
        if not sections:
            messagebox.showerror("Generate IDB", "No CEPS or CMU data available to generate an IDB file.")
            return
        for tab_key, config in sections:
            models = self.polynomial_models.get(tab_key, {})
            for header in config.polynomial_headers:
                if header not in models:
                    messagebox.showerror(
                        "Generate IDB",
                        f"Polynomial for {tab_key} parameter {header} is not defined. Adjust PWR values and try again.",
                    )
                    return
        timestamp = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        idb_path = self.loaded_base_path / Path(self.loaded_lst_name).with_suffix(".IDB").name
        lines: List[str] = [
            f"! Generated by MaterialGui on {timestamp}",
            f"! Source LST: {self.loaded_lst_name}",
            "",
            f"  {self.alpha0: .7E}  {self.idbkode}",
            "",
        ]
        for tab_key, config in sections:
            lines.append(f"! {config.notebook_label} parameters ({self.tab_extensions.get(tab_key)})")
            models = self.polynomial_models.get(tab_key, {})
            for header in config.polynomial_headers:
                model = models[header]
                lines.append(f"  {model['power']}")
                coeff_line = "  " + " ".join(f"{coeff: .7E}" for coeff in model["coeffs"])
                lines.append(coeff_line)
                lines.append("")
        try:
            idb_path.write_text("\n".join(lines).rstrip() + "\n")
        except OSError as exc:
            messagebox.showerror("Generate IDB", f"Unable to write {idb_path.name}: {exc}")
            return
        self.set_status(f"Generated {idb_path.name}")

    @staticmethod
    def _safe_float(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _try_parse_numeric(value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("D", "E").replace("d", "e")
        try:
            return float(normalized)
        except ValueError:
            return None

    def _moving_average_sequence(
        self,
        values: List[Optional[float]],
        radius: int,
    ) -> List[Optional[float]]:
        if radius <= 0:
            return values[:]
        averages: List[Optional[float]] = []
        total = len(values)
        for idx in range(total):
            start = max(0, idx - radius)
            end = min(total, idx + radius + 1)
            window = [val for val in values[start:end] if val is not None]
            averages.append(sum(window) / len(window) if window else None)
        return averages

    @staticmethod
    def _neighbor_value(values: List[Optional[float]], start: int, step: int) -> Optional[float]:
        idx = start
        while 0 <= idx < len(values):
            candidate = values[idx]
            if candidate is not None:
                return candidate
            idx += step
        return None

    def _moving_average_direction(
        self,
        averages: List[Optional[float]],
        index: int,
        tolerance: float,
    ) -> str:
        if not averages:
            return "flat"
        current = averages[index]
        if current is None:
            return "flat"
        prev_value = self._neighbor_value(averages, index - 1, -1)
        next_value = self._neighbor_value(averages, index + 1, 1)
        delta = 0.0
        if prev_value is not None:
            delta = current - prev_value
        elif next_value is not None:
            delta = next_value - current
        if delta > tolerance:
            return "up"
        if delta < -tolerance:
            return "down"
        return "flat"

    def _trend_tolerance(self, values: List[Optional[float]]) -> float:
        numeric = [val for val in values if val is not None]
        if not numeric:
            return 0.0
        scale = max(max(abs(val) for val in numeric), 1.0)
        return max(scale * 1e-4, 1e-6)

    def _is_trend_deviation(
        self,
        direction: str,
        values: List[Optional[float]],
        index: int,
        tolerance: float,
    ) -> bool:
        if direction == "flat":
            return False
        current = values[index]
        if current is None:
            return False
        prev_value = self._neighbor_value(values, index - 1, -1)
        if prev_value is None:
            return False
        delta = current - prev_value
        if direction == "up":
            return delta < -tolerance
        return delta > tolerance

    def _resolve_monotonic_direction(
        self,
        config: TabConfig,
        header: str,
        values: List[Optional[float]],
    ) -> str:
        rule = config.monotonic_constraints.get(header, "").lower()
        if not rule or rule in {"none", "off", "disabled"}:
            return "flat"
        if rule in {"increase", "increasing", "up", "+"}:
            return "up"
        if rule in {"decrease", "decreasing", "down", "-"}:
            return "down"
        return self._determine_monotonic_direction(values)

    def _determine_monotonic_direction(self, values: List[Optional[float]]) -> str:
        numeric = [value for value in values if value is not None]
        if len(numeric) < 2:
            return "flat"
        tolerance = self._trend_tolerance(values)
        diff = numeric[-1] - numeric[0]
        if diff > tolerance:
            return "up"
        if diff < -tolerance:
            return "down"
        for prev, curr in zip(numeric, numeric[1:]):
            delta = curr - prev
            if delta > tolerance:
                return "up"
            if delta < -tolerance:
                return "down"
        return "flat"

    def _enforce_monotonic_sequence(
        self,
        values: List[Optional[float]],
        direction: str,
    ) -> List[Optional[float]]:
        if direction == "flat" or not values:
            return values[:]
        result: List[Optional[float]] = []
        prev_value: Optional[float] = None
        for value in values:
            if value is None:
                result.append(prev_value)
                continue
            if prev_value is None:
                enforced = value
            elif direction == "up":
                enforced = max(value, prev_value)
            else:
                enforced = min(value, prev_value)
            result.append(enforced)
            prev_value = enforced
        return result

    @staticmethod
    def _normalize_header(text: str) -> str:
        return text.replace("▶", "").replace("•", "").strip()

    @staticmethod
    def normalize_alpha(value: Any) -> str:
        try:
            numeric = float(value)
            return f"{numeric:.4f}"
        except (TypeError, ValueError):
            return str(value).strip()

    @staticmethod
    def _format_value(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.6g}"

    def _measurement_path_for_stem(self, stem: str, base_path: Path) -> Optional[Path]:
        candidates = [".ROP", ".rop", ".EMU", ".emu"]
        for extension in candidates:
            path = self.resolve_stem(stem, base_path, extension)
            if path.exists():
                return path
        return None

    def generate_initial_fits(self) -> None:
        if not self.loaded_alpha_pairs or not self.loaded_base_path:
            messagebox.showerror("Generate Fits", "Load an .LST file before running fits.")
            return
        errors: List[str] = []
        total_success = 0
        for tab_key in ("EPS", "MUS"):
            total_success += self.run_tab_fits(
                tab_key,
                silent=True,
                error_list=errors,
                reload=False,
                selected_pairs=None,
            )
        if total_success:
            missing = self.populate_tables(self.loaded_alpha_pairs, self.loaded_base_path)
            if missing:
                errors.extend(f"{name}: could not reload" for name in missing)
            if self.loaded_lst_path:
                self.load_lsf_data(self.loaded_lst_path)
            self.update_plots()
            self.update_summary_tables()
            self.update_highlights()
            self.update_polynomial_models()
            for key in self.tabs:
                self.refresh_ini_table(key)
            self.set_status("Generated initial DBE/2LM files from measurement data.")
        if errors:
            messagebox.showerror(
                "Generate Fits",
                "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else ""),
            )

    def _build_debye_controls(self, alpha_key: str) -> DebyeControls:
        block = self.parameter_blocks.get("EPS", {}).get(alpha_key)
        if not block:
            return DebyeControls()
        values = self._block_to_dict(block)
        return DebyeControls(
            fres=self._parse_float(self._value_for_keys(values, "FRES0")),
            deps=self._parse_float(self._value_for_keys(values, "DEPS0")),
            epsv=self._parse_float(self._value_for_keys(values, "EPSV0")),
            gamma=self._parse_float(self._value_for_keys(values, "GAMMA0", "GAMM0")),
            sige=self._parse_float(self._value_for_keys(values, "SIGE0")),
        )

    def _build_double_lorentz_controls(self, alpha_key: str) -> DoubleLorentzControls:
        block = self.parameter_blocks.get("MUS", {}).get(alpha_key)
        if not block:
            return DoubleLorentzControls()
        values = self._block_to_dict(block)
        return DoubleLorentzControls(
            fres1=self._parse_float(self._value_for_keys(values, "FR_M01", "FR_M0D")),
            deps1=self._parse_float(self._value_for_keys(values, "DMUR01", "DMUR0D")),
            gamma1=self._parse_float(self._value_for_keys(values, "GAMM01", "GAMM0D")),
            fres2=self._parse_float(self._value_for_keys(values, "FR_M02", "FR_M0L")),
            deps2=self._parse_float(self._value_for_keys(values, "DMUR02", "DMUR0L")),
            gamma2=self._parse_float(self._value_for_keys(values, "GAMM02", "GAMM0L")),
            epsv=self._parse_float(self._value_for_keys(values, "MURV0")),
            sige=self._parse_float(self._value_for_keys(values, "SIGM0", "SIGE0")),
        )

    def _gather_parameter_entries(self, tab_key: str, header: str) -> List[Dict[str, Any]]:
        state = self.tabs.get(tab_key)
        if not state:
            return []
        mapping_entry = state.config.ini_mapping.get(header)
        if not mapping_entry:
            return []
        column_index = state.column_index.get(header)
        stem_index = state.column_index.get(state.config.stem_header)
        if column_index is None:
            return []
        entries: List[Dict[str, Any]] = []
        for item in state.large_tree.get_children():
            values = state.large_tree.item(item, "values")
            if not values:
                continue
            alpha_text = values[0]
            alpha_key = self.normalize_alpha(alpha_text)
            block = self.parameter_blocks.get(tab_key, {}).get(alpha_key)
            if not block:
                continue
            candidates = (mapping_entry,) if isinstance(mapping_entry, str) else tuple(mapping_entry)
            header_name = next((name for name in candidates if name in block.headers), None)
            if not header_name:
                continue
            try:
                idx = block.headers.index(header_name)
            except ValueError:
                continue
            file_name = ""
            if stem_index is not None and stem_index < len(values):
                file_name = values[stem_index]
            lsf_value: Optional[float] = None
            if column_index < len(values):
                lsf_value = self._try_parse_numeric(values[column_index])
            entries.append(
                {
                    "alpha_text": alpha_text,
                    "alpha_key": alpha_key,
                    "block": block,
                    "index": idx,
                    "lsf_value": lsf_value,
                    "file_name": file_name,
                }
            )
        return entries

    def _seed_monotonic_initial_guesses(self, tab_key: str, header: str) -> None:
        state = self.tabs[tab_key]
        config = state.config
        mapping_entry = config.ini_mapping.get(header)
        if not mapping_entry:
            return
        constraint_rule = config.monotonic_constraints.get(header, "")
        if not constraint_rule or constraint_rule.lower() in {"none", "off", "disabled"}:
            return
        column_index = state.column_index.get(header)
        if column_index is None:
            return
        blocks = self.parameter_blocks.get(tab_key, {})
        entries: List[Dict[str, Any]] = []
        for item in state.large_tree.get_children():
            values = state.large_tree.item(item, "values")
            if not values:
                continue
            alpha_text = values[0]
            alpha_key = self.normalize_alpha(alpha_text)
            if column_index >= len(values):
                continue
            numeric_value = self._try_parse_numeric(values[column_index])
            if numeric_value is None:
                continue
            block = blocks.get(alpha_key)
            if not block:
                continue
            candidates = (mapping_entry,) if isinstance(mapping_entry, str) else tuple(mapping_entry)
            header_name = next((name for name in candidates if name in block.headers), None)
            if not header_name:
                continue
            try:
                idx = block.headers.index(header_name)
            except ValueError:
                continue
            entries.append(
                {
                    "alpha_value": self._safe_float(alpha_text),
                    "column_value": numeric_value,
                    "block": block,
                    "index": idx,
                }
            )
        if len(entries) < 2:
            return
        entries.sort(key=lambda entry: entry["alpha_value"])
        column_values = [entry["column_value"] for entry in entries]
        direction = self._resolve_monotonic_direction(config, header, column_values)
        if direction == "flat":
            return
        enforced_values = self._enforce_monotonic_sequence(column_values, direction)
        tolerance = self._trend_tolerance(column_values)
        updated_blocks: Dict[Path, ParameterBlock] = {}
        updates = 0
        for entry, seed_value in zip(entries, enforced_values):
            if seed_value is None:
                continue
            block = entry["block"]
            idx = entry["index"]
            current_text = block.values[idx]
            mode = self._infer_control_mode(current_text)
            if mode != "negative":
                continue
            magnitude = abs(seed_value)
            if magnitude <= 0:
                continue
            existing = self._try_parse_numeric(current_text)
            if existing is not None and abs(abs(existing) - magnitude) <= tolerance:
                continue
            magnitude = max(magnitude, 1e-12)
            magnitude_text = self._format_value(magnitude) or f"{magnitude:.6g}"
            normalized = magnitude_text.strip().lstrip("+")
            normalized = normalized.lstrip("-")
            if not normalized or normalized in {"0", "0.0"}:
                continue
            formatted = f"-{normalized}"
            if formatted == current_text:
                continue
            block.values[idx] = formatted
            updated_blocks[block.path] = block
            updates += 1
        if not updated_blocks:
            return
        last_path: Optional[Path] = None
        for block in updated_blocks.values():
            last_path = block.path
            if not self.apply_ini_changes(block, block.values, reload_data=False):
                return
        if last_path:
            self.reload_after_ini_change(last_path)
            direction_label = "increasing" if direction == "up" else "decreasing"
            self.set_status(
                f"Seeded {updates} monotonic initial guesses for {header} ({direction_label})."
            )

    def update_selected_parameter(self, tab_key: str) -> None:
        selection = self.selection
        if not selection or selection.tab_key != tab_key:
            messagebox.showinfo("Update Parameter", "Select a parameter column first.")
            return
        self._seed_monotonic_initial_guesses(tab_key, selection.header)
        state = self.tabs[tab_key]
        config = state.config
        mapping_entry = config.ini_mapping.get(selection.header)
        if not mapping_entry:
            messagebox.showerror("Update Parameter", f"No INI controls available for {selection.header}.")
            return
        parameter_label = selection.header
        candidates = (mapping_entry,) if isinstance(mapping_entry, str) else tuple(mapping_entry)
        pairs: List[Tuple[str, str]] = []
        for item in state.ini_tree.get_children():
            param_name = state.ini_tree.set(item, "PARAM")
            if param_name not in candidates:
                continue
            alpha = state.ini_tree.set(item, "ALPHA")
            file_name = state.ini_tree.set(item, "FILE")
            if not alpha or not file_name:
                continue
            pairs.append((alpha, file_name))
        if not pairs:
            messagebox.showerror("Update Parameter", "No files found for the selected parameter.")
            return
        errors: List[str] = []
        successes = self.run_tab_fits(
            tab_key,
            silent=False,
            error_list=errors,
            reload=True,
            selected_pairs=pairs,
        )
        if successes:
            self.set_status(f"[{tab_key}] Updated {parameter_label} for {successes} file(s).")
        if errors:
            messagebox.showerror(
                "Update Parameter",
                "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else ""),
            )

    def _block_to_dict(self, block: ParameterBlock) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for header, value in zip(block.headers, block.values):
            normalized = self._normalize_param_name(header)
            mapping[normalized] = value
        return mapping

    def _normalize_param_name(self, header: str) -> str:
        normalized = header.strip().upper().replace("Δ", "D")
        normalized = normalized.replace("Â", "A")
        normalized = normalized.replace("4MUR", "DMUR")
        normalized = normalized.replace(" ", "")
        return normalized

    def _parse_float(self, value: Optional[str]) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _value_for_keys(values: Dict[str, str], *keys: str) -> Optional[str]:
        for key in keys:
            if key in values:
                return values[key]
        return None

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    MaterialGui().run()
