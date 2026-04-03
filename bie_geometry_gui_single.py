
"""
Single-file PySide6 GUI starter for 2D BIE geometry setup.

Features
- Tab 1: Geometry
  - Plot on left
  - Editable table on right with columns: Curve Name, ITYPE, ICARD, IPN1, IPN2
  - Load / Save geometry files
  - Add / Delete rows
  - Edit hidden geometry fields (N, XA, YA, XB, YB, ANG) for selected row
  - Optional normal display
- Tab 2: Solver
  - Frequencies (GHz), elevations (deg), polarization
  - Validate project
  - Run stub that summarizes parsed settings
- Validation
  - Checks referenced material.<region_id> and impedance.<card_id> files
  - Uses region 0 as free space
- Input conventions
  - Geometry rows: ITYPE N XA YA XB YB ANG ICARD IPN1 IPN2
  - Materials: frequency_GHz eps_real eps_imag mu_real mu_imag
  - Impedance: frequency_GHz z_real z_imag

Notes
- ANG is parsed and saved but ignored for plotting/mesh right now.
- Right-hand rule is used for normals from A -> B.
- This is a GUI shell / project editor, not yet a full MoM solver.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------
# Data model
# ----------------------------

@dataclass
class GeometryCard:
    curve_name: str
    itype: int
    n: int
    xa: float
    ya: float
    xb: float
    yb: float
    ang: float
    icard: int
    ipn1: int
    ipn2: int


@dataclass
class SolverConfig:
    frequencies_ghz: List[float] = field(default_factory=list)
    elevations_deg: List[float] = field(default_factory=list)
    polarization: str = "TM"


@dataclass
class ProjectModel:
    title: str = "Untitled"
    geometry_cards: List[GeometryCard] = field(default_factory=list)
    solver: SolverConfig = field(default_factory=SolverConfig)
    geometry_path: Optional[Path] = None


# ----------------------------
# Parsers / IO
# ----------------------------

def parse_geometry_file(path: Path) -> ProjectModel:
    title = "Untitled"
    current_curve = "Curve"
    saw_data = False
    cards: List[GeometryCard] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                label = line[1:].strip() or "Unnamed"
                if not saw_data and title == "Untitled":
                    title = label
                else:
                    current_curve = label
                continue

            toks = line.split()
            if len(toks) != 10:
                raise ValueError(
                    f"{path.name}:{lineno}: expected 10 fields, got {len(toks)} -> {line}"
                )

            try:
                card = GeometryCard(
                    curve_name=current_curve,
                    itype=int(toks[0]),
                    n=int(toks[1]),
                    xa=float(toks[2]),
                    ya=float(toks[3]),
                    xb=float(toks[4]),
                    yb=float(toks[5]),
                    ang=float(toks[6]),
                    icard=int(toks[7]),
                    ipn1=int(toks[8]),
                    ipn2=int(toks[9]),
                )
            except Exception as e:
                raise ValueError(f"{path.name}:{lineno}: parse error: {e}") from e

            cards.append(card)
            saw_data = True

    model = ProjectModel(title=title, geometry_cards=cards, geometry_path=path)
    return model

def save_geometry_file(project: ProjectModel, path: Path) -> None:
    lines = [f"#{project.title}"]
    last_curve = None
    for c in project.geometry_cards:
        if c.curve_name != last_curve:
            lines.append(f"#{c.curve_name}")
            last_curve = c.curve_name
        lines.append(
            f"{c.itype:d} {c.n:d} "
            f"{c.xa:.6f} {c.ya:.6f} {c.xb:.6f} {c.yb:.6f} {c.ang:.6f} "
            f"{c.icard:d} {c.ipn1:d} {c.ipn2:d}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def load_material_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) != 5:
                raise ValueError(f"{path.name}:{lineno}: expected 5 fields")
            freq = float(toks[0])
            eps = float(toks[1]) - 1j * float(toks[2])
            mu = float(toks[3]) - 1j * float(toks[4])
            rows.append((freq, eps, mu))
    if not rows:
        raise ValueError(f"{path.name}: no usable rows")
    rows.sort(key=lambda x: x[0])
    return rows


def load_impedance_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) != 3:
                raise ValueError(f"{path.name}:{lineno}: expected 3 fields")
            freq = float(toks[0])
            z = float(toks[1]) + 1j * float(toks[2])
            rows.append((freq, z))
    if not rows:
        raise ValueError(f"{path.name}: no usable rows")
    rows.sort(key=lambda x: x[0])
    return rows


def parse_number_list(text: str) -> List[float]:
    """
    Supports:
    - comma/space list: 1,2,3
    - range: start:stop:step
    """
    text = text.strip()
    if not text:
        return []

    if ":" in text and "," not in text and " " not in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError("Range form must be start:stop:step")
        start, stop, step = map(float, parts)
        if step == 0:
            raise ValueError("Step cannot be zero")
        vals = []
        x = start
        if step > 0:
            while x <= stop + 1e-12:
                vals.append(round(x, 12))
                x += step
        else:
            while x >= stop - 1e-12:
                vals.append(round(x, 12))
                x += step
        return vals

    parts = [p for p in text.replace(",", " ").split() if p]
    return [float(p) for p in parts]




def project_curves(project: ProjectModel):
    groups = []
    by_name = {}
    order = []
    for idx, card in enumerate(project.geometry_cards):
        if card.curve_name not in by_name:
            by_name[card.curve_name] = {"name": card.curve_name, "rows": [idx], "card": card}
            order.append(card.curve_name)
        else:
            by_name[card.curve_name]["rows"].append(idx)
    return [by_name[name] for name in order]

# ----------------------------
# Table model
# ----------------------------

class GeometryTableModel(QAbstractTableModel):
    headers = ["Curve Name", "ITYPE", "ICARD", "IPN1", "IPN2"]

    def __init__(self, project: ProjectModel):
        super().__init__()
        self.project = project

    @staticmethod
    def _itype_color(itype: int):
        from PySide6.QtGui import QColor
        mapping = {
            1: QColor(255, 230, 204),  # orange tint
            2: QColor(220, 245, 220),  # green tint
            3: QColor(220, 232, 255),  # blue tint
            4: QColor(230, 230, 230),  # gray tint
            5: QColor(235, 235, 235),  # near white for black type
        }
        return mapping.get(itype, QColor(255, 255, 255))

    def _groups(self):
        return project_curves(self.project)

    def rowCount(self, parent=QModelIndex()):
        return len(self._groups())

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        grp = self._groups()[index.row()]
        card = grp["card"]
        col = index.column()

        if role in (Qt.DisplayRole, Qt.EditRole):
            values = [
                grp["name"],
                card.itype,
                card.icard,
                card.ipn1,
                card.ipn2,
            ]
            return values[col]

        if role == Qt.BackgroundRole:
            return self._itype_color(card.itype)

        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        grp = self._groups()[index.row()]
        rows = grp["rows"]

        try:
            if index.column() == 0:
                new_name = str(value).strip() or "Curve"
                for r in rows:
                    self.project.geometry_cards[r].curve_name = new_name
            elif index.column() == 1:
                new_val = int(value)
                for r in rows:
                    self.project.geometry_cards[r].itype = new_val
            elif index.column() == 2:
                new_val = int(value)
                for r in rows:
                    self.project.geometry_cards[r].icard = new_val
            elif index.column() == 3:
                new_val = int(value)
                for r in rows:
                    self.project.geometry_cards[r].ipn1 = new_val
            elif index.column() == 4:
                new_val = int(value)
                for r in rows:
                    self.project.geometry_cards[r].ipn2 = new_val
            else:
                return False
        except Exception:
            return False

        self.layoutChanged.emit()
        return True

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.headers[section]
        return str(section + 1)

    def reset_model(self):
        self.beginResetModel()
        self.endResetModel()


# ----------------------------
# Geometry editor dialog
# ----------------------------

class CardEditDialog(QDialog):
    def __init__(self, card: GeometryCard, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Geometry Card")
        self.card = card

        self.curve_name = QLineEdit(card.curve_name)
        self.itype = QLineEdit(str(card.itype))
        self.n = QLineEdit(str(card.n))
        self.xa = QLineEdit(str(card.xa))
        self.ya = QLineEdit(str(card.ya))
        self.xb = QLineEdit(str(card.xb))
        self.yb = QLineEdit(str(card.yb))
        self.ang = QLineEdit(str(card.ang))
        self.icard = QLineEdit(str(card.icard))
        self.ipn1 = QLineEdit(str(card.ipn1))
        self.ipn2 = QLineEdit(str(card.ipn2))

        form = QFormLayout()
        form.addRow("Curve Name", self.curve_name)
        form.addRow("ITYPE", self.itype)
        form.addRow("N", self.n)
        form.addRow("XA", self.xa)
        form.addRow("YA", self.ya)
        form.addRow("XB", self.xb)
        form.addRow("YB", self.yb)
        form.addRow("ANG", self.ang)
        form.addRow("ICARD", self.icard)
        form.addRow("IPN1", self.ipn1)
        form.addRow("IPN2", self.ipn2)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def apply(self):
        self.card.curve_name = self.curve_name.text().strip() or "Curve"
        self.card.itype = int(self.itype.text())
        self.card.n = int(self.n.text())
        self.card.xa = float(self.xa.text())
        self.card.ya = float(self.ya.text())
        self.card.xb = float(self.xb.text())
        self.card.yb = float(self.yb.text())
        self.card.ang = float(self.ang.text())
        self.card.icard = int(self.icard.text())
        self.card.ipn1 = int(self.ipn1.text())
        self.card.ipn2 = int(self.ipn2.text())


# ----------------------------
# Plot canvas
# ----------------------------

class GeometryCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.line_to_curve = {}

    @staticmethod
    def _itype_color(itype: int) -> str:
        mapping = {
            1: "orange",
            2: "green",
            3: "blue",
            4: "gray",
            5: "black",
        }
        return mapping.get(itype, "black")

    @staticmethod
    def _normal(card: GeometryCard):
        dx = card.xb - card.xa
        dy = card.yb - card.ya
        L = math.hypot(dx, dy)
        if L <= 0:
            return 0.0, 0.0
        tx, ty = dx / L, dy / L
        # right-hand rotated tangent
        nx, ny = ty, -tx
        return nx, ny

    def draw_project(
        self,
        project: ProjectModel,
        selected_curve_names: Optional[List[str]] = None,
        show_normals: bool = False,
        show_labels: bool = False,
    ):
        selected = set(selected_curve_names or [])
        self.ax.clear()
        self.line_to_curve = {}

        xs = []
        ys = []

        for grp in project_curves(project):
            name = grp["name"]
            rows = grp["rows"]
            card0 = grp["card"]
            color = self._itype_color(card0.itype)
            is_selected = name in selected
            lw = 3.0 if is_selected else 1.8
            alpha = 1.0 if is_selected else 0.9

            for row_idx in rows:
                card = project.geometry_cards[row_idx]
                (line,) = self.ax.plot(
                    [card.xa, card.xb],
                    [card.ya, card.yb],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    picker=6,
                )
                self.line_to_curve[line] = name
                xs.extend([card.xa, card.xb])
                ys.extend([card.ya, card.yb])

                if show_normals:
                    xc = 0.5 * (card.xa + card.xb)
                    yc = 0.5 * (card.ya + card.yb)
                    nx, ny = self._normal(card)
                    L = math.hypot(card.xb - card.xa, card.yb - card.ya)
                    scale = 0.10 * L if L > 0 else 1.0
                    self.ax.arrow(
                        xc, yc, nx * scale, ny * scale,
                        width=0.0, head_width=max(scale * 0.25, 0.1), head_length=max(scale * 0.35, 0.15),
                        length_includes_head=True, color=color, alpha=0.8
                    )

            if show_labels and rows:
                xc = sum(0.5 * (project.geometry_cards[r].xa + project.geometry_cards[r].xb) for r in rows) / len(rows)
                yc = sum(0.5 * (project.geometry_cards[r].ya + project.geometry_cards[r].yb) for r in rows) / len(rows)
                self.ax.text(xc, yc, name, fontsize=8)

        self.ax.set_title(project.title or "Geometry")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.25)
        self.ax.axis("equal")

        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            dx = max(xmax - xmin, 1.0)
            dy = max(ymax - ymin, 1.0)
            pad_x = 0.08 * dx
            pad_y = 0.08 * dy
            self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
            self.ax.set_ylim(ymin - pad_y, ymax + pad_y)

        self.draw_idle()


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = ProjectModel()
        self.setWindowTitle("2D Boundary Geometry / Solver Setup")
        self.resize(1350, 850)

        self.table_model = GeometryTableModel(self.project)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_toolbar()
        self._build_geometry_tab()
        self._build_solver_tab()

        self.refresh_all()

    # ---- UI build ----

    def _build_toolbar(self):
        tb = QToolBar("Main")
        self.addToolBar(tb)

        act_load = QAction("Load Geometry", self)
        act_save = QAction("Save Geometry", self)
        act_validate = QAction("Validate", self)

        act_load.triggered.connect(self.load_geometry_dialog)
        act_save.triggered.connect(self.save_geometry_dialog)
        act_validate.triggered.connect(self.validate_project)

        tb.addAction(act_load)
        tb.addAction(act_save)
        tb.addSeparator()
        tb.addAction(act_validate)

    def _build_geometry_tab(self):
        tab = QWidget()
        outer = QVBoxLayout(tab)

        splitter = QSplitter()
        outer.addWidget(splitter)

        # Left: plot
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.canvas = GeometryCanvas()
        left_layout.addWidget(self.canvas)

        plot_controls = QHBoxLayout()
        self.chk_normals = QCheckBox("Show Normals")
        self.chk_labels = QCheckBox("Show Labels")
        btn_fit = QPushButton("Fit View")
        btn_fit.clicked.connect(self.refresh_plot)
        self.chk_normals.toggled.connect(self.refresh_plot)
        self.chk_labels.toggled.connect(self.refresh_plot)
        plot_controls.addWidget(self.chk_normals)
        plot_controls.addWidget(self.chk_labels)
        plot_controls.addStretch(1)
        plot_controls.addWidget(btn_fit)
        left_layout.addLayout(plot_controls)

        # Right: table + controls
        right = QWidget()
        right_layout = QVBoxLayout(right)

        btn_row_layout = QHBoxLayout()
        btn_add = QPushButton("Add Row")
        btn_del = QPushButton("Delete Row")
        btn_edit = QPushButton("Edit Geometry...")
        btn_load = QPushButton("Load")
        btn_save = QPushButton("Save")
        btn_row_layout.addWidget(btn_add)
        btn_row_layout.addWidget(btn_del)
        btn_row_layout.addWidget(btn_edit)
        btn_row_layout.addStretch(1)
        btn_row_layout.addWidget(btn_load)
        btn_row_layout.addWidget(btn_save)
        right_layout.addLayout(btn_row_layout)

        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self.refresh_plot())
        self.canvas.mpl_connect("pick_event", self.on_plot_pick)
        self.table.doubleClicked.connect(self.edit_selected_geometry)
        right_layout.addWidget(self.table)

        right_layout.addWidget(QLabel("Validation / Log"))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        right_layout.addWidget(self.log_box)

        btn_add.clicked.connect(self.add_row)
        btn_del.clicked.connect(self.delete_selected_row)
        btn_edit.clicked.connect(self.edit_selected_geometry)
        btn_load.clicked.connect(self.load_geometry_dialog)
        btn_save.clicked.connect(self.save_geometry_dialog)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.tabs.addTab(tab, "Geometry")

    def _build_solver_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        form = QFormLayout()
        self.freq_edit = QLineEdit("1.0")
        self.elev_edit = QLineEdit("0")
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(["TM", "TE", "HH", "VV"])
        form.addRow("Frequency / Frequencies (GHz)", self.freq_edit)
        form.addRow("Elevations (deg)", self.elev_edit)
        form.addRow("Polarization", self.pol_combo)
        layout.addLayout(form)

        btns = QHBoxLayout()
        self.btn_validate = QPushButton("Validate")
        self.btn_run = QPushButton("Run")
        btns.addWidget(self.btn_validate)
        btns.addWidget(self.btn_run)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(QLabel("Solver / Project Summary"))
        layout.addWidget(self.summary_box)

        self.btn_validate.clicked.connect(self.validate_project)
        self.btn_run.clicked.connect(self.run_stub)

        self.tabs.addTab(tab, "Solver")

    # ---- actions ----

    def load_geometry_dialog(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Geometry File", "", "Geometry Files (*.txt *.dat *.*)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.project = parse_geometry_file(path)
            # preserve prior solver fields
            self.project.solver.frequencies_ghz = parse_number_list(self.freq_edit.text()) if self.freq_edit.text().strip() else []
            self.project.solver.elevations_deg = parse_number_list(self.elev_edit.text()) if self.elev_edit.text().strip() else []
            self.project.solver.polarization = self.pol_combo.currentText()
            self.table_model.project = self.project
            self.table_model.reset_model()
            self.log(f"Loaded geometry: {path}")
            self.refresh_all()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def save_geometry_dialog(self):
        default_dir = str(self.project.geometry_path.parent) if self.project.geometry_path else ""
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Geometry File", default_dir, "Geometry Files (*.txt *.dat *.*)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            save_geometry_file(self.project, path)
            self.project.geometry_path = path
            self.log(f"Saved geometry: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def add_row(self):
        self.project.geometry_cards.append(
            GeometryCard(
                curve_name="New Curve",
                itype=2,
                n=-1,
                xa=0.0,
                ya=0.0,
                xb=1.0,
                yb=0.0,
                ang=0.0,
                icard=0,
                ipn1=0,
                ipn2=0,
            )
        )
        self.table_model.reset_model()
        self.refresh_all()

    def delete_selected_row(self):
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        groups = project_curves(self.project)
        names_to_delete = {groups[i.row()]["name"] for i in idxs if 0 <= i.row() < len(groups)}
        self.project.geometry_cards = [c for c in self.project.geometry_cards if c.curve_name not in names_to_delete]
        self.table_model.reset_model()
        self.refresh_all()

    def edit_selected_geometry(self):
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            QMessageBox.information(self, "Edit Geometry", "Select one row first.")
            return
        groups = project_curves(self.project)
        row = idxs[0].row()
        grp = groups[row]
        card = grp["card"]
        dlg = CardEditDialog(card, self)
        if dlg.exec() == QDialog.Accepted:
            try:
                dlg.apply()
                # propagate shared properties across the full curve
                for r in grp["rows"]:
                    c = self.project.geometry_cards[r]
                    c.curve_name = card.curve_name
                    c.itype = card.itype
                    c.icard = card.icard
                    c.ipn1 = card.ipn1
                    c.ipn2 = card.ipn2
                self.table_model.reset_model()
                self.refresh_all()
            except Exception as e:
                QMessageBox.critical(self, "Edit Error", str(e))


    def curve_names_for_selected_rows(self) -> List[str]:
        names = []
        groups = project_curves(self.project)
        for idx in self.table.selectionModel().selectedRows():
            if 0 <= idx.row() < len(groups):
                names.append(groups[idx.row()]["name"])
        return names

    def select_curve_in_table(self, curve_name: str):
        groups = project_curves(self.project)
        from PySide6.QtCore import QItemSelectionModel
        self.table.clearSelection()
        for i, grp in enumerate(groups):
            if grp["name"] == curve_name:
                model_index = self.table_model.index(i, 0)
                self.table.selectionModel().select(
                    model_index,
                    QItemSelectionModel.Select | QItemSelectionModel.Rows
                )
                self.table.scrollTo(model_index)
                break

    def on_plot_pick(self, event):
        artist = getattr(event, "artist", None)
        if artist is None:
            return
        curve_name = self.canvas.line_to_curve.get(artist)
        if curve_name:
            self.select_curve_in_table(curve_name)
            self.refresh_plot()

    # ---- refresh / validation ----

    def selected_rows(self) -> List[int]:
        return [idx.row() for idx in self.table.selectionModel().selectedRows()]

    def refresh_plot(self):
        self.canvas.draw_project(
            self.project,
            selected_curve_names=self.curve_names_for_selected_rows(),
            show_normals=self.chk_normals.isChecked(),
            show_labels=self.chk_labels.isChecked(),
        )

    def refresh_summary(self):
        try:
            freqs = parse_number_list(self.freq_edit.text())
        except Exception:
            freqs = []
        try:
            elevs = parse_number_list(self.elev_edit.text())
        except Exception:
            elevs = []

        itype_counts = {}
        region_ids = set()
        icards = set()
        for c in self.project.geometry_cards:
            itype_counts[c.itype] = itype_counts.get(c.itype, 0) + 1
            if c.ipn1 != 0:
                region_ids.add(c.ipn1)
            if c.ipn2 != 0:
                region_ids.add(c.ipn2)
            if c.icard != 0:
                icards.add(c.icard)

        lines = [
            f"Title: {self.project.title}",
            f"Geometry file: {self.project.geometry_path or '(unsaved)'}",
            f"Cards: {len(self.project.geometry_cards)}",
            f"Curves: {len(set(c.curve_name for c in self.project.geometry_cards))}",
            f"Frequencies (GHz): {freqs if freqs else '[]'}",
            f"Elevations (deg): {elevs if elevs else '[]'}",
            f"Polarization: {self.pol_combo.currentText()}",
            "",
            "ITYPE counts:",
        ]
        for k in sorted(itype_counts):
            lines.append(f"  Type {k}: {itype_counts[k]}")
        lines.append("")
        lines.append(f"Referenced region IDs: {sorted(region_ids) if region_ids else '[]'}")
        lines.append(f"Referenced impedance IDs: {sorted(icards) if icards else '[]'}")
        self.summary_box.setPlainText("\n".join(lines))

    def refresh_all(self):
        self.refresh_plot()
        self.refresh_summary()

    def log(self, text: str):
        self.log_box.appendPlainText(text)

    def validate_project(self):
        self.log_box.clear()
        errors = []
        warnings = []

        # solver fields
        try:
            freqs = parse_number_list(self.freq_edit.text())
            if not freqs:
                warnings.append("No frequencies entered.")
        except Exception as e:
            errors.append(f"Frequency parse error: {e}")
            freqs = []

        try:
            elevs = parse_number_list(self.elev_edit.text())
            if not elevs:
                warnings.append("No elevations entered.")
        except Exception as e:
            errors.append(f"Elevation parse error: {e}")
            elevs = []

        # geometry checks
        valid_itypes = {1, 2, 3, 4, 5}
        region_ids = set()
        icards = set()

        for i, c in enumerate(self.project.geometry_cards, start=1):
            if c.itype not in valid_itypes:
                errors.append(f"Row {i}: invalid ITYPE {c.itype}.")
            if c.n == 0:
                errors.append(f"Row {i}: N cannot be zero.")
            if math.hypot(c.xb - c.xa, c.yb - c.ya) <= 0:
                errors.append(f"Row {i}: zero-length segment.")
            if c.itype == 5 and (c.ipn1 == 0 or c.ipn2 == 0):
                errors.append(f"Row {i}: ITYPE 5 requires nonzero IPN1 and IPN2.")
            if c.itype in (3, 4) and c.ipn1 == 0:
                errors.append(f"Row {i}: ITYPE {c.itype} requires nonzero IPN1.")
            if c.ipn1 < 0 or c.ipn2 < 0:
                errors.append(f"Row {i}: region IDs must be >= 0.")
            if c.icard < 0:
                errors.append(f"Row {i}: ICARD must be >= 0.")

            if c.ipn1 != 0:
                region_ids.add(c.ipn1)
            if c.ipn2 != 0:
                region_ids.add(c.ipn2)
            if c.icard != 0:
                icards.add(c.icard)

        base_dir = self.project.geometry_path.parent if self.project.geometry_path else None

        if base_dir is None:
            warnings.append("Geometry file path is not set yet, so file existence checks were skipped.")
        else:
            for rid in sorted(region_ids):
                p = base_dir / f"material.{rid}"
                if not p.exists():
                    errors.append(f"Missing material file: {p.name}")
                else:
                    try:
                        load_material_file(p)
                    except Exception as e:
                        errors.append(f"Bad material file {p.name}: {e}")
            for cid in sorted(icards):
                p = base_dir / f"impedance.{cid}"
                if not p.exists():
                    errors.append(f"Missing impedance file: {p.name}")
                else:
                    try:
                        load_impedance_file(p)
                    except Exception as e:
                        errors.append(f"Bad impedance file {p.name}: {e}")

        self.log("Validation results")
        self.log("=" * 60)
        if not errors and not warnings:
            self.log("No issues found.")
        else:
            if errors:
                self.log("Errors:")
                for e in errors:
                    self.log(f"  - {e}")
            if warnings:
                self.log("")
                self.log("Warnings:")
                for w in warnings:
                    self.log(f"  - {w}")

        self.refresh_summary()

    def run_stub(self):
        self.validate_project()

        try:
            freqs = parse_number_list(self.freq_edit.text())
            elevs = parse_number_list(self.elev_edit.text())
        except Exception:
            return

        self.project.solver.frequencies_ghz = freqs
        self.project.solver.elevations_deg = elevs
        self.project.solver.polarization = self.pol_combo.currentText()

        lines = [
            "Run stub",
            "=" * 60,
            f"Polarization: {self.project.solver.polarization}",
            f"Frequencies (GHz): {self.project.solver.frequencies_ghz}",
            f"Elevations (deg): {self.project.solver.elevations_deg}",
            f"Geometry cards: {len(self.project.geometry_cards)}",
            "",
            "This version is the GUI/editor/validator shell.",
            "Next step is wiring in segment expansion and the actual BIE/MoM solver.",
        ]
        QMessageBox.information(self, "Run", "\n".join(lines))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
