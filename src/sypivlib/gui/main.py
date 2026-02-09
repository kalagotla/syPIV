"""
Entry point for the PyQt6-based planar syPIV GUI.

This provides a basic "digital twin" style layout:
 - Left: 2D scene where simple objects can be positioned.
 - Right: Parameter controls including Plot3D file loading.
 - Bottom: Preview area that will display generated PIV images or contours.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# Set matplotlib backend before importing FigureCanvas
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib.figure import Figure

from sypivlib.function.dataio import FlowIO, GridIO


@dataclass
class SceneObjectConfig:
    """Simple configuration for an object in the planar scene."""

    name: str
    color: QtGui.QColor
    rect: QtCore.QRectF


class DraggableRectItem(QtWidgets.QGraphicsRectItem):
    """
    A basic draggable rectangle representing an object in the planar setup.
    """

    def __init__(self, config: SceneObjectConfig, *args, **kwargs) -> None:
        super().__init__(config.rect, *args, **kwargs)
        self.setBrush(QtGui.QBrush(config.color))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._name = config.name

    @property
    def name(self) -> str:
        return self._name


class PlanarSceneView(QtWidgets.QGraphicsView):
    """
    Graphics view showing a simple 2D representation of the planar setup.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self._init_default_scene()

    def _init_default_scene(self) -> None:
        """
        Populate the scene with a simple coordinate frame and a couple of objects.
        """
        scene = self.scene()
        assert scene is not None

        # Background rectangle representing the interrogation area (IA)
        ia_rect = QtCore.QRectF(-200, -100, 400, 200)
        ia_item = scene.addRect(
            ia_rect,
            pen=QtGui.QPen(QtGui.QColor("lightgray")),
            brush=QtGui.QBrush(QtGui.QColor(240, 240, 240)),
        )
        ia_item.setZValue(-1)

        # Sample objects that can be moved around
        objects = [
            SceneObjectConfig(
                name="Object A",
                color=QtGui.QColor("red"),
                rect=QtCore.QRectF(-50, -20, 40, 40),
            ),
            SceneObjectConfig(
                name="Object B",
                color=QtGui.QColor("blue"),
                rect=QtCore.QRectF(20, 10, 50, 30),
            ),
        ]
        for cfg in objects:
            scene.addItem(DraggableRectItem(cfg))

        scene.setSceneRect(ia_rect.adjusted(-50, -50, 50, 50))


class FileLoadPanel(QtWidgets.QGroupBox):
    """
    Panel for loading Plot3D files using text inputs and browse buttons.
    """

    files_loaded = QtCore.pyqtSignal(object, object)  # grid, flow

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Plot3D Files", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Grid file
        grid_layout = QtWidgets.QHBoxLayout()
        grid_layout.addWidget(QtWidgets.QLabel("Grid (.x):", self))
        self.grid_path_edit = QtWidgets.QLineEdit(self)
        self.grid_path_edit.setPlaceholderText("Path to .x file")
        grid_browse_btn = QtWidgets.QPushButton("Browse...", self)
        grid_browse_btn.clicked.connect(self._browse_grid_file)
        grid_layout.addWidget(self.grid_path_edit, stretch=1)
        grid_layout.addWidget(grid_browse_btn)
        layout.addLayout(grid_layout)

        # Flow file
        flow_layout = QtWidgets.QHBoxLayout()
        flow_layout.addWidget(QtWidgets.QLabel("Flow (.q):", self))
        self.flow_path_edit = QtWidgets.QLineEdit(self)
        self.flow_path_edit.setPlaceholderText("Path to .q file")
        flow_browse_btn = QtWidgets.QPushButton("Browse...", self)
        flow_browse_btn.clicked.connect(self._browse_flow_file)
        flow_layout.addWidget(self.flow_path_edit, stretch=1)
        flow_layout.addWidget(flow_browse_btn)
        layout.addLayout(flow_layout)

        # Load button
        self.load_btn = QtWidgets.QPushButton("Load Files", self)
        self.load_btn.clicked.connect(self._load_files)
        layout.addWidget(self.load_btn)

        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _browse_grid_file(self) -> None:
        """Open file dialog for grid file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Plot3D grid file (.x)",
            self.grid_path_edit.text() or "",
            "Plot3D grid (*.x);;All files (*)",
        )
        if path:
            self.grid_path_edit.setText(path)

    def _browse_flow_file(self) -> None:
        """Open file dialog for flow file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Plot3D flow file (.q)",
            self.flow_path_edit.text() or "",
            "Plot3D flow (*.q);;All files (*)",
        )
        if path:
            self.flow_path_edit.setText(path)

    def _load_files(self) -> None:
        """Load the Plot3D files and emit signal."""
        grid_path = self.grid_path_edit.text().strip()
        flow_path = self.flow_path_edit.text().strip()

        if not grid_path:
            self._status_label.setText("<font color='red'>Please specify grid file path</font>")
            return

        if not flow_path:
            self._status_label.setText("<font color='red'>Please specify flow file path</font>")
            return

        if not os.path.exists(grid_path):
            self._status_label.setText(f"<font color='red'>Grid file not found: {grid_path}</font>")
            return

        if not os.path.exists(flow_path):
            self._status_label.setText(f"<font color='red'>Flow file not found: {flow_path}</font>")
            return

        self._status_label.setText("Loading files...")
        self.load_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        try:
            grid = GridIO(grid_path)
            grid.read_grid()
            flow = FlowIO(flow_path)
            flow.read_flow()

            self._status_label.setText(f"<font color='green'>Loaded: {os.path.basename(grid_path)}, {os.path.basename(flow_path)}</font>")
            self.files_loaded.emit(grid, flow)

        except Exception as exc:
            self._status_label.setText(f"<font color='red'>Error: {str(exc)}</font>")
            import traceback
            print(f"Error loading Plot3D files:\n{traceback.format_exc()}", file=sys.stderr)
        finally:
            self.load_btn.setEnabled(True)


class ParameterPanel(QtWidgets.QWidget):
    """
    Right-hand panel for adjusting parameters that affect PIV generation.
    """

    parameters_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # File loading panel
        self.file_panel = FileLoadPanel(self)
        layout.addWidget(self.file_panel)

        layout.addWidget(QtWidgets.QLabel("PIV Parameters:", self))

        # PIV parameters
        param_layout = QtWidgets.QFormLayout()

        self.seeding_density_spin = QtWidgets.QDoubleSpinBox(self)
        self.seeding_density_spin.setRange(0.0, 1e6)
        self.seeding_density_spin.setDecimals(0)
        self.seeding_density_spin.setValue(1e4)
        param_layout.addRow("Seeding concentration [1/m³]:", self.seeding_density_spin)

        self.pulse_time_spin = QtWidgets.QDoubleSpinBox(self)
        self.pulse_time_spin.setRange(0.0, 1.0)
        self.pulse_time_spin.setDecimals(5)
        self.pulse_time_spin.setSingleStep(1e-4)
        self.pulse_time_spin.setValue(5e-3)
        param_layout.addRow("Laser pulse time [s]:", self.pulse_time_spin)

        self.camera_dpi_spin = QtWidgets.QSpinBox(self)
        self.camera_dpi_spin.setRange(50, 1200)
        self.camera_dpi_spin.setValue(300)
        param_layout.addRow("Camera DPI:", self.camera_dpi_spin)

        layout.addLayout(param_layout)

        # Emit a single signal whenever any parameter changes
        for widget in (self.seeding_density_spin, self.pulse_time_spin, self.camera_dpi_spin):
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.valueChanged.connect(self.parameters_changed.emit)  # type: ignore[arg-type]
            else:
                widget.valueChanged.connect(self.parameters_changed.emit)  # type: ignore[arg-type]

        layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
                                             QtWidgets.QSizePolicy.Policy.Expanding))


class PreviewPanel(QtWidgets.QWidget):
    """
    Bottom panel that displays contours or PIV-like images using matplotlib.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        try:
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            self._matplotlib_ok = True
        except Exception as e:
            self._matplotlib_ok = False
            self._error_label = QtWidgets.QLabel(
                f"Matplotlib initialization failed: {e}\n"
                "Contour visualization will not be available.",
                self
            )
            self._error_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._error_label.setWordWrap(True)
            layout.addWidget(self._error_label)

    def show_contour(self, x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str) -> None:
        """
        Render a filled contour of the provided scalar field.
        """
        if not self._matplotlib_ok:
            return

        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            cf = ax.contourf(x, y, field, levels=50, cmap="viridis")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            self.figure.colorbar(cf, ax=ax)
            self.canvas.draw_idle()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.parent(),
                "Contour rendering error",
                f"Failed to render contour:\n{e}",
            )


class MainWindow(QtWidgets.QMainWindow):
    """
    Main window tying together the planar scene, parameter panel, and preview.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("syPIV Planar Digital Twin")
        self._grid: Optional[GridIO] = None
        self._flow: Optional[FlowIO] = None
        self._current_variable: str = "rho"
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)

        # Top split: scene (left) + parameters (right)
        top_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)

        self.scene_view = PlanarSceneView(top_split)
        self.param_panel = ParameterPanel(top_split)

        top_split.addWidget(self.scene_view)
        top_split.addWidget(self.param_panel)
        top_split.setStretchFactor(0, 3)
        top_split.setStretchFactor(1, 1)

        # Bottom: preview panel
        self.preview_panel = PreviewPanel(self)

        root_layout.addWidget(top_split, stretch=3)
        root_layout.addWidget(self.preview_panel, stretch=1)

        # Toolbar
        toolbar = self.addToolBar("Simulation")

        self.variable_combo = QtWidgets.QComboBox(self)
        self.variable_combo.addItems(["rho", "u", "v", "w", "vel_mag"])
        self.variable_combo.setToolTip("Select flow variable to contour.")
        toolbar.addWidget(QtWidgets.QLabel("Variable:", self))
        toolbar.addWidget(self.variable_combo)

        toolbar.addSeparator()

        self.run_action = toolbar.addAction("Generate PIV")
        self.run_action.setToolTip("Run a PIV image generation using current configuration.")

    def _connect_signals(self) -> None:
        self.param_panel.file_panel.files_loaded.connect(self.on_files_loaded)
        self.variable_combo.currentTextChanged.connect(self.on_variable_changed)
        self.run_action.triggered.connect(self.on_run_clicked)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)

    # -----------------------------
    # Slots / event handlers
    # -----------------------------
    @QtCore.pyqtSlot(object, object)
    def on_files_loaded(self, grid: GridIO, flow: FlowIO) -> None:
        """Handle Plot3D files being loaded."""
        self._grid = grid
        self._flow = flow
        self.statusBar().showMessage("Plot3D files loaded successfully", 3000)
        self._update_contour()

    @QtCore.pyqtSlot(str)
    def on_variable_changed(self, name: str) -> None:
        """React to variable selection changes and refresh the contour."""
        self._current_variable = name
        self._update_contour()

    @QtCore.pyqtSlot()
    def on_run_clicked(self) -> None:
        """Trigger a PIV generation."""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Generate PIV")
        msg_box.setText(
            "Generate PIV clicked.\n"
            "This will be wired to the syPIV image_gen module in a future update.",
        )
        msg_box.exec()

    @QtCore.pyqtSlot()
    def on_parameters_changed(self) -> None:
        """Handle updates to parameters from the control panel."""
        self.statusBar().showMessage("Parameters updated", 2000)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _update_contour(self) -> None:
        """
        If Plot3D data is loaded, draw a contour of the selected variable
        on the mid-span plane of block 0.
        """
        if self._grid is None or self._flow is None:
            return

        try:
            ni0 = int(self._grid.ni[0])
            nj0 = int(self._grid.nj[0])
            nk0 = int(self._grid.nk[0])

            # Use mid-plane in k-direction
            k_idx = nk0 // 2

            x = self._grid.grd[:ni0, :nj0, k_idx, 0, 0]
            y = self._grid.grd[:ni0, :nj0, k_idx, 1, 0]
            q = self._flow.q[:ni0, :nj0, k_idx, :, 0]

            rho = q[..., 0]
            u = q[..., 1] / rho
            v = q[..., 2] / rho
            w = q[..., 3] / rho

            if self._current_variable == "rho":
                field = rho
                title = r"$\rho$ (density)"
            elif self._current_variable == "u":
                field = u
                title = "u-velocity"
            elif self._current_variable == "v":
                field = v
                title = "v-velocity"
            elif self._current_variable == "w":
                field = w
                title = "w-velocity"
            else:  # vel_mag
                field = np.sqrt(u**2 + v**2 + w**2)
                title = "|V| (velocity magnitude)"

            self.preview_panel.show_contour(x, y, field, f"{title} (k index = {k_idx})")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Contour update error",
                f"Failed to update contour:\n{e}",
            )


def main() -> None:
    """
    Launch the planar syPIV GUI.
    """
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.resize(1200, 800)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error launching GUI: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
