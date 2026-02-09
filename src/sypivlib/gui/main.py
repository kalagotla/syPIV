"""
Entry point for the PyQt6-based planar syPIV GUI.

This provides a basic "digital twin" style layout:
 - Left: 2D scene where simple objects can be positioned.
 - Right: Parameter controls (placeholders for now).
 - Bottom: Preview area that will display generated PIV images.

The initial implementation is intentionally minimal and focuses on a clean,
testable scaffold that can be extended with real syPIV integration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
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


class ParameterPanel(QtWidgets.QWidget):
    """
    Right-hand panel for adjusting parameters that affect PIV generation.

    For now this contains only a few placeholder controls; these map cleanly
    onto typical syPIV configuration options (e.g. seeding density, pulse time).
    """

    parameters_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)

        self.seeding_density_spin = QtWidgets.QDoubleSpinBox(self)
        self.seeding_density_spin.setRange(0.0, 1e6)
        self.seeding_density_spin.setDecimals(0)
        self.seeding_density_spin.setValue(1e4)
        layout.addRow("Seeding concentration [1/m³]:", self.seeding_density_spin)

        self.pulse_time_spin = QtWidgets.QDoubleSpinBox(self)
        self.pulse_time_spin.setRange(0.0, 1.0)
        self.pulse_time_spin.setDecimals(5)
        self.pulse_time_spin.setSingleStep(1e-4)
        self.pulse_time_spin.setValue(5e-3)
        layout.addRow("Laser pulse time [s]:", self.pulse_time_spin)

        self.camera_dpi_spin = QtWidgets.QSpinBox(self)
        self.camera_dpi_spin.setRange(50, 1200)
        self.camera_dpi_spin.setValue(300)
        layout.addRow("Camera DPI:", self.camera_dpi_spin)

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

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def show_contour(self, x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str) -> None:
        """
        Render a filled contour of the provided scalar field.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cf = ax.contourf(x, y, field, levels=50, cmap="viridis")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        self.figure.colorbar(cf, ax=ax)
        self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    """
    Main window tying together the planar scene, parameter panel, and preview.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("syPIV Planar Digital Twin (Prototype)")
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

        # Toolbar / actions
        toolbar = self.addToolBar("Simulation")
        self.load_plot3d_action = toolbar.addAction("Load Plot3D")
        self.load_plot3d_action.setToolTip("Load Plot3D grid (.x) and flow (.q) files for contour visualization.")

        toolbar.addSeparator()

        self.variable_combo = QtWidgets.QComboBox(self)
        self.variable_combo.addItems(["rho", "u", "v", "w", "vel_mag"])
        self.variable_combo.setToolTip("Select flow variable to contour.")
        toolbar.addWidget(QtWidgets.QLabel("Variable:", self))
        toolbar.addWidget(self.variable_combo)

        toolbar.addSeparator()

        self.run_action = toolbar.addAction("Generate PIV")
        self.run_action.setToolTip("Run a PIV image generation using current configuration.")

    def _connect_signals(self) -> None:
        self.load_plot3d_action.triggered.connect(self.on_load_plot3d_clicked)
        self.variable_combo.currentTextChanged.connect(self.on_variable_changed)
        self.run_action.triggered.connect(self.on_run_clicked)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)

    # -----------------------------
    # Slots / event handlers
    # -----------------------------
    @QtCore.pyqtSlot()
    def on_load_plot3d_clicked(self) -> None:
        """
        Let the user select Plot3D grid (.x) and flow (.q) files and show a contour.
        """
        x_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Plot3D grid file (.x)",
            "",
            "Plot3D grid (*.x);;All files (*)",
        )
        if not x_path:
            return

        q_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Plot3D flow file (.q)",
            "",
            "Plot3D flow (*.q);;All files (*)",
        )
        if not q_path:
            return

        try:
            grid = GridIO(x_path)
            grid.read_grid()
            flow = FlowIO(q_path)
            flow.read_flow()
        except Exception as exc:  # pragma: no cover - GUI error dialog
            QtWidgets.QMessageBox.critical(
                self,
                "Error loading Plot3D",
                f"Failed to read Plot3D files:\n{exc}",
            )
            return

        self._grid = grid
        self._flow = flow
        self.statusBar().showMessage(f"Loaded Plot3D: {x_path} / {q_path}", 4000)
        self._update_contour()

    @QtCore.pyqtSlot(str)
    def on_variable_changed(self, name: str) -> None:
        """
        React to variable selection changes and refresh the contour.
        """
        self._current_variable = name
        self._update_contour()

    @QtCore.pyqtSlot()
    def on_run_clicked(self) -> None:
        """
        Trigger a PIV generation.

        In this prototype we only update the label text. In a subsequent
        iteration this method will:
          - Read the current object positions from the scene.
          - Build or update a syPIV configuration.
          - Call the image generation pipeline and render the resulting image.
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Generate PIV")
        msg_box.setText(
            "Generate PIV clicked.\n"
            "This prototype currently focuses on Plot3D contour visualization.\n"
            "In a subsequent step this action will be wired to the syPIV image_gen module.",
        )
        msg_box.exec()

    @QtCore.pyqtSlot()
    def on_parameters_changed(self) -> None:
        """
        Handle updates to parameters from the control panel.
        """
        # For now we only reflect that parameters changed in the window status bar.
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
            title = r"$\\rho$ (density)"
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


def main() -> None:
    """
    Launch the planar syPIV GUI.
    """
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

