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

from PyQt6 import QtCore, QtGui, QtWidgets


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
    Bottom panel that will eventually display generated PIV images.

    The initial implementation only shows a placeholder label, to keep the
    scaffolding light-weight and avoid pulling a Qt-matplotlib dependency
    into the very first iteration.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(
            "PIV image preview will appear here.\n"
            "In the next iteration this will be wired to the syPIV image_gen module.",
            self,
        )
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout.addWidget(self.label)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main window tying together the planar scene, parameter panel, and preview.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("syPIV Planar Digital Twin (Prototype)")
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
        self.run_action = toolbar.addAction("Generate PIV")
        self.run_action.setToolTip("Run a PIV image generation using current configuration.")

    def _connect_signals(self) -> None:
        self.run_action.triggered.connect(self.on_run_clicked)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)

    # -----------------------------
    # Slots / event handlers
    # -----------------------------
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
        msg = (
            "Generate PIV clicked.\n"
            "This prototype does not yet call the full syPIV pipeline, "
            "but the wiring is in place."
        )
        self.preview_panel.label.setText(msg)

    @QtCore.pyqtSlot()
    def on_parameters_changed(self) -> None:
        """
        Handle updates to parameters from the control panel.
        """
        # For now we only reflect that parameters changed in the window status bar.
        self.statusBar().showMessage("Parameters updated", 2000)


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

