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
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    try:
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        NavigationToolbar = None

from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from sypivlib.function.dataio import FlowIO, GridIO
from sypivlib.gui.sypiv_worker import SyPIVWorker


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
    Displays the interrogation area (IA) region selected from the contour plot.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        self._ia_item = None
        self._init_default_scene()

    def _init_default_scene(self) -> None:
        """
        Populate the scene with a simple coordinate frame and a couple of objects.
        """
        scene = self.scene()
        assert scene is not None

        # Background rectangle representing the interrogation area (IA).
        # Initial size is arbitrary; it will be updated to match the selected IA
        # from the contour plot and the view will auto-fit.
        ia_rect = QtCore.QRectF(-1.0, -0.5, 2.0, 1.0)
        self._ia_item = scene.addRect(
            ia_rect,
            pen=QtGui.QPen(QtGui.QColor("lightgray"), 2),
            brush=QtGui.QBrush(QtGui.QColor(240, 240, 240, 100)),
        )
        self._ia_item.setZValue(-1)

        # Sample objects that can be moved around; start them near the IA centre
        cx = ia_rect.center().x()
        cy = ia_rect.center().y()
        objects = [
            SceneObjectConfig(
                name="Object A",
                color=QtGui.QColor("red"),
                rect=QtCore.QRectF(cx - 0.1, cy - 0.05, 0.2, 0.1),
            ),
            SceneObjectConfig(
                name="Object B",
                color=QtGui.QColor("blue"),
                rect=QtCore.QRectF(cx + 0.05, cy + 0.02, 0.2, 0.08),
            ),
        ]
        for cfg in objects:
            scene.addItem(DraggableRectItem(cfg))

        scene.setSceneRect(ia_rect.adjusted(-50, -50, 50, 50))

    def update_ia_region(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """
        Update the interrogation area rectangle to match the selected region.
        """
        if self._ia_item is None:
            return
        
        # Create rectangle from bounds in the same physical coordinate system as the grid
        ia_rect = QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
        self._ia_item.setRect(ia_rect)

        # Re-centre movable objects inside the new IA
        scene = self.scene()
        if scene is not None:
            cx = ia_rect.center().x()
            cy = ia_rect.center().y()
            for item in scene.items():
                if isinstance(item, DraggableRectItem):
                    r = item.rect()
                    w, h = r.width(), r.height()
                    item.setRect(cx - w / 2, cy - h / 2, w, h)

            # Update scene rect to include the new IA region with some margin
            margin = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.2 or 1e-6
            scene.setSceneRect(ia_rect.adjusted(-margin, -margin, margin, margin))

        # Fit view to the IA region for clear visualization
        self.fitInView(self._ia_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()


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
            # Compute grid metrics so that search/interpolation/integration work correctly
            grid.compute_metrics()
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
    Uses tabs to organize parameters into logical groups.
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

        # Use tabs for parameter organization
        tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(tabs)

        # Particle parameters tab
        particle_tab = QtWidgets.QWidget()
        particle_layout = QtWidgets.QFormLayout(particle_tab)

        self.particle_n_concentration = QtWidgets.QSpinBox()
        self.particle_n_concentration.setRange(1, 1000000)
        self.particle_n_concentration.setValue(500)
        particle_layout.addRow("Particle count:", self.particle_n_concentration)

        self.particle_min_dia = QtWidgets.QDoubleSpinBox()
        self.particle_min_dia.setRange(1e-9, 1e-3)
        self.particle_min_dia.setDecimals(9)
        self.particle_min_dia.setValue(144e-9)
        self.particle_min_dia.setSuffix(" m")
        particle_layout.addRow("Min diameter:", self.particle_min_dia)

        self.particle_max_dia = QtWidgets.QDoubleSpinBox()
        self.particle_max_dia.setRange(1e-9, 1e-3)
        self.particle_max_dia.setDecimals(9)
        self.particle_max_dia.setValue(573e-9)
        self.particle_max_dia.setSuffix(" m")
        particle_layout.addRow("Max diameter:", self.particle_max_dia)

        self.particle_mean_dia = QtWidgets.QDoubleSpinBox()
        self.particle_mean_dia.setRange(1e-9, 1e-3)
        self.particle_mean_dia.setDecimals(9)
        self.particle_mean_dia.setValue(281e-9)
        self.particle_mean_dia.setSuffix(" m")
        particle_layout.addRow("Mean diameter:", self.particle_mean_dia)

        self.particle_std_dia = QtWidgets.QDoubleSpinBox()
        self.particle_std_dia.setRange(0, 1e-3)
        self.particle_std_dia.setDecimals(9)
        self.particle_std_dia.setValue(97e-9)
        self.particle_std_dia.setSuffix(" m")
        particle_layout.addRow("Std diameter:", self.particle_std_dia)

        self.particle_density = QtWidgets.QDoubleSpinBox()
        self.particle_density.setRange(1, 10000)
        self.particle_density.setValue(810)
        self.particle_density.setSuffix(" kg/m³")
        particle_layout.addRow("Particle density:", self.particle_density)

        self.particle_in_plane = QtWidgets.QDoubleSpinBox()
        self.particle_in_plane.setRange(0, 100)
        self.particle_in_plane.setDecimals(1)
        self.particle_in_plane.setValue(90.0)
        self.particle_in_plane.setSuffix(" %")
        particle_layout.addRow("In-plane percentage:", self.particle_in_plane)

        tabs.addTab(particle_tab, "Particles")

        # Laser parameters tab
        laser_tab = QtWidgets.QWidget()
        laser_layout = QtWidgets.QFormLayout(laser_tab)

        self.laser_position = QtWidgets.QDoubleSpinBox()
        self.laser_position.setRange(-10, 10)
        self.laser_position.setDecimals(6)
        self.laser_position.setValue(0.0009)
        self.laser_position.setSuffix(" m")
        laser_layout.addRow("Laser position (z):", self.laser_position)

        self.laser_thickness = QtWidgets.QDoubleSpinBox()
        self.laser_thickness.setRange(1e-6, 1e-2)
        self.laser_thickness.setDecimals(6)
        self.laser_thickness.setValue(0.0001)
        self.laser_thickness.setSuffix(" m")
        laser_layout.addRow("Laser thickness:", self.laser_thickness)

        self.laser_pulse_time = QtWidgets.QDoubleSpinBox()
        self.laser_pulse_time.setRange(1e-9, 1.0)
        self.laser_pulse_time.setDecimals(9)
        self.laser_pulse_time.setValue(1e-7)
        self.laser_pulse_time.setSuffix(" s")
        laser_layout.addRow("Pulse time:", self.laser_pulse_time)

        tabs.addTab(laser_tab, "Laser")

        # CCD parameters tab
        ccd_tab = QtWidgets.QWidget()
        ccd_layout = QtWidgets.QFormLayout(ccd_tab)

        self.ccd_xres = QtWidgets.QSpinBox()
        self.ccd_xres.setRange(16, 8192)
        self.ccd_xres.setValue(512)
        ccd_layout.addRow("X resolution (px):", self.ccd_xres)

        self.ccd_yres = QtWidgets.QSpinBox()
        self.ccd_yres.setRange(16, 8192)
        self.ccd_yres.setValue(512)
        ccd_layout.addRow("Y resolution (px):", self.ccd_yres)

        self.ccd_dpi = QtWidgets.QSpinBox()
        self.ccd_dpi.setRange(50, 1200)
        self.ccd_dpi.setValue(96)
        ccd_layout.addRow("DPI:", self.ccd_dpi)

        self.ccd_d_ccd = QtWidgets.QDoubleSpinBox()
        self.ccd_d_ccd.setRange(1e-4, 1.0)
        self.ccd_d_ccd.setDecimals(6)
        self.ccd_d_ccd.setValue(0.0135)  # Will be auto-calculated
        self.ccd_d_ccd.setSuffix(" m")
        ccd_layout.addRow("CCD distance:", self.ccd_d_ccd)

        self.ccd_d_ia = QtWidgets.QDoubleSpinBox()
        self.ccd_d_ia.setRange(1e-4, 1.0)
        self.ccd_d_ia.setDecimals(6)
        self.ccd_d_ia.setValue(0.0009)
        self.ccd_d_ia.setSuffix(" m")
        ccd_layout.addRow("IA distance:", self.ccd_d_ia)

        tabs.addTab(ccd_tab, "CCD")

        # Intensity parameters tab
        intensity_tab = QtWidgets.QWidget()
        intensity_layout = QtWidgets.QFormLayout(intensity_tab)

        self.intensity_sx = QtWidgets.QDoubleSpinBox()
        self.intensity_sx.setRange(0.1, 10.0)
        self.intensity_sx.setDecimals(2)
        self.intensity_sx.setValue(2.0)
        intensity_layout.addRow("Sx (pixel spread x):", self.intensity_sx)

        self.intensity_sy = QtWidgets.QDoubleSpinBox()
        self.intensity_sy.setRange(0.1, 10.0)
        self.intensity_sy.setDecimals(2)
        self.intensity_sy.setValue(2.0)
        intensity_layout.addRow("Sy (pixel spread y):", self.intensity_sy)

        self.intensity_frx = QtWidgets.QDoubleSpinBox()
        self.intensity_frx.setRange(0.1, 10.0)
        self.intensity_frx.setDecimals(2)
        self.intensity_frx.setValue(1.0)
        intensity_layout.addRow("Frx (frame size x):", self.intensity_frx)

        self.intensity_fry = QtWidgets.QDoubleSpinBox()
        self.intensity_fry.setRange(0.1, 10.0)
        self.intensity_fry.setDecimals(2)
        self.intensity_fry.setValue(1.0)
        intensity_layout.addRow("Fry (frame size y):", self.intensity_fry)

        self.intensity_s = QtWidgets.QDoubleSpinBox()
        self.intensity_s.setRange(1, 10000)
        self.intensity_s.setDecimals(0)
        self.intensity_s.setValue(2)
        intensity_layout.addRow("S (shape factor):", self.intensity_s)

        self.intensity_q = QtWidgets.QDoubleSpinBox()
        self.intensity_q.setRange(0.1, 10.0)
        self.intensity_q.setDecimals(2)
        self.intensity_q.setValue(1.0)
        intensity_layout.addRow("Q (efficiency):", self.intensity_q)

        tabs.addTab(intensity_tab, "Intensity")

        layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Policy.Expanding,
                                             QtWidgets.QSizePolicy.Policy.Expanding))

        # Connect all parameter changes
        all_widgets = [
            self.particle_n_concentration, self.particle_min_dia, self.particle_max_dia,
            self.particle_mean_dia, self.particle_std_dia, self.particle_density, self.particle_in_plane,
            self.laser_position, self.laser_thickness, self.laser_pulse_time,
            self.ccd_xres, self.ccd_yres, self.ccd_dpi, self.ccd_d_ccd, self.ccd_d_ia,
            self.intensity_sx, self.intensity_sy, self.intensity_frx, self.intensity_fry,
            self.intensity_s, self.intensity_q,
        ]
        for widget in all_widgets:
            widget.valueChanged.connect(self.parameters_changed.emit)  # type: ignore[arg-type]

    def get_particle_params(self) -> dict:
        """Get particle parameters as a dictionary."""
        return {
            "n_concentration": self.particle_n_concentration.value(),
            "min_dia": self.particle_min_dia.value(),
            "max_dia": self.particle_max_dia.value(),
            "mean_dia": self.particle_mean_dia.value(),
            "std_dia": self.particle_std_dia.value(),
            "density": self.particle_density.value(),
            "in_plane": self.particle_in_plane.value(),
            "distribution": "gaussian",
        }

    def get_laser_params(self) -> dict:
        """Get laser parameters as a dictionary."""
        return {
            "position": self.laser_position.value(),
            "thickness": self.laser_thickness.value(),
            "pulse_time": self.laser_pulse_time.value(),
        }

    def get_ccd_params(self) -> dict:
        """Get CCD parameters as a dictionary."""
        return {
            "xres": self.ccd_xres.value(),
            "yres": self.ccd_yres.value(),
            "dpi": self.ccd_dpi.value(),
            "d_ccd": self.ccd_d_ccd.value(),
            "d_ia": self.ccd_d_ia.value(),
        }

    def get_intensity_params(self) -> dict:
        """Get intensity parameters as a dictionary."""
        return {
            "sx": self.intensity_sx.value(),
            "sy": self.intensity_sy.value(),
            "frx": self.intensity_frx.value(),
            "fry": self.intensity_fry.value(),
            "s": int(self.intensity_s.value()),
            "q": self.intensity_q.value(),
        }


class PreviewPanel(QtWidgets.QWidget):
    """
    Bottom panel that displays contours or PIV-like images using matplotlib.
    Supports interactive navigation and interrogation region selection.
    """

    region_selected = QtCore.pyqtSignal(float, float, float, float)  # x_min, x_max, y_min, y_max

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        try:
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            
            # Add navigation toolbar for zoom/pan
            if NavigationToolbar is not None:
                self.toolbar = NavigationToolbar(self.canvas, self)
                layout.addWidget(self.toolbar)
            
            layout.addWidget(self.canvas)
            self._matplotlib_ok = True
            self._selector = None
            self._current_x = None
            self._current_y = None
            self._ax = None
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
        Enables interactive region selection.
        """
        if not self._matplotlib_ok:
            return

        try:
            self.figure.clear()
            self._ax = self.figure.add_subplot(111)
            cf = self._ax.contourf(x, y, field, levels=50, cmap="viridis")
            self._ax.set_aspect("equal", adjustable="box")
            self._ax.set_title(title)
            self.figure.colorbar(cf, ax=self._ax)
            
            # Store coordinates for region selection
            self._current_x = x
            self._current_y = y
            
            # Remove old selector if it exists
            if self._selector is not None:
                self._selector.set_active(False)
            
            # Add rectangle selector for interrogation region
            self._selector = RectangleSelector(
                self._ax,
                self._on_region_selected,
                useblit=True,
                button=[1],  # Left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            
            # Add instruction text
            self._ax.text(
                0.02, 0.98,
                "Click and drag to select\ninterrogation region",
                transform=self._ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            
            self.canvas.draw_idle()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.parent(),
                "Contour rendering error",
                f"Failed to render contour:\n{e}",
            )

    def _on_region_selected(self, eclick, erelease) -> None:
        """
        Callback when a region is selected with the rectangle selector.
        """
        if self._ax is None or self._current_x is None or self._current_y is None:
            return
        
        # Get the selected region bounds in data coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        # Emit signal with selected region bounds
        self.region_selected.emit(x_min, x_max, y_min, y_max)

    def set_region_selection_enabled(self, enabled: bool) -> None:
        """Enable or disable the rectangle selector."""
        if self._selector is not None:
            self._selector.set_active(enabled)


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
        self._ia_bounds: Optional[list[float]] = None  # [x_min, x_max, y_min, y_max]
        self._worker: Optional[SyPIVWorker] = None
        self._snapshots: dict[int, dict[str, np.ndarray]] = {}
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

        self.select_region_action = toolbar.addAction("Select IA Region")
        self.select_region_action.setCheckable(True)
        self.select_region_action.setToolTip("Enable/disable interrogation region selection on contour plot (click and drag).")
        self.select_region_action.setChecked(True)

        toolbar.addSeparator()

        self.run_action = toolbar.addAction("Generate PIV")
        self.run_action.setToolTip("Run a PIV image generation using current configuration.")

        toolbar.addSeparator()

        self.save_current_action = toolbar.addAction("Save Current Pair")
        self.save_current_action.setToolTip("Save the most recent snapshot pair to disk.")

        self.save_all_action = toolbar.addAction("Save All Pairs")
        self.save_all_action.setToolTip("Save all generated snapshot pairs to disk.")

    def _connect_signals(self) -> None:
        self.param_panel.file_panel.files_loaded.connect(self.on_files_loaded)
        self.variable_combo.currentTextChanged.connect(self.on_variable_changed)
        self.run_action.triggered.connect(self.on_run_clicked)
        self.param_panel.parameters_changed.connect(self.on_parameters_changed)
        self.preview_panel.region_selected.connect(self.on_region_selected)
        self.select_region_action.toggled.connect(self.on_region_selection_toggled)
        self.save_current_action.triggered.connect(self.on_save_current_pair)
        self.save_all_action.triggered.connect(self.on_save_all_pairs)

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
        """Trigger a PIV generation using the syPIV pipeline."""
        if self._grid is None or self._flow is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing data",
                "Please load Plot3D grid (.x) and flow (.q) files before generating PIV images.",
            )
            return

        if self._ia_bounds is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing IA region",
                "Please select an interrogation region on the contour plot before generating PIV images.",
            )
            return

        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Processing in progress",
                "A syPIV computation is already running. Please wait for it to finish.",
            )
            return

        particle_params = self.param_panel.get_particle_params()
        laser_params = self.param_panel.get_laser_params()
        ccd_params = self.param_panel.get_ccd_params()
        intensity_params = self.param_panel.get_intensity_params()

        self._snapshots.clear()

        self._worker = SyPIVWorker(
            grid=self._grid,
            flow=self._flow,
            ia_bounds=self._ia_bounds,
            particle_params=particle_params,
            laser_params=laser_params,
            ccd_params=ccd_params,
            intensity_params=intensity_params,
            num_snapshots=3,
            parent=self,
        )
        self._worker.progress.connect(self.on_worker_progress)
        self._worker.snapshot_complete.connect(self.on_snapshot_complete)
        self._worker.finished.connect(self.on_worker_finished)
        self._worker.error.connect(self.on_worker_error)

        self.statusBar().showMessage("Starting syPIV computation...")
        self._worker.start()

    @QtCore.pyqtSlot()
    def on_parameters_changed(self) -> None:
        """Handle updates to parameters from the control panel."""
        self.statusBar().showMessage("Parameters updated", 2000)

    @QtCore.pyqtSlot(float, float, float, float)
    def on_region_selected(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Handle interrogation region selection from the contour plot."""
        self._ia_bounds = [x_min, x_max, y_min, y_max]
        self.scene_view.update_ia_region(x_min, x_max, y_min, y_max)
        self.statusBar().showMessage(
            f"IA region selected: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]",
            3000
        )

    @QtCore.pyqtSlot(bool)
    def on_region_selection_toggled(self, enabled: bool) -> None:
        """Enable or disable region selection on the contour plot."""
        self.preview_panel.set_region_selection_enabled(enabled)

    @QtCore.pyqtSlot(str)
    def on_worker_progress(self, message: str) -> None:
        """Update status bar with worker progress."""
        self.statusBar().showMessage(message, 5000)

    @QtCore.pyqtSlot(int, object, object)
    def on_snapshot_complete(self, snap_num: int, img1: object, img2: object) -> None:
        """
        Handle completion of a snapshot pair from the worker.
        Store intensity arrays for later saving and show the latest pair in the preview.
        """
        try:
            # Store intensity arrays
            arr1 = img1.intensity.values
            arr2 = img2.intensity.values
            self._snapshots[snap_num] = {"image1": arr1, "image2": arr2}

            # Show the first image of the latest pair in the preview panel as a quick look
            xres = arr1.shape[1]
            yres = arr1.shape[0]
            x = np.linspace(-xres / 2, xres / 2, xres)
            y = np.linspace(-yres / 2, yres / 2, yres)
            X, Y = np.meshgrid(x, y)
            self.preview_panel.show_contour(X, Y, arr1, f"PIV snapshot {snap_num} (image 1)")

            self.statusBar().showMessage(f"Snapshot pair {snap_num} completed", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Snapshot handling error",
                f"Failed to handle snapshot pair {snap_num}:\n{e}",
            )

    @QtCore.pyqtSlot(bool, str)
    def on_worker_finished(self, success: bool, message: str) -> None:
        """Handle worker completion."""
        self.statusBar().showMessage(message, 8000)
        if not success:
            QtWidgets.QMessageBox.warning(self, "syPIV computation", message)

    @QtCore.pyqtSlot(str)
    def on_worker_error(self, message: str) -> None:
        """Show detailed error from the worker."""
        QtWidgets.QMessageBox.critical(self, "syPIV error", message)

    @QtCore.pyqtSlot()
    def on_save_current_pair(self) -> None:
        """Save the most recent snapshot pair."""
        if not self._snapshots:
            QtWidgets.QMessageBox.information(
                self,
                "No snapshots",
                "No snapshot pairs have been generated yet.",
            )
            return

        latest_snap = max(self._snapshots.keys())
        arrs = self._snapshots[latest_snap]

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select directory to save current pair",
            "",
        )
        if not directory:
            return

        base = os.path.join(directory, f"pair{latest_snap}")
        np.save(base + "_1.npy", arrs["image1"])
        np.save(base + "_2.npy", arrs["image2"])

        self.statusBar().showMessage(f"Saved current pair {latest_snap} as .npy files in {directory}", 8000)

    @QtCore.pyqtSlot()
    def on_save_all_pairs(self) -> None:
        """Save all generated snapshot pairs."""
        if not self._snapshots:
            QtWidgets.QMessageBox.information(
                self,
                "No snapshots",
                "No snapshot pairs have been generated yet.",
            )
            return

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select directory to save all pairs",
            "",
        )
        if not directory:
            return

        for snap_num, arrs in self._snapshots.items():
            base = os.path.join(directory, f"pair{snap_num}")
            np.save(base + "_1.npy", arrs["image1"])
            np.save(base + "_2.npy", arrs["image2"])

        self.statusBar().showMessage(f"Saved {len(self._snapshots)} snapshot pairs as .npy files in {directory}", 8000)

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
