import os
from functools import partial
from typing import Dict, List, Optional

from qgis.PyQt.QtCore import Qt, QVariant, pyqtSignal
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStyle,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QDoubleSpinBox,
)
from qgis.core import (
    QgsCoordinateTransform,
    QgsField,
    QgsFeature,
    QgsGeometry,
    Qgis,
    QgsMapLayerType,
    QgsPointXY,
    QgsProject,
    QgsApplication,
    QgsVectorLayer,
    QgsVectorFileWriter,
    QgsWkbTypes,
)
from qgis.gui import QgsMapToolEmitPoint, QgsVertexMarker


class TerraMatchPointCaptureTool(QgsMapToolEmitPoint):
    pointCaptured = pyqtSignal(QgsPointXY)
    captureFinished = pyqtSignal()
    _finish_emitted = False

    def canvasPressEvent(self, event):
        if self._is_finish_click(event):
            if not self._finish_emitted:
                self._finish_emitted = True
                self.captureFinished.emit()
            return
        if event.button() == Qt.LeftButton:
            self.pointCaptured.emit(self.toMapCoordinates(event.pos()))

    def canvasReleaseEvent(self, event):
        # Some macOS devices report secondary-click more reliably on release.
        if self._is_finish_click(event):
            if not self._finish_emitted:
                self._finish_emitted = True
                self.captureFinished.emit()
            return

    def _is_finish_click(self, event) -> bool:
        return (
            event.button() == Qt.RightButton
            or (
                event.button() == Qt.LeftButton
                and bool(event.modifiers() & Qt.ControlModifier)
            )
        )


class TerraMatchDialog(QDialog):
    MARKER_TAG = "TerraMatchMarker"
    requestShow = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowFlags(
            self.windowFlags()
            | Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("TerraMatch")
        self.resize(900, 680)

        self._captured_points: List[QgsPointXY] = []
        self._capture_crs = None
        self._capture_tool: Optional[TerraMatchPointCaptureTool] = None
        self._previous_map_tool = None
        self._capturing = False
        self._markers: List[QgsVertexMarker] = []
        self._previous_window_modality = Qt.ApplicationModal
        self._accepted_params: Optional[Dict] = None

        self._build_ui()
        self._connect_project_signals()
        self.refresh_layers()
        if self.predictor_table.rowCount() == 0:
            self.add_predictor_row()
        # Clean up stale markers left by prior plugin sessions.
        self._purge_stale_markers(include_legacy=True)

    def _build_ui(self):
        root = QHBoxLayout(self)

        main_panel = QWidget(self)
        main_layout = QVBoxLayout(main_panel)
        main_layout.addWidget(self._build_predictors_group())
        main_layout.addWidget(self._build_training_group())
        main_layout.addWidget(self._build_output_group())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, main_panel)
        buttons.button(QDialogButtonBox.Ok).setText("Run TerraMatch")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        root.addWidget(main_panel, 3)
        help_panel = self._build_help_panel()
        help_panel.setMinimumWidth(240)
        help_panel.setMaximumWidth(280)
        root.addWidget(help_panel, 1)

    def _build_help_panel(self) -> QWidget:
        group = QGroupBox("Help & Definitions", self)
        layout = QVBoxLayout(group)
        help_text = QLabel(
            "Goal\n"
            "TerraMatch finds places with conditions similar to your successful training points.\n"
            "\n"
            "Workflow\n"
            "1) Add predictor layers.\n"
            "2) Choose training points (clicked points or point layer).\n"
            "3) Set per-layer buffer %, then run.\n"
            "\n"
            "Suitability Modes\n"
            "- Within: Keep cells within buffered min/max from training values.\n"
            "- Or Higher: Keep cells >= buffered minimum.\n"
            "- Or Lower: Keep cells <= buffered maximum.\n"
            "- Match: Keep exact training values (categories or numeric values).\n"
            "\n"
            "Why Use Buffer %\n"
            "A buffer expands the sampled min/max range so matching is less strict and can include nearby "
            "conditions.\n"
            "\n"
            "Vector Field Rules\n"
            "- Numeric vector fields use Within / Or Higher / Or Lower / Match.\n"
            "- Non-numeric vector fields default to Match.\n"
            "\n"
            "General Rules\n"
            "- All predictor layers must pass for a cell to be suitable.\n"
            "- NoData is treated as unsuitable.\n"
            "- Leave output directory blank for temporary output.",
            group,
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.PlainText)
        help_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(help_text)
        layout.addStretch()
        return group

    def _build_predictors_group(self) -> QWidget:
        group = QGroupBox("Predictor Inputs", self)
        layout = QVBoxLayout(group)

        help_label = QLabel(
            "Add any number of raster or polygon vector predictors. "
            "For vectors, select the field to evaluate.",
            group,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.predictor_table = QTableWidget(0, 5, group)
        self.predictor_table.setHorizontalHeaderLabels(
            ["Layer", "Type", "Field (vectors)", "Buffer %", "Suitability Mode"]
        )
        header = self.predictor_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.predictor_table)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Predictor", group)
        remove_button = QPushButton("Remove Selected", group)
        refresh_button = QPushButton("Refresh Layers", group)
        add_button.clicked.connect(self.add_predictor_row)
        remove_button.clicked.connect(self.remove_selected_predictors)
        refresh_button.clicked.connect(self.refresh_layers)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(refresh_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        return group

    def _build_training_group(self) -> QWidget:
        group = QGroupBox("Training Data", self)
        layout = QGridLayout(group)

        self.clicked_radio = QRadioButton("Use clicked map points", group)
        self.layer_radio = QRadioButton("Use existing point layer", group)
        self.clicked_radio.setChecked(True)

        self.capture_button = QPushButton("Start Capture", group)
        self.capture_button.clicked.connect(self.start_capture)
        self.clear_capture_button = QPushButton("Clear Captured Points", group)
        self.clear_capture_button.clicked.connect(self.clear_captured_points)
        self.save_points_button = QToolButton(group)
        self.save_points_button.setIcon(
            self._icon_with_fallback("/mActionFileSave.svg", QStyle.SP_DialogSaveButton)
        )
        self.save_points_button.setToolTip("Save captured points")
        self.save_points_button.clicked.connect(self.save_captured_points)
        self.load_points_button = QToolButton(group)
        self.load_points_button.setIcon(
            self._icon_with_fallback("/mActionFileOpen.svg", QStyle.SP_DialogOpenButton)
        )
        self.load_points_button.setToolTip("Load captured points")
        self.load_points_button.clicked.connect(self.load_captured_points)
        self.capture_count_label = QLabel("Captured points: 0", group)

        self.point_layer_combo = QComboBox(group)

        self.clicked_radio.toggled.connect(self._sync_training_mode_ui)
        self.layer_radio.toggled.connect(self._sync_training_mode_ui)

        help_label = QLabel(
            "Click left mouse button to add training points, right mouse button to finish capture.",
            group,
        )
        help_label.setWordWrap(True)

        layout.addWidget(self.clicked_radio, 0, 0, 1, 5)
        layout.addWidget(self.capture_button, 1, 0)
        layout.addWidget(self.clear_capture_button, 1, 1)
        layout.addWidget(self.save_points_button, 1, 2)
        layout.addWidget(self.load_points_button, 1, 3)
        layout.addWidget(self.capture_count_label, 1, 4)
        layout.addWidget(help_label, 2, 0, 1, 5)
        layout.addWidget(self.layer_radio, 3, 0, 1, 5)
        layout.addWidget(QLabel("Point layer:", group), 4, 0)
        layout.addWidget(self.point_layer_combo, 4, 1, 1, 4)

        return group

    def _icon_with_fallback(self, qgis_theme_path: str, fallback_style_icon: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon()
        try:
            icon = QgsApplication.getThemeIcon(qgis_theme_path)
        except Exception:
            icon = QIcon()
        if icon.isNull():
            icon = QApplication.style().standardIcon(fallback_style_icon)
        return icon

    def _build_output_group(self) -> QWidget:
        group = QGroupBox("Output", self)
        layout = QGridLayout(group)

        self.output_dir_edit = QLineEdit(group)
        self.output_dir_browse = QPushButton("Browse...", group)
        self.output_dir_browse.clicked.connect(self._browse_output_dir)

        note_label = QLabel(
            "Leave output directory blank to write temporary output.\n"
            "Vector-only analyses always auto-select grid size from input layers.",
            group,
        )
        note_label.setWordWrap(True)

        layout.addWidget(QLabel("Output directory (optional):", group), 0, 0)
        layout.addWidget(self.output_dir_edit, 0, 1)
        layout.addWidget(self.output_dir_browse, 0, 2)
        layout.addWidget(note_label, 1, 0, 1, 3)

        return group

    def _connect_project_signals(self):
        project = QgsProject.instance()
        project.layersAdded.connect(self.refresh_layers)
        project.layersRemoved.connect(self.refresh_layers)

    def refresh_layers(self):
        for row in range(self.predictor_table.rowCount()):
            layer_combo = self.predictor_table.cellWidget(row, 0)
            if isinstance(layer_combo, QComboBox):
                current_layer_id = layer_combo.currentData()
                self._populate_predictor_layer_combo(layer_combo, current_layer_id)
                self._update_predictor_row(row)
        self._populate_point_layer_combo()
        self._sync_training_mode_ui()

    def add_predictor_row(self):
        row = self.predictor_table.rowCount()
        self.predictor_table.insertRow(row)

        layer_combo = QComboBox(self.predictor_table)
        self._populate_predictor_layer_combo(layer_combo, None)
        layer_combo.currentIndexChanged.connect(partial(self._on_layer_combo_changed, layer_combo))

        type_label = QLabel("", self.predictor_table)
        field_combo = QComboBox(self.predictor_table)
        field_combo.currentIndexChanged.connect(partial(self._on_field_combo_changed, field_combo))

        buffer_spin = QDoubleSpinBox(self.predictor_table)
        buffer_spin.setDecimals(2)
        buffer_spin.setRange(0.0, 500.0)
        buffer_spin.setValue(10.0)
        buffer_spin.setSuffix(" %")

        mode_combo = QComboBox(self.predictor_table)
        self._set_mode_combo_state(mode_combo, mode_profile="numeric_raster")

        self.predictor_table.setCellWidget(row, 0, layer_combo)
        self.predictor_table.setCellWidget(row, 1, type_label)
        self.predictor_table.setCellWidget(row, 2, field_combo)
        self.predictor_table.setCellWidget(row, 3, buffer_spin)
        self.predictor_table.setCellWidget(row, 4, mode_combo)

        self._update_predictor_row(row)

    def remove_selected_predictors(self):
        selected_rows = sorted(
            {index.row() for index in self.predictor_table.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            self.predictor_table.removeRow(row)

    def _populate_predictor_layer_combo(
        self, combo: QComboBox, selected_layer_id: Optional[str]
    ):
        layers = self._eligible_predictor_layers()
        combo.blockSignals(True)
        combo.clear()
        for layer in layers:
            combo.addItem(layer.name(), layer.id())
        if selected_layer_id:
            idx = combo.findData(selected_layer_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _populate_point_layer_combo(self):
        current_id = self.point_layer_combo.currentData()
        point_layers = self._point_layers()
        self.point_layer_combo.clear()
        for layer in point_layers:
            self.point_layer_combo.addItem(layer.name(), layer.id())
        if current_id:
            idx = self.point_layer_combo.findData(current_id)
            if idx >= 0:
                self.point_layer_combo.setCurrentIndex(idx)

    def _eligible_predictor_layers(self):
        layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayerType.RasterLayer and layer.isValid():
                layers.append(layer)
                continue

            if (
                layer.type() == QgsMapLayerType.VectorLayer
                and layer.isValid()
                and layer.geometryType() == QgsWkbTypes.PolygonGeometry
            ):
                layers.append(layer)
        layers.sort(key=lambda lyr: lyr.name().lower())
        return layers

    def _point_layers(self):
        layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if (
                isinstance(layer, QgsVectorLayer)
                and layer.isValid()
                and layer.geometryType() == QgsWkbTypes.PointGeometry
            ):
                layers.append(layer)
        layers.sort(key=lambda lyr: lyr.name().lower())
        return layers

    def _on_layer_combo_changed(self, layer_combo: QComboBox):
        row = self._find_row_for_widget(layer_combo)
        if row >= 0:
            self._update_predictor_row(row)

    def _on_field_combo_changed(self, field_combo: QComboBox):
        row = self._find_row_for_widget(field_combo)
        if row >= 0:
            self._update_predictor_mode_for_row(row)

    def _find_row_for_widget(self, widget) -> int:
        for row in range(self.predictor_table.rowCount()):
            for col in range(self.predictor_table.columnCount()):
                if self.predictor_table.cellWidget(row, col) is widget:
                    return row
        return -1

    def _update_predictor_row(self, row: int):
        layer_combo = self.predictor_table.cellWidget(row, 0)
        type_label = self.predictor_table.cellWidget(row, 1)
        field_combo = self.predictor_table.cellWidget(row, 2)
        mode_combo = self.predictor_table.cellWidget(row, 4)

        if not isinstance(layer_combo, QComboBox):
            return
        if not isinstance(type_label, QLabel):
            return
        if not isinstance(field_combo, QComboBox):
            return
        if not isinstance(mode_combo, QComboBox):
            return

        layer_id = layer_combo.currentData()
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None

        field_combo.blockSignals(True)
        field_combo.clear()

        if layer is None:
            type_label.setText("Missing")
            field_combo.addItem("(no layer)")
            field_combo.setEnabled(False)
            self._set_mode_combo_state(mode_combo, mode_profile="missing")
        elif layer.type() == QgsMapLayerType.RasterLayer:
            type_label.setText("Raster")
            field_combo.addItem("(band 1)")
            field_combo.setEnabled(False)
            self._set_mode_combo_state(mode_combo, mode_profile="numeric_raster")
        else:
            type_label.setText("Polygon Vector")
            for field in layer.fields():
                field_combo.addItem(field.name())
            field_combo.setEnabled(True)
            if field_combo.count() == 0:
                field_combo.addItem("(no fields)")
                field_combo.setEnabled(False)
            self._update_predictor_mode_for_row(row)

        field_combo.blockSignals(False)

    def _update_predictor_mode_for_row(self, row: int):
        layer_combo = self.predictor_table.cellWidget(row, 0)
        field_combo = self.predictor_table.cellWidget(row, 2)
        mode_combo = self.predictor_table.cellWidget(row, 4)
        if not isinstance(layer_combo, QComboBox):
            return
        if not isinstance(field_combo, QComboBox):
            return
        if not isinstance(mode_combo, QComboBox):
            return

        layer_id = layer_combo.currentData()
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
        if layer is None:
            self._set_mode_combo_state(mode_combo, mode_profile="missing")
            return
        if layer.type() == QgsMapLayerType.RasterLayer:
            self._set_mode_combo_state(mode_combo, mode_profile="numeric_raster")
            return

        field_name = field_combo.currentText()
        field = layer.fields().field(field_name) if field_name else None
        if self._is_numeric_field(field):
            self._set_mode_combo_state(mode_combo, mode_profile="numeric_vector")
            return
        self._set_mode_combo_state(mode_combo, mode_profile="match")

    def _set_mode_combo_state(self, mode_combo: QComboBox, mode_profile: str):
        selected_mode = mode_combo.currentData()
        mode_combo.blockSignals(True)
        mode_combo.clear()

        if mode_profile == "numeric_raster":
            mode_combo.addItem("Within", "within")
            mode_combo.addItem("Or Higher", "or_higher")
            mode_combo.addItem("Or Lower", "or_lower")
            default_mode = "within"
            mode_combo.setEnabled(True)
        elif mode_profile == "numeric_vector":
            mode_combo.addItem("Within", "within")
            mode_combo.addItem("Or Higher", "or_higher")
            mode_combo.addItem("Or Lower", "or_lower")
            mode_combo.addItem("Match", "match")
            default_mode = "within"
            mode_combo.setEnabled(True)
        elif mode_profile == "match":
            mode_combo.addItem("Match", "match")
            default_mode = "match"
            mode_combo.setEnabled(False)
        else:
            mode_combo.addItem("Within", "within")
            default_mode = "within"
            mode_combo.setEnabled(False)

        if selected_mode is None:
            selected_mode = default_mode
        idx = mode_combo.findData(selected_mode)
        if idx < 0:
            idx = mode_combo.findData(default_mode)
        if idx >= 0:
            mode_combo.setCurrentIndex(idx)
        mode_combo.blockSignals(False)

    def _is_numeric_field(self, field) -> bool:
        if field is None:
            return False
        numeric_types = {
            getattr(QVariant, "Int", None),
            getattr(QVariant, "UInt", None),
            getattr(QVariant, "LongLong", None),
            getattr(QVariant, "ULongLong", None),
            getattr(QVariant, "Double", None),
        }
        return field.type() in {v for v in numeric_types if v is not None}

    def _sync_training_mode_ui(self):
        clicked_mode = self.clicked_radio.isChecked()
        self.capture_button.setEnabled(clicked_mode and not self._capturing)
        self.clear_capture_button.setEnabled(
            clicked_mode and (not self._capturing) and len(self._captured_points) > 0
        )
        self.save_points_button.setEnabled((not self._capturing) and len(self._captured_points) > 0)
        self.load_points_button.setEnabled(not self._capturing)
        self.point_layer_combo.setEnabled(not clicked_mode)

    def start_capture(self):
        if self.iface is None:
            return
        if self._capturing:
            return

        canvas = self.iface.mapCanvas()
        self._capture_crs = canvas.mapSettings().destinationCrs()
        self._previous_map_tool = canvas.mapTool()
        self._previous_window_modality = self.windowModality()

        self._capture_tool = TerraMatchPointCaptureTool(canvas)
        self._capture_tool.pointCaptured.connect(self._on_point_captured)
        self._capture_tool.captureFinished.connect(self.finish_capture)
        canvas.setMapTool(self._capture_tool)

        # Keep the dialog alive while allowing map interaction for point capture.
        self.setWindowModality(Qt.NonModal)
        self._capturing = True
        self._sync_training_mode_ui()
        self.showMinimized()
        self.iface.mainWindow().activateWindow()
        canvas.setFocus()
        self.iface.messageBar().pushMessage(
            "TerraMatch",
            "Capture mode active: left-click points, right-click to finish.",
            level=Qgis.Info,
            duration=5,
        )

    def _on_point_captured(self, point: QgsPointXY):
        self._add_captured_point(point)
        self.capture_count_label.setText(f"Captured points: {len(self._captured_points)}")
        self._sync_training_mode_ui()

    def finish_capture(self):
        self._stop_capture_mode()
        self._restore_after_capture()
        self.requestShow.emit()
        self.iface.messageBar().pushMessage(
            "TerraMatch",
            f"Capture finished with {len(self._captured_points)} points.",
            level=Qgis.Info,
            duration=4,
        )

    def _restore_after_capture(self):
        if not self.isVisible():
            self.show()
        self.setWindowState(Qt.WindowNoState)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def _stop_capture_mode(self):
        if not self._capturing or self.iface is None:
            self._restore_window_modality()
            return
        canvas = self.iface.mapCanvas()
        try:
            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
            else:
                canvas.unsetMapTool(self._capture_tool)
        except Exception:
            pass

        if self._capture_tool is not None:
            try:
                self._capture_tool.pointCaptured.disconnect(self._on_point_captured)
                self._capture_tool.captureFinished.disconnect(self.finish_capture)
            except Exception:
                pass
        self._capture_tool = None
        self._previous_map_tool = None
        self._capturing = False
        self._restore_window_modality()
        self._sync_training_mode_ui()

    def _restore_window_modality(self):
        try:
            self.setWindowModality(self._previous_window_modality)
        except Exception:
            pass

    def clear_captured_points(self):
        self._captured_points = []
        self.capture_count_label.setText("Captured points: 0")
        while self._markers:
            marker = self._markers.pop()
            self._remove_marker_item(marker)
        # Also sweep stale markers that may not be in the local list.
        self._purge_stale_markers(include_legacy=False)
        self._sync_training_mode_ui()

    def _add_captured_point(self, point: QgsPointXY):
        self._captured_points.append(point)
        marker = QgsVertexMarker(self.iface.mapCanvas())
        marker.setCenter(point)
        marker.setColor(QColor("red"))
        marker.setIconType(QgsVertexMarker.ICON_CROSS)
        marker.setIconSize(9)
        marker.setPenWidth(3)
        try:
            marker.setData(0, self.MARKER_TAG)
        except Exception:
            pass
        self._markers.append(marker)

    def load_captured_points(self):
        try:
            parent = self.iface.mainWindow() if self.iface is not None else self
            path, _ = QFileDialog.getOpenFileName(
                parent,
                "Load Captured Points",
                os.path.join(os.path.expanduser("~"), "Desktop"),
                "Vector files (*.gpkg *.geojson *.json *.shp);;All files (*.*)",
            )
            if not path:
                return

            layer = QgsVectorLayer(path, "loaded_capture_points", "ogr")
            if not layer.isValid():
                raise RuntimeError("Selected file is not a valid vector layer.")
            if layer.geometryType() != QgsWkbTypes.PointGeometry:
                raise RuntimeError("Selected layer is not a point layer.")

            loaded_points = []
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                if QgsWkbTypes.isMultiType(geom.wkbType()):
                    loaded_points.extend(list(geom.asMultiPoint()))
                else:
                    loaded_points.append(geom.asPoint())

            if not loaded_points:
                raise RuntimeError("No valid point geometries found in selected layer.")

            target_crs = None
            if self.iface is not None:
                target_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if target_crs is None:
                target_crs = QgsProject.instance().crs()

            if layer.crs() != target_crs:
                transform = QgsCoordinateTransform(
                    layer.crs(), target_crs, QgsProject.instance().transformContext()
                )
                transformed = []
                for point in loaded_points:
                    transformed.append(transform.transform(point))
                loaded_points = transformed

            self.clear_captured_points()
            self._capture_crs = target_crs
            for point in loaded_points:
                self._add_captured_point(QgsPointXY(point))
            self.capture_count_label.setText(f"Captured points: {len(self._captured_points)}")
            self._sync_training_mode_ui()
            QMessageBox.information(
                self,
                "TerraMatch",
                f"Loaded {len(self._captured_points)} captured points.",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "TerraMatch",
                f"Failed to load captured points:\n{exc}",
            )

    def save_captured_points(self):
        try:
            if len(self._captured_points) == 0:
                QMessageBox.information(self, "TerraMatch", "No captured points to save.")
                return

            default_path = os.path.join(
                os.path.expanduser("~"),
                "Desktop",
                "terramatch_points.gpkg",
            )
            parent = self.iface.mainWindow() if self.iface is not None else self
            dialog_options = QFileDialog.Options()
            dialog_options |= QFileDialog.DontUseNativeDialog
            path, selected_filter = QFileDialog.getSaveFileName(
                parent,
                "Save Captured Points",
                default_path,
                "GeoPackage (*.gpkg);;GeoJSON (*.geojson *.json);;ESRI Shapefile (*.shp)",
                options=dialog_options,
            )
            if not path:
                return

            path, driver_name = self._normalize_output_vector_path(path, selected_filter)

            capture_crs = self._capture_crs
            if capture_crs is None and self.iface is not None:
                capture_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            if capture_crs is None:
                capture_crs = QgsProject.instance().crs()

            memory_uri = f"Point?crs={capture_crs.authid() or capture_crs.toWkt()}"
            memory_layer = QgsVectorLayer(memory_uri, "captured_points", "memory")
            if not memory_layer.isValid():
                raise RuntimeError("Unable to create in-memory point layer for export.")
            provider = memory_layer.dataProvider()
            provider.addAttributes([QgsField("id", QVariant.Int)])
            memory_layer.updateFields()

            features = []
            for idx, point in enumerate(self._captured_points, start=1):
                feat = QgsFeature(memory_layer.fields())
                feat.setGeometry(QgsGeometry.fromPointXY(point))
                feat.setAttribute("id", idx)
                features.append(feat)
            provider.addFeatures(features)
            memory_layer.updateExtents()

            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = driver_name
            options.fileEncoding = "UTF-8"
            options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
            if driver_name == "GPKG":
                options.layerName = os.path.splitext(os.path.basename(path))[0] or "captured_points"

            write_result = QgsVectorFileWriter.writeAsVectorFormatV2(
                memory_layer,
                path,
                QgsProject.instance().transformContext(),
                options,
            )
            err = write_result[0] if isinstance(write_result, tuple) else write_result
            if err != QgsVectorFileWriter.NoError:
                raise RuntimeError(f"QGIS writer returned error code: {err}")

            QMessageBox.information(
                self,
                "TerraMatch",
                f"Saved {len(self._captured_points)} points:\n{path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "TerraMatch",
                f"Failed to save captured points:\n{exc}",
            )

    def _normalize_output_vector_path(self, path: str, selected_filter: str):
        filter_text = (selected_filter or "").lower()
        lower = path.lower()

        if "geojson" in filter_text:
            driver = "GeoJSON"
            if not (lower.endswith(".geojson") or lower.endswith(".json")):
                path = f"{path}.geojson"
        elif "shapefile" in filter_text:
            driver = "ESRI Shapefile"
            if not lower.endswith(".shp"):
                path = f"{path}.shp"
        else:
            driver = "GPKG"
            if not lower.endswith(".gpkg"):
                path = f"{path}.gpkg"

        return path, driver

    def force_cleanup(self):
        self._stop_capture_mode()
        self.clear_captured_points()
        self._purge_stale_markers(include_legacy=True)

    def _remove_marker_item(self, marker):
        if self.iface is None:
            return
        try:
            self.iface.mapCanvas().scene().removeItem(marker)
        except Exception:
            pass

    def _is_legacy_terramatch_marker(self, marker) -> bool:
        # Backward compatibility: remove old untagged TerraMatch markers.
        try:
            return (
                marker.iconType() == QgsVertexMarker.ICON_CROSS
                and marker.iconSize() == 9
                and marker.penWidth() == 3
                and marker.color().name().lower() == QColor("red").name().lower()
            )
        except Exception:
            return False

    def _purge_stale_markers(self, include_legacy: bool):
        if self.iface is None:
            return
        scene = self.iface.mapCanvas().scene()
        for item in list(scene.items()):
            if not isinstance(item, QgsVertexMarker):
                continue

            is_tagged = False
            try:
                is_tagged = item.data(0) == self.MARKER_TAG
            except Exception:
                is_tagged = False

            if is_tagged or (include_legacy and self._is_legacy_terramatch_marker(item)):
                self._remove_marker_item(item)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def build_parameters(self) -> Dict:
        predictors = []
        for row in range(self.predictor_table.rowCount()):
            layer_combo = self.predictor_table.cellWidget(row, 0)
            type_label = self.predictor_table.cellWidget(row, 1)
            field_combo = self.predictor_table.cellWidget(row, 2)
            buffer_spin = self.predictor_table.cellWidget(row, 3)
            mode_combo = self.predictor_table.cellWidget(row, 4)
            if not isinstance(layer_combo, QComboBox):
                continue
            if not isinstance(type_label, QLabel):
                continue
            if not isinstance(field_combo, QComboBox):
                continue
            if not isinstance(buffer_spin, QDoubleSpinBox):
                continue
            if not isinstance(mode_combo, QComboBox):
                continue

            layer_id = layer_combo.currentData()
            if not layer_id:
                continue
            layer_type = "raster" if type_label.text() == "Raster" else "vector"
            field_name = None
            if layer_type == "vector" and field_combo.isEnabled():
                field_name = field_combo.currentText()
            predictors.append(
                {
                    "layer_id": layer_id,
                    "layer_type": layer_type,
                    "field_name": field_name,
                    "buffer_pct": float(buffer_spin.value()),
                    "match_mode": mode_combo.currentData() or "within",
                }
            )

        output_dir = self.output_dir_edit.text().strip() or None
        return {
            "predictors": predictors,
            "training_mode": "clicked" if self.clicked_radio.isChecked() else "point_layer",
            "clicked_points": list(self._captured_points),
            "clicked_crs": self._capture_crs,
            "point_layer_id": self.point_layer_combo.currentData(),
            "output_dir": output_dir,
        }

    def _validate_before_run(self) -> Optional[str]:
        params = self.build_parameters()
        predictors = params["predictors"]

        if not predictors:
            return "Add at least one predictor layer."

        for pred in predictors:
            layer = QgsProject.instance().mapLayer(pred["layer_id"])
            if layer is None:
                return "One of the selected predictor layers is no longer available."
            if pred["layer_type"] == "vector" and not pred["field_name"]:
                return f"Select a field for vector predictor '{layer.name()}'."

        if params["training_mode"] == "clicked":
            if len(params["clicked_points"]) == 0:
                return "Capture at least one map point for clicked-point training mode."
        else:
            point_layer_id = params["point_layer_id"]
            point_layer = QgsProject.instance().mapLayer(point_layer_id) if point_layer_id else None
            if point_layer is None:
                return "Select a point layer for layer-based training mode."
            if point_layer.featureCount() == 0:
                return "Selected point layer has no features."

        return None

    def accept(self):
        error = self._validate_before_run()
        if error:
            QMessageBox.warning(self, "TerraMatch", error)
            return
        try:
            self._accepted_params = self.build_parameters()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "TerraMatch",
                f"Failed to capture dialog parameters:\n{exc}",
            )
            return
        self._stop_capture_mode()
        super().accept()

    def reject(self):
        self._accepted_params = None
        self._stop_capture_mode()
        super().reject()

    def accepted_parameters(self) -> Optional[Dict]:
        return self._accepted_params

    def closeEvent(self, event):
        self._stop_capture_mode()
        self.clear_captured_points()
        project = QgsProject.instance()
        try:
            project.layersAdded.disconnect(self.refresh_layers)
        except Exception:
            pass
        try:
            project.layersRemoved.disconnect(self.refresh_layers)
        except Exception:
            pass
        super().closeEvent(event)
