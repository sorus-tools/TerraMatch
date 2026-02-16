import os
import traceback

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QCursor, QIcon
from qgis.PyQt.QtWidgets import QAction, QApplication, QDialog
from qgis.core import Qgis, QgsApplication, QgsProject, QgsRasterLayer
from qgis.gui import QgsVertexMarker
try:
    from qgis.PyQt import sip
except Exception:  # pragma: no cover
    sip = None

from .processing_provider import TerraMatchProcessingProvider
from .processor import TerraMatchProcessor
from .run_report import (
    TerraMatchRunReport,
    default_report_path_for_output,
)
from .terramatch_dialog import TerraMatchDialog


class TerraMatchPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dialog = None
        self.processing_provider = None

    def initGui(self):
        self._purge_canvas_markers(include_legacy=True)
        self.action = QAction("Run TerraMatch", self.iface.mainWindow())
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            if not icon.isNull():
                self.action.setIcon(icon)
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&TerraMatch", self.action)
        self._register_processing_provider()

    def unload(self):
        self._unregister_processing_provider()
        self._purge_canvas_markers(include_legacy=True)
        if self.dialog is not None:
            try:
                self.dialog.force_cleanup()
            except Exception:
                pass
            try:
                self.dialog.close()
            except Exception:
                pass
            self.dialog = None

        if self.action is None:
            return
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&TerraMatch", self.action)
        self.action = None

    def _register_processing_provider(self):
        try:
            registry = QgsApplication.processingRegistry()
        except Exception:
            registry = None
        if registry is None:
            return
        if self.processing_provider is None:
            self.processing_provider = TerraMatchProcessingProvider(self.iface)
        try:
            added = registry.addProvider(self.processing_provider)
            if not added:
                self.processing_provider = None
        except Exception:
            self.processing_provider = None

    def _unregister_processing_provider(self):
        if self.processing_provider is None:
            return
        try:
            registry = QgsApplication.processingRegistry()
            if registry is not None:
                registry.removeProvider(self.processing_provider)
        except Exception:
            pass
        self.processing_provider = None

    def _purge_canvas_markers(self, include_legacy: bool):
        if self.iface is None:
            return
        canvas = self.iface.mapCanvas()
        if canvas is None:
            return
        scene = canvas.scene()
        if scene is None:
            return

        legacy_red = QColor("red").name().lower()
        for item in list(scene.items()):
            if not isinstance(item, QgsVertexMarker):
                continue

            is_tagged = False
            try:
                is_tagged = item.data(0) == TerraMatchDialog.MARKER_TAG
            except Exception:
                is_tagged = False

            is_legacy = False
            if include_legacy:
                try:
                    is_legacy = (
                        item.iconType() == QgsVertexMarker.ICON_CROSS
                        and item.iconSize() == 9
                        and item.penWidth() == 3
                        and item.color().name().lower() == legacy_red
                    )
                except Exception:
                    is_legacy = False

            if is_tagged or is_legacy:
                try:
                    scene.removeItem(item)
                except Exception:
                    pass

    def run(self):
        if self.dialog is not None:
            try:
                if sip is not None and sip.isdeleted(self.dialog):
                    self.dialog = None
            except Exception:
                self.dialog = None

        if self.dialog is not None:
            try:
                self.dialog.setVisible(True)
                self.dialog.setWindowState(Qt.WindowNoState)
                self.dialog.showNormal()
                self.dialog.raise_()
                self.dialog.activateWindow()
                QApplication.processEvents()
                if self.dialog.isVisible():
                    return
            except Exception:
                pass
            try:
                self.dialog.close()
            except Exception:
                pass
            self.dialog = None

        self.dialog = TerraMatchDialog(self.iface)
        self.dialog.setModal(False)
        self.dialog.requestShow.connect(self._restore_dialog_window)
        self.dialog.accepted.connect(self._on_dialog_accepted)
        self.dialog.rejected.connect(self._on_dialog_rejected)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _restore_dialog_window(self):
        if self.dialog is None:
            return
        try:
            self.dialog.setVisible(True)
            self.dialog.setWindowState(Qt.WindowNoState)
            self.dialog.showNormal()
            self.dialog.raise_()
            self.dialog.activateWindow()
        except Exception:
            pass

    def _on_dialog_accepted(self):
        dialog = self.dialog
        if dialog is not None and getattr(dialog, "_capturing", False):
            # Ignore accidental accept signals during capture mode transitions.
            return
        self.dialog = None
        if dialog is None:
            return
        try:
            params = dialog.accepted_parameters() or dialog.build_parameters()
        except Exception as exc:
            report_path = default_report_path_for_output(None)
            report = TerraMatchRunReport(report_path)
            report.section("Run Configuration")
            report.exception(exc, context="Failed to read accepted dialog parameters")
            try:
                report.write()
            except Exception:
                pass
            self.iface.messageBar().pushMessage(
                "TerraMatch",
                f"Unexpected error: {exc}",
                level=Qgis.Critical,
                duration=10,
            )
            try:
                dialog.force_cleanup()
            except Exception:
                pass
            return

        try:
            dialog.force_cleanup()
        except Exception:
            pass
        self._execute_run(params)

    def _on_dialog_rejected(self):
        dialog = self.dialog
        if dialog is not None and getattr(dialog, "_capturing", False):
            # Hiding/minimizing for capture can trigger rejection on some platforms.
            # Keep the dialog reference alive so restore-on-finish works.
            return
        self.dialog = None
        if dialog is None:
            return
        try:
            dialog.force_cleanup()
        except Exception:
            pass

    def _execute_run(self, params):

        report_path = None
        report = None
        success = False
        output_path = None

        try:
            report_path = default_report_path_for_output(params.get("output_dir"))
            report = TerraMatchRunReport(report_path)
            report.section("Run Configuration")
            report.info(f"predictor_count={len(params.get('predictors', []))}")
            report.info(f"training_mode={params.get('training_mode')}")
            report.info(f"output_dir={params.get('output_dir') or '(temporary)'}")

            processor = TerraMatchProcessor()
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            try:
                output_path = processor.run(params, report)
                success = True
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as exc:
            if report is None:
                report_path = default_report_path_for_output(None)
                report = TerraMatchRunReport(report_path)
                report.section("Run Configuration")
                report.info("Parameters could not be fully captured.")
            report.exception(exc, context="TerraMatch plugin error")
            report.error(traceback.format_exc())
            self.iface.messageBar().pushMessage(
                "TerraMatch",
                f"Unexpected error: {exc}",
                level=Qgis.Critical,
                duration=10,
            )
        finally:
            try:
                if report is not None:
                    report.write()
            except Exception:
                pass

        if success and output_path and os.path.exists(output_path):
            layer_name = os.path.splitext(os.path.basename(output_path))[0]
            out_layer = QgsRasterLayer(output_path, layer_name)
            if out_layer.isValid():
                QgsProject.instance().addMapLayer(out_layer)
                self.iface.messageBar().pushMessage(
                    "TerraMatch",
                    f"Suitability raster created: {output_path}",
                    level=Qgis.Success,
                    duration=6,
                )
            else:
                self.iface.messageBar().pushMessage(
                    "TerraMatch",
                    f"Raster written but failed to load in QGIS: {output_path}",
                    level=Qgis.Warning,
                    duration=8,
                )
        else:
            self.iface.messageBar().pushMessage(
                "TerraMatch",
                "Processing failed. See run report for details.",
                level=Qgis.Critical,
                duration=10,
            )
