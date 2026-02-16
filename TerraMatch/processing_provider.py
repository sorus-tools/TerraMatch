import os
import traceback
from typing import Dict

from qgis.PyQt.QtWidgets import QDialog
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputString,
    QgsProcessingProvider,
    QgsProject,
    QgsRasterLayer,
)

from .processor import TerraMatchProcessor
from .run_report import TerraMatchRunReport, default_report_path_for_output
from .terramatch_dialog import TerraMatchDialog


class TerraMatchToolboxAlgorithm(QgsProcessingAlgorithm):
    OUTPUT_RASTER = "OUTPUT_RASTER"
    REPORT_PATH = "REPORT_PATH"

    def __init__(self, iface=None):
        super().__init__()
        self._iface = iface

    def name(self) -> str:
        return "terramatch"

    def displayName(self) -> str:
        return "TerraMatch"

    def group(self) -> str:
        return "TerraMatch"

    def groupId(self) -> str:
        return "terramatch"

    def shortHelpString(self) -> str:
        return (
            "Launches the TerraMatch dialog and runs suitability modeling from selected predictors "
            "and training points. The resulting binary raster is added to the map."
        )

    def initAlgorithm(self, config=None):
        self.addOutput(QgsProcessingOutputString(self.OUTPUT_RASTER, "Output suitability raster"))
        self.addOutput(QgsProcessingOutputString(self.REPORT_PATH, "Run report path"))

    def createInstance(self):
        return TerraMatchToolboxAlgorithm(self._iface)

    def flags(self):
        return super().flags() | QgsProcessingAlgorithm.FlagNoThreading

    def processAlgorithm(self, parameters, context, feedback) -> Dict[str, str]:
        if self._iface is None:
            raise QgsProcessingException("TerraMatch requires the QGIS desktop interface (iface).")

        dialog = TerraMatchDialog(self._iface, self._iface.mainWindow())
        dialog.setModal(True)

        result = dialog.exec()
        if result != QDialog.Accepted:
            try:
                dialog.force_cleanup()
            except Exception:
                pass
            feedback.pushInfo("TerraMatch canceled by user.")
            return {self.OUTPUT_RASTER: "", self.REPORT_PATH: ""}

        try:
            params = dialog.accepted_parameters() or dialog.build_parameters()
        except Exception as exc:
            try:
                dialog.force_cleanup()
            except Exception:
                pass
            raise QgsProcessingException(f"Failed to read TerraMatch parameters: {exc}")

        try:
            dialog.force_cleanup()
        except Exception:
            pass

        report_path = default_report_path_for_output(params.get("output_dir"))
        report = TerraMatchRunReport(report_path)
        report.section("Run Configuration")
        report.info(f"predictor_count={len(params.get('predictors', []))}")
        report.info(f"training_mode={params.get('training_mode')}")
        report.info(f"output_dir={params.get('output_dir') or '(temporary)'}")

        output_path = ""
        try:
            output_path = TerraMatchProcessor().run(params, report)
        except Exception as exc:
            report.exception(exc, context="TerraMatch processing algorithm error")
            report.error(traceback.format_exc())
            try:
                report.write()
            except Exception:
                pass
            raise QgsProcessingException(str(exc))

        try:
            report.write()
        except Exception:
            pass

        if output_path and os.path.exists(output_path):
            layer_name = os.path.splitext(os.path.basename(output_path))[0]
            out_layer = QgsRasterLayer(output_path, layer_name)
            if out_layer.isValid():
                QgsProject.instance().addMapLayer(out_layer)
            else:
                feedback.reportError(
                    f"Output raster was written but could not be loaded in QGIS: {output_path}"
                )

        return {self.OUTPUT_RASTER: output_path, self.REPORT_PATH: report_path}


class TerraMatchProcessingProvider(QgsProcessingProvider):
    def __init__(self, iface=None):
        super().__init__()
        self._iface = iface

    def id(self) -> str:
        return "sorus_terramatch"

    def name(self) -> str:
        return "SORUS"

    def longName(self) -> str:
        return self.name()

    def loadAlgorithms(self):
        self.addAlgorithm(TerraMatchToolboxAlgorithm(self._iface))
