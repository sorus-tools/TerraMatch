import datetime as _dt
import os
import platform
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, Optional

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout


class TerraMatchRunReport:
    def __init__(self, report_path: str):
        self.report_path = os.path.abspath(report_path)
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._lines = []
        self._meta: Dict[str, Any] = {}

        self._append_line("TerraMatch Run Report")
        self._append_line(
            f"Started: {_dt.datetime.now().isoformat(sep=' ', timespec='seconds')}"
        )
        self._append_line(f"Host: {platform.node()}")
        self._append_line(f"OS: {platform.platform()}")
        self._append_line("")

    def set_meta(self, **kwargs: Any):
        with self._lock:
            self._meta.update(kwargs)

    def _append_line(self, line: str):
        with self._lock:
            self._lines.append(line)

    def _elapsed(self) -> str:
        return f"{time.perf_counter() - self._t0:0.2f}s"

    def section(self, title: str):
        self._append_line("")
        self._append_line(f"== {title} ==")

    def info(self, message: str):
        self._append_line(f"[{self._elapsed()}] INFO  {message}")

    def warn(self, message: str):
        self._append_line(f"[{self._elapsed()}] WARN  {message}")

    def error(self, message: str):
        self._append_line(f"[{self._elapsed()}] ERROR {message}")

    def exception(self, exc: BaseException, context: str = "Unhandled exception"):
        self.error(f"{context}: {exc}")
        self._append_line(traceback.format_exc())

    def write(self) -> str:
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with self._lock:
            lines = list(self._lines)
            meta = dict(self._meta)

        if meta:
            lines.append("")
            lines.append("== Metadata ==")
            for key in sorted(meta.keys()):
                lines.append(f"{key}: {meta[key]}")

        with open(self.report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n")
        return self.report_path


def default_report_path_for_output(output_dir: Optional[str]) -> str:
    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(out_dir, f"terramatch_run_report_{stamp}.txt")


def show_report_dialog(iface, report_path: str, title: str) -> None:
    if iface is None or not report_path or not os.path.exists(report_path):
        return

    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except Exception as exc:
        content = f"Unable to read report file:\n{exc}"

    dialog = QDialog(iface.mainWindow())
    dialog.setWindowTitle(title)
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)
    text_widget = QPlainTextEdit(dialog)
    text_widget.setReadOnly(True)
    text_widget.setLineWrapMode(QPlainTextEdit.NoWrap)
    text_widget.setPlainText(content)
    layout.addWidget(text_widget)

    button_row = QHBoxLayout()
    open_button = QPushButton("Open Report File", dialog)
    close_button = QPushButton("Close", dialog)

    open_button.clicked.connect(
        lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
    )
    close_button.clicked.connect(dialog.accept)

    button_row.addWidget(open_button)
    button_row.addStretch()
    button_row.addWidget(close_button)
    layout.addLayout(button_row)

    dialog.resize(900, 600)
    dialog.exec_()
