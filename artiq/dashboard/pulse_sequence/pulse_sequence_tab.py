import labrad
import asyncio
from PyQt5 import QtCore, QtGui, QtWidgets
from artiq import __artiq_dir__ as artiq_dir
from artiq.dashboard.pulse_sequence import pulse_sequence_visualizer


class PulseSequenceTab(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        qfm = QtGui.QFontMetrics(self.font())
        self.resize(140*qfm.averageCharWidth(), 38*qfm.lineSpacing())
        self.exit_request = asyncio.Event()
        self.dock = pulse_sequence_visualizer.PulseSequenceVisualizer()
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.dock)
        self.dock.show()

    def closeEvent(self, event):
        event.ignore()
        self.exit_request.set()

    def save_state(self):
        return {
            "state": bytes(self.saveState()),
            "geometry": bytes(self.saveGeometry())
        }

    def restore_state(self, state):
        self.restoreGeometry(QtCore.QByteArray(state["geometry"]))
        self.restoreState(QtCore.QByteArray(state["state"]))