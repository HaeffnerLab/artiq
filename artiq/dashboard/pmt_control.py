import labrad
import os
from labrad.units import WithUnit as U
import logging
from PyQt5 import QtCore, QtWidgets, QtGui
from artiq.protocols.pc_rpc import Client
from artiq.protocols import pyon
from twisted.internet.defer import inlineCallbacks
from runpy import run_path


logger = logging.getLogger(__name__)


class PMTControlDock(QtWidgets.QDockWidget):
    def __init__(self, acxn):
        QtWidgets.QDockWidget.__init__(self, "Manual Controls")
        self.setObjectName("pmt_control")
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable)
        self.pv = None
        self.pm = None
        self.bb = None
        self.acxn = acxn
        self.setup_listeners()

        self.dset_ctl = Client("::1", 3251, "master_dataset_db")
        self.scheduler = Client("::1", 3251, "master_schedule")
        self.dataset_db = Client("::1", 3251, "master_dataset_db")
        self.rid = None
        self.pulsed = False
        self.expid_continuous = {
            "arguments": {},
            "class_name": "pmt_collect_continuously",
            "file": "run_continuously/run_pmt_continuously.py",
            "log_level": 30,
            "repo_rev": None,
            "priority": 0
        }

        self.expid_pulsed = {
            "arguments": {},
            "class_name": "pmt_collect_pulsed",
            "file": "run_continuously/run_pmt_pulsed.py",
            "log_level": 30,
            "repo_rev": None,
            "priority": 0
        }

        self.expid_ttl = {
            "class_name": "change_ttl",
            "file": "misc/manual_ttl_control.py",
            "log_level": 30,
            "repo_rev": None,
            "priority": 1
        }

        self.expid_dds = {
            "arguments": {},
            "class_name": "change_cw",
            "file": "misc/manual_dds_control.py",
            "log_level": 30,
            "repo_rev": None,
            "priority": 1
        }

        self.expid_dc = {
            "arguments": {},
            "class_name": "set_dopplercooling_and_statereadout",
            "file": "misc/set_dopplercooling_and_statereadout.py",
            "log_level": 30,
            "repo_rev": None,
            "priority": 2
        }

        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout()
        pmt_frame = self.create_pmt_frame()
        linetrigger_frame = self.create_linetrigger_frame()
        dds_frame = self.create_dds_frame()
        picomotor_frame = self.create_picomotor_frame()
        layout.addWidget(pmt_frame)
        layout.addWidget(dds_frame)
        layout.addWidget(linetrigger_frame)
        layout.addWidget(picomotor_frame)
        layout.setSpacing(50)
        layout.setContentsMargins(0, 50, 0, 50)
        frame.setLayout(layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(False)
        scroll.setWidget(frame)
        scroll.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        scroll.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                             QtWidgets.QSizePolicy.MinimumExpanding)
        self.setWidget(scroll)

        self.connect_servers()

    def create_pmt_frame(self):
        pmtLabel = boldLabel("PMT")
        self.onButton = QtWidgets.QPushButton("On")
        self.onButton.setCheckable(True)
        self.onButton.clicked[bool].connect(self.set_state)
        self.setDCButton = QtWidgets.QPushButton("set")
        self.setDCButton.clicked.connect(self.set_dc_and_state_readout)
        self.clearPMTPlotButton = QtWidgets.QPushButton("clear")
        self.clearPMTPlotButton.clicked.connect(self.clear_pmt_plot)
        self.autoLoadButton = QtWidgets.QPushButton("On")
        self.autoLoadButton.setCheckable(True)
        self.autoLoadButton.clicked[bool].connect(self.toggle_autoload)
        self.autoLoadSpin = customIntSpinBox(0, (0, 1000000))
        self.autoLoadCurrentSpin = customSpinBox(0, (0, 10), " A")
        self.countDisplay = QtWidgets.QLCDNumber()
        self.countDisplay.setSegmentStyle(2)
        self.countDisplay.display(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateLCD)
        self.timer.start(250)
        self.countDisplay.setStyleSheet("background-color: lightGray;"
                                        "color: green;")
        self.unitsLabel = QtWidgets.QLabel("kcounts / s")

        self.durationLabel = QtWidgets.QLabel("Duration (ms): ")
        self.duration = QtWidgets.QLineEdit("100")
        try:
            with labrad.connection() as cxn:
                p = cxn.parametervault
                p.set_parameter(["PmtReadout", "duration", U(100, "ms")])
        except:
            logger.error("Failed to initially connect to labrad.")
            self.duration.setDisabled(True)
        validator = QtGui.QDoubleValidator()
        self.duration.setValidator(validator)
        self.duration.returnPressed.connect(self.duration_changed)
        self.duration.setStyleSheet("QLineEdit { background-color:  #c4df9b}" )
        self.modeLabel = QtWidgets.QLabel("Mode: ")
        self.setMode = customComboBox(["continuous", "pulsed"])
        self.setMode.currentIndexChanged.connect(self.set_mode)
        layout = QtWidgets.QGridLayout()
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        frame.setLineWidth(2)
        frame.setMidLineWidth(3)
        layout.addWidget(pmtLabel, 0, 0, 1, 3)
        layout.addWidget(self.onButton, 1, 0)
        layout.addWidget(self.countDisplay, 1, 1)
        layout.addWidget(self.unitsLabel, 1, 2)
        layout.addWidget(self.durationLabel, 2, 0)
        layout.addWidget(self.duration, 2, 1, 1, 2)
        layout.addWidget(self.modeLabel, 3, 0)
        layout.addWidget(self.setMode, 3, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel("Autoload: "), 4, 0)
        layout.addWidget(self.autoLoadButton, 4, 1)
        layout.addWidget(self.autoLoadSpin, 4, 2)
        layout.addWidget(QtWidgets.QLabel("Current: "), 5, 0)
        layout.addWidget(self.autoLoadCurrentSpin, 5, 1)
        dcLabel = QtWidgets.QLabel("Set Doppler cooling and state readout: ")
        layout.addWidget(dcLabel, 6, 0, 1, 2)
        layout.addWidget(self.setDCButton, 6, 2)
        # clearLabel = QtWidgets.QLabel("Reset PMT plot: ")
        # layout.addWidget(clearLabel, 7, 0)
        # layout.addWidget(self.clearPMTPlotButton, 7, 1, 1, 2)
        frame.setLayout(layout)
        return frame

    def create_linetrigger_frame(self):
        linetriggerLabel = boldLabel("LINETRIGGER")
        self.linetriggerButton = QtWidgets.QPushButton("Off")
        self.linetriggerButton.setCheckable(True)
        self.linetriggerButton.setChecked(True)
        self.linetriggerButton.clicked[bool].connect(self.toggle_linetrigger)
        self.linetriggerLineEdit = QtWidgets.QLineEdit("0")
        self.linetriggerLineEdit.returnPressed.connect(self.linetrigger_duration_changed)
        layout = QtWidgets.QGridLayout()
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        frame.setLineWidth(2)
        frame.setMidLineWidth(3)
        layout.addWidget(linetriggerLabel, 0, 0, 1, 3)
        layout.addWidget(self.linetriggerButton, 1, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Offset duration (us): "), 2, 0)
        layout.addWidget(self.linetriggerLineEdit, 2, 1, 1, 2)
        frame.setLayout(layout)
        return frame

    def create_picomotor_frame(self):
        layout = QtWidgets.QGridLayout()
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        frame.setLineWidth(2)
        frame.setMidLineWidth(3)
        piezoLabel = boldLabel("PICOMOTOR")
        ctls = ["Local X", "Local Y", "Global X", "Global Y"]
        self.piezoStepSize = dict()
        self.piezoCurrentPos = dict()
        self.piezoLastPos = dict()
        layout.addWidget(piezoLabel, 0, 0, 1, 3)
        for i, ctl in enumerate(ctls):
            layout.addWidget(QtWidgets.QLabel(ctl + ": "), i + 1, 0)
            self.piezoStepSize[ctl] = customIntSpinBox(0, (0, 300))
            self.piezoStepSize[ctl].setToolTip("Set step size.")
            self.piezoStepSize[ctl].setObjectName(ctl)
            self.piezoStepSize[ctl].setKeyboardTracking(False)
            self.piezoStepSize[ctl].valueChanged.connect(self.piezo_step_size_changed)
            self.piezoStepSize[ctl].setRange(0, 300)
            layout.addWidget(self.piezoStepSize[ctl], i + 1, 1)
            self.piezoCurrentPos[ctl] = QtWidgets.QSpinBox()
            self.piezoCurrentPos[ctl].setSingleStep(0)
            self.piezoCurrentPos[ctl].setToolTip("Current position.")
            self.piezoCurrentPos[ctl].setRange(-100000, 100000)
            self.piezoCurrentPos[ctl].setObjectName(str(i + 1))
            self.piezoLastPos[i + 1] = 0
            self.piezoCurrentPos[ctl].setKeyboardTracking(False)
            self.piezoCurrentPos[ctl].valueChanged.connect(self.piezo_changed)
            layout.addWidget(self.piezoCurrentPos[ctl], i + 1, 2)
        frame.setLayout(layout)
        return frame

    def create_dds_frame(self):
        layout = QtWidgets.QGridLayout()
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        frame.setLineWidth(2)
        frame.setMidLineWidth(3)
        ddsLabel = boldLabel("DDS Control")
        layout.addWidget(ddsLabel, 0, 0, 1, 2)
        home_dir = os.path.expanduser("~")
        dir_ = os.path.join(home_dir, "artiq-master/HardwareConfiguration.py")
        settings = run_path(dir_)
        dds_dict = settings["dds_config"]
        self.all_dds_specs = dict()
        self.dds_widgets = dict()

        try:
            cxn = labrad.connect()
            p = cxn.parametervault
            for (name, specs) in dds_dict.items():
                try:
                    params = p.get_parameter(["dds_cw_parameters", name])
                    freq, amp, state, att = params[1]
                except:
                    freq = str(specs.default_freq)
                    att = str(specs.default_att)
                    amp = str(1.)
                    state = str(0)
                    p.new_parameter("dds_cw_parameters", name,
                                    ("cw_settings", [freq, amp, state, att]))
                self.all_dds_specs[name] = {"cpld": int(specs.urukul),
                                            "frequency": float(freq) * 1e6,
                                            "att": float(att),
                                            "state": bool(int(state)),
                                            "amplitude": float(amp)}
            for i, (name, specs) in enumerate(sorted(dds_dict.items())):
                widget = ddsControlWidget(name, specs, self.scheduler, self)
                layout.addWidget(widget, i // 2 + 1 , i % 2)
                self.dds_widgets[name] = widget
            frame.setLayout(layout)
            cxn.disconnect()
        except:
            logger.error("Failed to initially connect to labrad.")

        return frame

    def set_state(self, override=False):
        if override:
            flag = True
        else:
            flag = self.onButton.isChecked()

        if flag:
            if self.rid is None:
                if self.setMode.currentText() == "continuous":
                    self.rid = self.scheduler.submit("main", self.expid_continuous, 0)
                else:
                    self.rid = self.scheduler.submit("main", self.expid_pulsed, 0)
            self.onButton.setText("Off")

        else:
            if self.rid is None:
                return  # Shouldn't happen
            else:
                self.scheduler.request_termination(self.rid)
                self.rid = None
            self.onButton.setText("On")

    def set_dc_and_state_readout(self):
        self.scheduler.submit("main", self.expid_dc, 2)

    def clear_pmt_plot(self):
        self.dataset_db.set("clear_pmt_plot", True)

    @inlineCallbacks
    def duration_changed(self, *args, **kwargs):
        # connect to parametervault here
        if self.pv is None:
            self.pv = yield self.acxn.get_server("ParameterVault")
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        if state == QtGui.QValidator.Acceptable:
            color = "#c4df9b" # green
        elif state == QtGui.QValidator.Intermediate:
            color = "#fff79a" # yellow
        else:
            color = "#f6989d" # red
        sender.setStyleSheet("QLineEdit { background-color: %s }" %color)
        try:
            min = 1e-3 # 1 us
            raw_duration = float(sender.text())
            duration = raw_duration if raw_duration >= min else min
            yield self.pv.set_parameter(["PmtReadout", "duration", U(duration, "ms")])
            a = yield self.pv.get_parameter(["PmtReadout", "duration"])
            if self.rid is None:
                return
            else:
                self.scheduler.request_termination(self.rid)
                self.rid = None
                self.set_state(True)
        except ValueError:
            # Shouldn't happen
            yield print("")
            logger.warning("Problem trying to update duration", exc_info=True)

    def set_mode(self):
        txt = str(self.setMode.currentText())
        if self.rid is None:
            self.pulsed = txt == "pulsed"
        elif  txt == "continuous":
            if not self.pulsed:
                return
            else:
                self.pulsed = False
                self.scheduler.request_termination(self.rid)
                self.rid = self.scheduler.submit("main", self.expid_continuous, 0)
        else:
            if self.pulsed:
                return
            else:
                self.pulsed = True
                self.scheduler.request_termination(self.rid)
                self.rid = self.scheduler.submit("main", self.expid_pulsed, 0)

    def updateLCD(self):
        if not self.onButton.isChecked():
            self.countDisplay.display(0)
            return
        try:
            raw_val = self.dset_ctl.get("pmt_counts")[-1]
            try:
                # duration in mseconds
                duration = float(self.duration.text())
            except ValueError:
                # picked up a backspace or something
                logger.warning("Failed to update LCD", exc_info=True)
                return
            val = raw_val / duration  # kcounts / second
            self.countDisplay.display(val)
        except KeyError:
            # dataset doesn't exist
            logger.info("dataset doesn't exist yet")
            self.countDisplay.display(0)
        except IndexError:
            # timer too fast
            pass

    @inlineCallbacks
    def toggle_linetrigger(self, *args):
        sender = self.sender()
        flag = sender.isChecked()
        if flag:
            sender.setText("Off")
            yield self.pv.set_parameter(["line_trigger_settings", "enabled", True])
        else:
            sender.setText("On")
            yield self.pv.set_parameter(["line_trigger_settings", "enabled", False])

    @inlineCallbacks
    def linetrigger_duration_changed(self, *args):
        value = float(self.sender().text())
        yield self.pv.set_parameter(["line_trigger_settings", "offset_duration", value])

    def piezo_step_size_changed(self):
        sender = self.sender()
        step = int(sender.value())
        ctl = sender.objectName()
        self.piezoCurrentPos[ctl].setSingleStep(step)

    @inlineCallbacks
    def piezo_changed(self, *args):
        if self.pm is None:
            yield print("not connected to picomotor")
        sender = self.sender()
        piezo = int(sender.objectName())
        current_pos = int(sender.value())
        last_pos = self.piezoLastPos[piezo]
        self.piezoLastPos[piezo] = current_pos
        move = current_pos - last_pos
        yield self.pm.relative_move(piezo, move)

    @inlineCallbacks
    def toggle_autoload(self, *args):
        sender = self.autoLoadButton
        flag = sender.isChecked()
        if flag:
            try:
                self.check_pmt_data_length = len(self.dataset_db.get("pmt_counts"))
            except KeyError:
                sender.setChecked(False)
                return
            sender.setText("Off")
            self.expid_ttl.update({"arguments": {"device": "blue_PIs",
                                                 "state": True}})
            if not hasattr(self, "check_pmt_timer"):
                self.check_pmt_timer = QtCore.QTimer()
                self.check_pmt_timer.timeout.connect(self.check_pmt_counts)
            self.check_pmt_timer.start(100)
            yield self.bb.connect()
            yield self.bb.set_current(self.autoLoadCurrentSpin.value())
            yield self.bb.on()
        else:
            sender.setText("On")
            if not hasattr(self, "check_pmt_timer"):
                return
            self.check_pmt_timer.stop()
            self.expid_ttl.update({"arguments": {"device": "blue_PIs",
                                                 "state": False}})
            yield self.bb.off()
        self.scheduler.submit("main", self.expid_ttl, priority=1)

    def check_pmt_counts(self):
        try:
            counts = self.dataset_db.get("pmt_counts")[self.check_pmt_data_length:]
        except KeyError:
            return
        if len(counts) == 0:
            return
        if max(counts) > int(self.autoLoadSpin.value()):
            self.autoLoadButton.setChecked(False)
            self.autoLoadButton.clicked.emit()

    def save_state(self):
        return {"ctl": {ctl: self.piezoStepSize[ctl].value()
                        for ctl in self.piezoStepSize.keys()},
                "offset":  self.linetriggerLineEdit.text(),
                "autoload": self.autoLoadSpin.value(),
                "mode": self.setMode.currentText(),
                "ltrigger": self.linetriggerButton.isChecked(),
                "current": self.autoLoadCurrentSpin.value()}

    def restore_state(self, state):
        for ctl, value in state["ctl"].items():
            self.piezoStepSize[ctl].setValue(value)
            self.piezoCurrentPos[ctl].setSingleStep(int(value))
        self.linetriggerLineEdit.setText(state["offset"])
        self.autoLoadSpin.setValue(int(state["autoload"]))
        self.setMode.setCurrentText(state["mode"])
        self.linetriggerButton.setChecked(state["ltrigger"])
        d = {False: "On", True: "Off"}
        self.linetriggerButton.setText(d[state["ltrigger"]])
        self.autoLoadCurrentSpin.setValue(state["current"])

    def setup_listeners(self):
        self.acxn.add_on_connect("ParameterVault", self.parameter_vault_connect)
        self.acxn.add_on_disconnect("ParameterVault", self.parameter_vault_disconnect)
        self.acxn.add_on_connect("picomotorserver", self.picomotor_connect)
        self.acxn.add_on_disconnect("picomotorserver", self.picomotor_disconnect)
        self.acxn.add_on_connect("barebonese3663a", self.barebones_connect)
        self.acxn.add_on_disconnect("barebonese3663a", self.barebones_disconnect)

    def parameter_vault_connect(self):
        self.duration.setDisabled(False)

    def parameter_vault_disconnect(self):
        self.duration.setDisabled(True)

    def picomotor_connect(self):
        for spinbox in self.piezoStepSize.values():
            spinbox.setDisabled(False)
        for spinbox in self.piezoCurrentPos.values():
            spinbox.setDisabled(False)

    def picomotor_disconnect(self):
        for spinbox in self.piezoStepSize.values():
            spinbox.setDisabled(True)
        for spinbox in self.piezoCurrentPos.values():
            spinbox.setDisabled(True)

    def barebones_connect(self):
        self.autoLoadCurrentSpin.setDisabled(False)

    def barebones_disconnect(self):
        self.autoLoadCurrentSpin.setDisabled(True)

    @inlineCallbacks
    def connect_servers(self):
        if self.pv is None:
            try:
                self.pv = yield self.acxn.get_server("ParameterVault")
            except:
                self.parameter_vault_disconnect()
        if self.pm is None:
            try:
                self.pm = yield self.acxn.get_server("picomotorserver")
            except:
                self.picomotor_disconnect()
        if self.bb is None:
            try:
                self.bb = yield self.acxn.get_server("barebonese3663a")
            except:
                self.barebones_disconnect()


class ddsControlWidget(QtWidgets.QFrame):
    def __init__(self, name, specs, scheduler, parent):
        QtWidgets.QFrame.__init__(self)
        self.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Raised)

        self.parent = parent
        self.name = name
        self.freq = parent.all_dds_specs[name]["frequency"] * 1e-6
        self.att = parent.all_dds_specs[name]["att"]
        self.state = parent.all_dds_specs[name]["state"]
        self.cpld = specs.urukul
        self.amplitude = parent.all_dds_specs[name]["amplitude"]
        unum = str(int(specs.urukul))
        min_att, max_att = specs.min_att, specs.max_att
        min_freq, max_freq = specs.min_freq, specs.max_freq
        self.scheduler = scheduler
        self.expid = {"arguments": {"specs": pyon.encode(self.parent.all_dds_specs),
                                    "urukul_number": unum,
                                    "dds_name": name},
                      "class_name": "change_cw",
                      "file": "misc/manual_dds_control.py",
                      "log_level": 30,
                      "repo_rev": None,
                      "priority": 1}
        self.parameters_changed()

        layout = QtWidgets.QGridLayout()
        layout.addWidget(boldLabel(name), 0, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel("Frequency"), 1, 0)
        layout.addWidget(QtWidgets.QLabel("Amplitude"), 3, 0)
        layout.addWidget(QtWidgets.QLabel("Attenuation"), 3, 1)

        self.freq_spin = customSpinBox(self.freq, (min_freq, max_freq), " MHz")
        self.freq_spin.editingFinished.connect(self.freq_spin_changed)
        self.amp_spin = customSpinBox(1, (0, 1), None)
        self.amp_spin.editingFinished.connect(self.amp_spin_changed)
        self.att_spin = customSpinBox(self.att, (min_att, max_att), " dB")
        self.att_spin.editingFinished.connect(self.att_spin_changed)
        self.state_button = QtWidgets.QPushButton("O")
        self.state_button.setCheckable(True)
        self.state_button.toggled.connect(self.button_clicked)
        self.state_button.setChecked(self.state)

        layout.addWidget(self.freq_spin, 2, 0)
        layout.addWidget(self.amp_spin, 4, 0)
        layout.addWidget(self.att_spin, 4, 1)
        layout.addWidget(self.state_button, 2, 1)
        self.setLayout(layout)

    def button_clicked(self, val):
        bttn = self.sender()
        if val:
            bttn.setText("|")
        else:
            bttn.setText("O")
        self.state = val
        self.parameters_changed()

    def freq_spin_changed(self):
        self.freq = self.sender().value()
        self.parameters_changed()

    def att_spin_changed(self):
        self.att = self.sender().value()
        self.parameters_changed()

    def amp_spin_changed(self):
        self.amplitude = self.sender().value()
        self.parameters_changed()

    def parameters_changed(self):
        new_values = {"frequency": float(self.freq) * 1e6,
                      "att": float(self.att),
                      "state": self.state,
                      "cpld": int(self.cpld),
                      "amplitude": float(self.amplitude)}
        self.parent.all_dds_specs.update({self.name: new_values})
        self.parent.expid_dds["arguments"].update(
                {"specs": pyon.encode(self.parent.all_dds_specs)})
        if os.name == "nt":
            # Skip automatic DDS param updates when running from Windows
            return
        self.scheduler.submit("main", self.parent.expid_dds, priority=1)
        cxn = labrad.connect()
        p = cxn.parametervault
        p.set_parameter(["dds_cw_parameters", self.name,
                [str(self.freq), str(self.amplitude), str(int(self.state)), str(self.att)]])
        cxn.disconnect()


class boldLabel(QtWidgets.QLabel):
    def __init__(self, txt):
        QtWidgets.QLabel.__init__(self, txt)
        boldFont = QtGui.QFont()
        boldFont.setBold(True)
        self.setFont(boldFont)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class customSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, value, range_, suffix=""):
        QtWidgets.QDoubleSpinBox.__init__(self)
        self.setValue(value)
        self.setRange(*range_)
        if suffix is not None:
            self.setSuffix(suffix)
        self.setSingleStep(0.1)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def focusInEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        super(customSpinBox, self).focusInEvent(event)

    def focusOutEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        super(customSpinBox, self).focusOutEvent(event)

    def wheelEvent(self, event):
        if self.hasFocus():
            return super(customSpinBox, self).wheelEvent(event)
        else:
            event.ignore()


class customIntSpinBox(QtWidgets.QSpinBox):
    def __init__(self, value, range_):
        QtWidgets.QSpinBox.__init__(self)
        self.setValue(value)
        self.setRange(*range_)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def focusInEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        super(customIntSpinBox, self).focusInEvent(event)

    def focusOutEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        super(customIntSpinBox, self).focusOutEvent(event)

    def wheelEvent(self, event):
        if self.hasFocus():
            return super(customIntSpinBox, self).wheelEvent(event)
        else:
            event.ignore()


class customComboBox(QtWidgets.QComboBox):
    def __init__(self, items):
        QtWidgets.QComboBox.__init__(self)
        for item in items:
            self.addItem(item)

    def focusInEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        super(customComboBox, self).focusInEvent(event)

    def focusOutEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        super(customComboBox, self).focusOutEvent(event)

    def wheelEvent(self, event):
        if self.hasFocus():
            return super(customComboBox, self).wheelEvent(event)
        else:
            event.ignore()
