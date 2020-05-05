import asyncio
import labrad
import labrad.units as u
from labrad.units import WithUnit as U
from labrad.types import types as labradTypes
from lattice.clients.connection import connection
from ast import literal_eval
from decimal import Decimal
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from artiq.protocols.pc_rpc import Client
from artiq.gui.tools import LayoutWidget
import logging
from twisted.internet.defer import inlineCallbacks

logger = logging.getLogger(__name__)


# Read all .py files under artiq-work and add their attributes to a list
# of commonly_used_parameters.
# Note that this only gets parameters used by subsequences. 
# Parameters in accessed_params lists will not appear in this list
# (since those are presumably specific to a particular experiment).
commonly_used_parameters = []

import os
import imp
import importlib
import ast
work_folder = os.path.join(os.path.expanduser("~"), "artiq-work")
for path, subdirs, files in os.walk(work_folder):
    for filename in files:
        if filename.endswith(".py"):
            experiment_file = os.path.join(path, filename)
            classes = []
            try:
                with open(experiment_file) as f:
                    node = ast.parse(f.read())
                    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
                    experiment_source = imp.load_source('', experiment_file, f)
                    for class_ in classes:
                        sequence_class = getattr(experiment_source, class_.name)
                        d = sequence_class.__dict__
                        for key, value in d.items():
                            if type(value) == str:
                                try:
                                    c, v = value.split(".")
                                except AttributeError:
                                    continue
                                except ValueError:
                                    continue
                                commonly_used_parameters.append(value)
            except:
                continue

parameterchangedID = 612512
types = ["parameter",
         "scan",  # Not used here
         "line_selection",
      #  "sideband_selection",
         "selection_simple",
         "bool",
         "spectrum_sensitivity",  # Not currently being used?
      #  "string", fails when using python3 client but python2 pylabrad
         "int_list"]

class ParameterEditorDock(QtWidgets.QDockWidget):

    def __init__(self, acxn=None, name="Parameter Editor", accessed_params=None, expand_accessed_params=True):
        QtWidgets.QDockWidget.__init__(self, name)
        self.expand_accessed_params = expand_accessed_params
        self.acxn = acxn if acxn else connection()
        self.accessed_params = accessed_params
        self.setObjectName(name.replace(" ", "_"))
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable)
        self.exit_request = asyncio.Event()
        global types
        self.types = types

        self.cxn = None
        try:
            self.cxn = labrad.connect()
        except:
            logger.warning("Parameter Editor failed to connect to labrad.", exc_info=True)
            self.setDisabled(True)
        self.setup_listeners()
        self.make_GUI()

    def make_region_item(self, label):
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, label)
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(0, font)
        bgcolor = QtGui.QColor(192, 192, 192)
        item.setBackground(0, bgcolor)
        item.setBackground(1, bgcolor)
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsSelectable)
        self.table.addTopLevelItem(item)
        self.region_widget_items[label] = item
        return item

    def make_collection_item(self, registry, collection, region_index):
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, collection)
        font = QtGui.QFont()
        font.setBold(True)
        item.setFont(0, font)
        item.setBackground(0, QtGui.QColor(248, 248, 248))
        item.setBackground(1, QtGui.QColor(248, 248, 248))
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsSelectable)
        params = sorted(registry[collection].keys())
        num_children = 0
        for param in params:
            try:
                value = registry[collection][param]
                child = QtWidgets.QTreeWidgetItem([param, None])
                if num_children % 2 == 0:
                    child.setBackground(0, QtGui.QColor(248, 248, 248))
                else:
                    child.setBackground(0, QtGui.QColor(228, 228, 228))
                assert type(value) == tuple
                assert value[0] in self.types
                _child = EditorFactory.get_editor(value, self.acxn,
                                                  (collection, param), child)
                if _child is None:
                    # Unrecognized registry key format, ignore
                    continue
                item.addChild(child)
                size = QtCore.QSize()
                size.setHeight(15)
                child.setSizeHint(0, size)
                child.setBackground(1, QtGui.QColor(248, 248, 248))
                self.table.setItemWidget(child, 1, _child)
                num_children += 1
                self.param_widget_items[region_index][collection, param] = _child
            except (AssertionError, TypeError):
                # logger.info("Unrecognized parameter vault registry key, value "
                #             "pair format for: {}, {}".format(collection, param))
                continue
        if num_children > 0:
            return item
        return None

    def make_GUI(self):
        grid = LayoutWidget()
        self.setWidget(grid)
        self.table = QtWidgets.QTreeWidget()
        self.table.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.table.setSortingEnabled(False)
        self.table.header().setStretchLastSection(False)
        p = QtGui.QPalette()
        p.setColor(9, QtGui.QColor(248,248,248))
        self.setPalette(p)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_menu)
        self.table.setIndentation(10)
        grid.addWidget(self.table, 0, 0)

        self.table.setHeaderLabels(["Collection", "Value"])

        self.region_widget_items = dict()
        self.collection_widget_items = [dict(), dict()]
        self.param_widget_items = [dict(), dict()]

        if not self.cxn:
            return

        all_params_registry = dict()
        common_params_registry = dict()
        accessed_params_registry = dict()

        r = self.cxn["registry"]

        if self.accessed_params:
            # accessed parameters list
            for param in self.accessed_params:
                param_split = param.split(".")
                collection = param_split[0]
                param_name = param_split[1]
                if not collection in accessed_params_registry.keys():
                    accessed_params_registry[collection] = dict()
                r.cd("", "Servers", "Parameter Vault", collection)
                accessed_params_registry[collection][param_name] = r.get(param_name)
            for collection in sorted(accessed_params_registry.keys()):
                collection_item = self.make_collection_item(accessed_params_registry, collection, region_index=0)
                if collection_item:
                    self.table.addTopLevelItem(collection_item)
                    if self.expand_accessed_params:
                        collection_item.setExpanded(True)
                    else:
                        collection_item.setExpanded(False)
                    self.collection_widget_items[0][collection] = collection_item
        else:
            # set up top-level items for all parameters and common parameters
            region_common_params_item = self.make_region_item("Commonly Used Parameters")
            region_all_params_item = self.make_region_item("All Parameters")

            # common parameters list
            for param in commonly_used_parameters:
                param_split = param.split(".")
                collection = param_split[0]
                param_name = param_split[1]
                param_value = ""
                try:
                    r.cd("", "Servers", "Parameter Vault", collection)
                    param_value = r.get(param_name)
                except:
                    continue
                if not collection in common_params_registry.keys():
                    common_params_registry[collection] = dict()
                common_params_registry[collection][param_name] = param_value
            for collection in sorted(common_params_registry.keys()):
                collection_item = self.make_collection_item(common_params_registry, collection, region_index=0)
                if collection_item:
                    region_common_params_item.addChild(collection_item)
                    self.collection_widget_items[0][collection] = collection_item

            # all parameters list
            r.cd("", "Servers", "Parameter Vault")
            collections = r.dir()[0]
            for collection in collections:
                collection_dict = dict()
                r.cd("", "Servers", "Parameter Vault", collection)
                params = r.dir()[1]
                for param in params:
                    collection_dict[param] = r.get(param)
                all_params_registry[collection] = collection_dict
            for collection in sorted(all_params_registry.keys()):
                collection_item = self.make_collection_item(all_params_registry, collection, region_index=1)
                if collection_item:
                    region_all_params_item.addChild(collection_item)
                    self.collection_widget_items[1][collection] = collection_item

        self.table.setColumnWidth(0, 225)
        self.table.setColumnWidth(1, 150)
        self.table.header().setFocusPolicy(QtCore.Qt.NoFocus)

        self.cxn.disconnect()

    @inlineCallbacks
    def setup_listeners(self):
        try:
            yield self.acxn.connect(host='localhost', password='lab')
            context = yield self.acxn.context()
            p = yield self.acxn.get_server("ParameterVault")
            yield p.signal__parameter_change(parameterchangedID, context=context)
            yield p.addListener(listener=self.refresh_values, source=None,
                                ID=parameterchangedID, context=context)
        except:
            import traceback
            logger.warning(traceback.format_exc())
            logger.warning("failed to add parameter changed listener for dock: " + self.objectName())

    @inlineCallbacks
    def refresh_values(self, *args):
        loc = args[1]
        p = yield self.acxn.get_server("ParameterVault")
        for region_index in [0,1]:
            try:
                val = yield p.get_parameter(loc)
                self.param_widget_items[region_index][loc].update_value(val)
            except KeyError:
                pass

    def open_menu(self, position):
        menu = QtWidgets.QMenu()
        editAction = menu.addAction(self.tr("Edit Parameter"))
        editAction.triggered.connect(self.on_edit_action)
        newparamAction = menu.addAction(self.tr("New Parameter"))
        newparamAction.triggered.connect(self.on_newparam_action)
        saveparamsAction = menu.addAction(self.tr("Save Parameters to Registry"))
        saveparamsAction.triggered.connect(self.on_saveparams_action)
        # loadparamsAction = menu.addAction(self.tr("Load Parameters from Registry"))
        # loadparamsAction.triggered.connect(self.on_loadparams_action)
        menu.exec_(self.table.viewport().mapToGlobal(position))

    def on_edit_action(self, *params):
        if len(self.table.selectedItems()) == 0:
            return
        try:
            cxn = labrad.connect()
            r = cxn.registry
        except:
            logger.error("In trying to edit registry, failed to "
                         "connect to labrad.", exc_info=True)
            return
        sitem = self.table.selectedItems()[0]
        name = sitem.text(0)
        collection = sitem.parent().text(0)
        r.cd("", "Servers", "Parameter Vault", collection)
        item_info = r.get(name)
        self.edit_menu = editInputMenu(collection, name, item_info,
                                       cxn, self)
        self.edit_menu.show()

    def on_newparam_action(self):
        try:
            cxn = labrad.connect()
        except:
            logger.error("In trying to add parameter to registry "
                         "failed to connect to labrad.", exc_info=True)
            return
        self.new_param_menu = newParamMenu(cxn, self)
        self.new_param_menu.show()

    @inlineCallbacks
    def on_saveparams_action(self, *params):
        p = yield self.acxn.get_server("ParameterVault")
        yield p.save_parameters_to_registry()

    # @inlineCallbacks
    def on_loadparams_action(self, *params):
        pass
        # p = yield self.acxn.parametervault
        # yield p.reload_parameters()

    def closeEvent(self, event):
        event.ignore()
        self.exit_request.set()

    def save_state(self):
        dr = dict((x, y.isExpanded()) for x, y in self.region_widget_items.items())
        d0 = dict((x, y.isExpanded()) for x, y in self.collection_widget_items[0].items())
        d1 = dict((x, y.isExpanded()) for x, y in self.collection_widget_items[1].items())
        return {"scroll": self.table.verticalScrollBar().value(),
               "geometry": bytes(self.saveGeometry()),
               "expanded_r": dr,
               "expanded_0": d0,
               "expanded_1": d1}

    def restore_widget_items_state(self, state, state_key, widget_items):
        d = state[state_key]
        dkeys = widget_items.keys()
        for key, value in d.items():
            if key in dkeys and value:
                widget_items[key].setExpanded(True)

    def restore_state(self, state):
        self.restoreGeometry(QtCore.QByteArray(state["geometry"]))
        self.restore_widget_items_state(state, "expanded_r", self.region_widget_items)
        self.restore_widget_items_state(state, "expanded_0", self.collection_widget_items[0])
        self.restore_widget_items_state(state, "expanded_1", self.collection_widget_items[1])
        self.table.verticalScrollBar().setSliderPosition(state["scroll"])

class editInputMenu(QtWidgets.QDialog):
    def __init__(self, collection, name, item, cxn, parent=None):
        super(editInputMenu, self).__init__(parent)
        type_ = item[0]
        self.cxn = cxn
        self.name = name
        self.collection = collection
        self.item = item
        self.parent = parent
        self.setWindowTitle("Edit: {}.{}".format(collection,
                                                 name
                                                 ))
        topLayout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QGridLayout()
        okLayout = QtWidgets.QVBoxLayout()
        self.ok = QtWidgets.QPushButton("OK")
        self.ok.clicked.connect(self.on_ok_pressed)
        okLayout.addWidget(self.ok)
        global types
        if type_ in ["bool", "selection_simple"]:
            layout.addWidget(QtWidgets.QLabel("Nothing to edit."), 0, 0)

        elif type_ not in types:
            layout.addWidget(QtWidgets.QLabel("Unrecognized parameter type"),
                             0, 0)

        elif type_ == "parameter":
            min_, max_, current = item[1]
            try:
                self.units = current.units
            except (AttributeError, KeyError):
                self.units = ""
                min_ = U(float(min_), "")
                max_ = U(float(max_), "")
                current = U(current, "")
            self.min_spin = QtWidgets.QDoubleSpinBox()
            self.min_spin.setObjectName("min")
            self.min_spin.setMinimum(-1e20)
            self.min_spin.setMaximum(1e20)
            self.min_spin.setValue(min_[self.units])
            self.min_spin.setSuffix(" " + self.units)
            self.min_spin.setMaximum(current[self.units])
            self.max_spin = QtWidgets.QDoubleSpinBox()
            self.max_spin.setObjectName("max")
            self.max_spin.setMinimum(-1e20)
            self.max_spin.setMaximum(1e20)
            self.max_spin.setValue(max_[self.units])
            self.max_spin.setMinimum(current[self.units])
            self.max_spin.setSuffix(" " + self.units)
            self.max_spin.valueChanged.connect(self.on_minmax_change)
            self.min_spin.valueChanged.connect(self.on_minmax_change)
            layout.addWidget(QtWidgets.QLabel("Min: "), 0, 0)
            layout.addWidget(QtWidgets.QLabel("Max: "), 1, 0)
            layout.addWidget(self.min_spin, 0, 1)
            layout.addWidget(self.max_spin, 1, 1)

        elif type_ == "line_selection":
            self.ok.setAutoDefault(False)
            d = dict(self.item[1][1])
            self.inputs = []
            for i, key in enumerate(d.keys()):
                layout.addWidget(QtWidgets.QLabel("Edit Mapping"), 0, 0, 1, 2)
                layout.addWidget(QtWidgets.QLabel(key + ": "), i + 1, 0)
                line_edit = QtWidgets.QLineEdit()
                line_edit.setObjectName(key)
                line_edit.setText(d[key])
                line_edit.editingFinished.connect(self.on_line_selection_mapping_change)
                self.inputs.append(line_edit)
                layout.addWidget(line_edit, i + 1, 1)

        elif type_ == "int_list":
            layout.addWidget(QtWidgets.QLabel("No. Thresholds: "), 0, 0)
            self.spin_box = QtWidgets.QSpinBox()
            layout.addWidget(self.spin_box, 0, 1)
            self.spin_box.valueChanged.connect(self.no_thresholds_change)
            self.spin_box.setValue(len(self.item[1]))

        topLayout.addLayout(layout)
        topLayout.addLayout(okLayout)
        self.setLayout(topLayout)
        self.setMinimumWidth(300)

    def on_ok_pressed(self):
        self.cxn.disconnect()
        self.close()

    def on_minmax_change(self):
        sender = self.sender()
        idx = 0 if sender.objectName() == "min" else 1
        self.item[1][idx] = U(sender.value(), self.units)
        r = self.cxn.registry
        p = self.cxn.parametervault
        r.cd("", "Servers", "Parameter Vault", self.collection)
        for region_index in [0,1]:
            try:
                widget = self.parent.param_widget_items[region_index][self.collection, self.name]
                widget.min_ = self.item[1][0]#U(float(self.item[1][0]), self.units)
                widget.max_ =  self.item[1][1]#U(float(self.item[1][1]), self.units)
                widget.state = self.item[1]
            except KeyError:
                pass
        r.set(self.name, self.item)
        p.set_parameter(self.collection, self.name, self.item, True)

    def on_line_selection_mapping_change(self):
        sender = self.sender()
        key = sender.objectName()
        d = dict(self.item[1][1])
        d[key] = sender.text()
        self.item = (self.item[0], (self.item[1][0], list(d.items())))
        for region_index in [0,1]:
            try:
                widget = self.parent.param_widget_items[region_index][self.collection, self.name]
                widget.state = self.item[1]
                widget.clear()
                widget.addItems(d.values())
                widget.setCurrentIndex(widget.findText(d[self.item[1][0]]))
            except KeyError:
                pass
        r = self.cxn.registry
        p = self.cxn.parametervault
        r.cd("", "Servers", "Parameter Vault", self.collection)
        r.set(self.name, self.item)
        p.set_parameter(self.collection, self.name, self.item[1])

    def no_thresholds_change(self, n):
        val = self.item[1]
        diff = n - len(val)
        if diff == 0:
            return
        elif diff > 0:
            if len(self.item[1]) > 0:
                nmax = self.item[1][-1]
            else:
                nmax = 0
            for i in range(diff):
                val = np.append(val, nmax + (i + 1) * 2)
        elif diff < 0:
            val = val[:diff]
        for region_index in [0,1]:
            try:
                widget = self.parent.param_widget_items[region_index][self.collection, self.name]
                widget.update_value(val)
            except KeyError:
                pass
        r = self.cxn.registry
        p = self.cxn.parametervault
        p.set_parameter(self.collection, self.name, val)
        r.cd("", "Servers", "Parameter Vault", self.collection)
        r.set(self.name, (self.item[0], val))
        self.item = (self.item[0], val)


class newParamMenu(QtWidgets.QDialog):
    def __init__(self, cxn, parent):
        super(newParamMenu, self).__init__(parent)
        self.r = cxn.registry
        self.cxn = cxn
        layout = QtWidgets.QVBoxLayout()

        sublayout1 = QtWidgets.QHBoxLayout()
        sublayout1.addWidget(QtWidgets.QLabel("Collection: "))
        self.collection_combo = QtWidgets.QComboBox()
        self.r.cd("", "Servers", "Parameter Vault")
        collections = self.r.dir()[0]
        self.collection_combo.addItems(collections)
        sublayout1.addWidget(self.collection_combo)

        sublayout2 = QtWidgets.QHBoxLayout()
        self.key_label = QtWidgets.QLabel("key: ")
        self.key_label.setMinimumWidth(60)
        sublayout2.addWidget(self.key_label)
        self.key_edit = QtWidgets.QLineEdit()
        sublayout2.addWidget(self.key_edit)

        sublayout3 = QtWidgets.QHBoxLayout()
        self.value_label = QtWidgets.QLabel("value: ")
        self.value_label.setMinimumWidth(60)
        sublayout3.addWidget(self.value_label)
        self.value_edit = QtWidgets.QLineEdit()
        sublayout3.addWidget(self.value_edit)

        sublayout4 = QtWidgets.QHBoxLayout()
        self.ok = QtWidgets.QPushButton("Input")
        self.ok.clicked.connect(self.on_ok_pressed)
        sublayout4.addWidget(self.ok)

        layout.addLayout(sublayout1)
        layout.addLayout(sublayout2)
        layout.addLayout(sublayout3)
        layout.addLayout(sublayout4)
        self.setLayout(layout)

        self.setWindowTitle("Create New Parameter")
        self.finished.connect(self.closeEvent)

    def on_ok_pressed(self):
        collection = self.collection_combo.currentText()
        key = self.key_edit.text()
        value = self.value_edit.text()

        units = None
        for name in dir(u):
            s = " " + name
            if s + "," in value:
                units = name
                obj = getattr(u, name)
                base_units = obj.base_unit.name
                factor = "{:.1e}".format(Decimal(obj.factor)).split("e")[-1]
                value = value.replace(s, "e" + factor)
                break
        try:
            value = literal_eval(value)
        except:
            logger.error("Failed to add new parameter {}".format(value))
            return
        if not type(value) == tuple:
            return
        global types
        if not value[0] in types:
            return
        if value[0] == "parameter":
            for i, val in enumerate(value[1]):
                value[1][i] = U(val, base_units).inUnitsOf(units)
        self.r.cd("", "Servers", "Parameter Vault", collection)
        try:
            self.r.get(key)
        except:
            self.r.set(key, value)


    def closeEvent(self, event):
        self.cxn.disconnect()
        self.close()

class EditorFactory():
    editor = None
    editor_versions = dict()
    def __init__(self):
        self.p = None
        self.r = None

    @classmethod
    def register(cls):
        EditorFactory.editor_versions[cls.editor] = cls

    @classmethod
    def get_editor(cls, editor_info, acxn=None, descr=None, widget=None):
        editor_name, editor_settings = editor_info
        editor = EditorFactory.editor_versions.get(editor_name, None)
        if editor is None:
            return None
        return editor(editor_settings, acxn, descr, widget)

    def check_bounds(self, val):
        return True

    @inlineCallbacks
    def check_connection(self, cxn):
        try:
            if self.p is None:
                self.p = yield cxn.get_server("ParameterVault")
            else:
                yield self.p.ID
            if self.r is None:
                self.r = yield cxn.get_server("registry")
            else:
                yield self.r.ID
            self.setDisabled(False)
            return True
        except:
            self.setDisabled(True)
            return False

    def on_param_changed_locally(self, val):
        pass

    def update_value(self, val):
        pass


class BaseEditor(QtWidgets.QWidget, EditorFactory):
    def __init__(self, *params):
        self.state, self.acxn, descr, self.prt = params
        self.collection, self.name = descr

    def wheelEvent(self, *args, **kwargs):
        # Prevent widgets from interacting with mouse wheel
        pass


class BoolEditor(QtWidgets.QCheckBox, BaseEditor):
    editor = "bool"
    def __init__(self, *params):
        BaseEditor.__init__(self, *params)
        QtWidgets.QCheckBox.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setChecked(self.state)
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignVCenter)
        self.setLayout(layout)

        self.stateChanged.connect(self.on_param_changed_locally)

    @inlineCallbacks
    def on_param_changed_locally(self, val):
        if self.check_connection(self.acxn):
            yield self.p.set_parameter([self.collection, self.name, bool(val)])
            # HACK: avoid having to save parameters to registry through server
            # why do we even use the registry??
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, bool(val)))

    def update_value(self, val):
        if self.state == val:
            return
        self.setChecked(val)
        self.state = 2 if val else 0


class SelectionSimpleEditor(QtWidgets.QComboBox, BaseEditor):
    editor = "selection_simple"
    def __init__(self, *params):
        BaseEditor.__init__(self, *params)
        QtWidgets.QComboBox.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        if not type(self.state) == tuple:
            return
        selected, selections = self.state
        curr_idx = 0
        for i, selection in enumerate(selections):
            if selection == selected:
                curr_idx = i
            self.addItem(selection)
        self.setCurrentIndex(curr_idx)
        self.setSizeAdjustPolicy(self.AdjustToMinimumContentsLength)
        self.setFrame(False)

        self.currentIndexChanged.connect(self.on_param_changed_locally)

    @inlineCallbacks
    def on_param_changed_locally(self, idx):
        if self.check_connection(self.acxn):
            val = self.currentText()
            self.state = val, self.state[1]
            yield self.p.set_parameter([self.collection, self.name, self.state])
            # HACK: avoid having to save parameters to registry through server
            # why do we even use the registry??
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, self.state))

    def update_value(self, val):
        if self.state == val:
            return
        idx = self.findText(val)
        self.setCurrentIndex(idx)
        self.state = val, self.state[1]


class LineSelectionEditor(QtWidgets.QComboBox, BaseEditor):
    editor = "line_selection"
    def __init__(self, *params):
        BaseEditor.__init__(self, *params)
        QtWidgets.QComboBox.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        if (not type(self.state) == tuple or
            not type(self.state[1]) == labradTypes.LazyList):
            # Improper format
            return
        selected, selections = self.state
        d_selections = dict(selections)
        self.d_selections = d_selections
        for i, selection in enumerate(d_selections.values()):
            if d_selections[selected] == selection:
                curr_idx = i
            self.addItem(selection)
        self.setCurrentIndex(curr_idx)
        self.setFrame(False)

        self.currentIndexChanged.connect(self.on_param_changed_locally)

    @inlineCallbacks
    def on_param_changed_locally(self, idx):
        if self.check_connection(self.acxn):
            _val = self.currentText()
            if _val == "":
                return
            d = dict((x, y) for y, x in self.state[1])
            val = d[_val]
            self.state = val, self.state[1]
            yield self.p.set_parameter([self.collection, self.name, self.state])
            # HACK: avoid having to save parameters to registry through server
            # why do we even use the registry??
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, self.state))

    def update_value(self, val):
        if self.state[0] == val[0]:
            return
        idx = self.findText(self.d_selections[val])
        self.setCurrentIndex(idx)
        self.state = val, self.state[1]


class ParameterSelectionEditor(QtWidgets.QDoubleSpinBox, BaseEditor):
    editor = "parameter"
    def __init__(self, *params):
        BaseEditor.__init__(self, *params)
        QtWidgets.QDoubleSpinBox.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setDecimals(5)
        self.setMinimum(-1e20)
        self.setMaximum(1e20)
        if not (type(self.state) == u.ValueArray or
                type(self.state) == u.DimensionlessArray):
            # Improper format
            return
        self.min_, self.max_, val = self.state
        self.units = ""
        try:
            self.units = val.units
            self.setValue(val[val.units])
            txt = "  " + val.units
            txt = txt.replace("u", "\u00B5")
            self.setSuffix(txt)
        except AttributeError:
            # dimensionless
            self.state[2] = U(self.state[2], "")
            self.setValue(val)
        self.setButtonSymbols(2)
        self.setSingleStep(.1)

        self.setKeyboardTracking(False)
        self.valueChanged.connect(self.on_param_changed_locally)

    def check_bounds(self, val):
        if self.min_ <= U(val, self.units) <= self.max_:
            return True
        else:
            return False

    @inlineCallbacks
    def on_param_changed_locally(self, val):
        U_val = U(val, self.units)
        if self.check_connection(self.acxn) and self.check_bounds(val):
            self.state[-1] = U_val
            yield self.p.set_parameter([self.collection, self.name, U_val])
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, self.state))
        else:
            try:
                self.setValue(self.state[-1][self.units])
            except IndexError:
                self.setValue(self.state[-1])

    def update_value(self, val):
        changed = self.state[-1] == val
        if changed:
            return
        try:
            self.setValue(val[self.units])
            self.state[-1] = val
        except:
            try:
                self.setValue(val)
                self.state[-1] = val
            except TypeError:
                # not sure what's happening here
                self.units = val.units
                self.setValue(val[self.units])
                self.state[-1] = val[self.units]

    def focusOutEvent(self, event):
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        super(ParameterSelectionEditor, self).focusOutEvent(event)


class IntListEditor(BaseEditor):
    editor = "int_list"
    refreshsignal = QtCore.pyqtSignal(list)
    def __init__(self, *params):
        BaseEditor.__init__(self, *params)
        QtWidgets.QWidget.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        try:
            assert type(self.state) == np.ndarray
            for i in self.state:
                int(i)
        except (AssertionError, ValueError):
            logger.info("Improper format for int_list "
                        "registry entry: {}".format(self.state))
            return
        self.layout = QtWidgets.QHBoxLayout()
        self.widgets = []
        for i in range(len(self.state)):
            widget = QtWidgets.QSpinBox()
            widget.setButtonSymbols(2)
            widget.setObjectName(str(i))
            widget.setValue(int(self.state[i]))
            widget.valueChanged.connect(self.on_param_changed_locally)
            self.layout.addWidget(widget)
            self.widgets.append(widget)
        self.setLayout(self.layout)
        self.refreshsignal.connect(self.refresh_widgets)

    def check_bounds(self, val):
        blist = [val[i] < val[i+1] for i in range(len(val) - 1)]
        if False in blist:
            return False
        else:
            return True

    @inlineCallbacks
    def on_param_changed_locally(self, newval):
        try:
            sender = self.sender()
            obj_idx = int(sender.objectName())
        except (AttributeError, ValueError):
            obj_idx = 0
        val = [int(widget.value()) for widget in self.widgets]
        if self.check_connection(self.acxn) and self.check_bounds(val):
            yield self.p.set_parameter([self.collection, self.name, val])
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, val))
            self.state[obj_idx] = newval
        else:
            sender.setValue(self.state[obj_idx])

    @inlineCallbacks
    def update_value(self, val):
        if list(self.state) == list(val):
            return
        lstate = len(self.state)
        lval = len(val)
        diff = lval - lstate
        if diff == 0:
            for i, widget in enumerate(self.widgets):
                widget.setValue(int(val[i]))
        else:
            self.refreshsignal.emit(list(val))
        self.state = val
        if self.r is not None:
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, val))
        else:
            self.r = yield self.acxn.get_server("registry")
            yield self.r.cd("", "Servers", "Parameter Vault", self.collection)
            yield self.r.set(self.name, (self.editor, val))

    def refresh_widgets(self, val):
        for widget in self.widgets:
            self.layout.removeWidget(widget)
            widget.deleteLater()
            widget = None
        del self.widgets
        self.widgets = []
        for i in range(len(val)):
            try:
                widget = QtWidgets.QSpinBox()
                widget.setButtonSymbols(2)
                widget.setObjectName(str(i))
                widget.setValue(int(val[i]))
                widget.valueChanged.connect(self.on_param_changed_locally)
                self.layout.addWidget(widget)
                self.widgets.append(widget)
            except:
                logger.warning("Couldn't refresh int_list widget", exc_info=True)


BoolEditor.register()
SelectionSimpleEditor.register()
LineSelectionEditor.register()
ParameterSelectionEditor.register()
IntListEditor.register()
