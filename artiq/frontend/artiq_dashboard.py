#!/usr/bin/env python3)
import argparse
import asyncio
import atexit
import os
import logging

from PyQt5 import QtCore, QtGui, QtWidgets
from quamash import QEventLoop

from artiq import __artiq_dir__ as artiq_dir, __version__ as artiq_version
from artiq.tools import (atexit_register_coroutine, add_common_args,
                         get_user_config_dir)
from artiq.protocols.pc_rpc import AsyncioClient, Client
from artiq.protocols.broadcast import Receiver
from artiq.gui.models import ModelSubscriber
from artiq.gui import state, log
from artiq.dashboard import (experiments, shortcuts, explorer,
                             moninj, datasets, schedule, applets_ccb,
                             pmt_control, parameter_editor)
from artiq.dashboard.laser_room.laser_room_tab import LaserRoomTab
#from artiq.dashboard.temp_controller.temp_controller_tab import TempControllerTab
from artiq.dashboard.drift_tracker.drift_tracker import DriftTracker
from artiq.dashboard.readout_histograms.readout_histograms import ReadoutHistograms
import labrad
from lattice.clients.connection import connection
from twisted.internet.defer import inlineCallbacks


needs_parameter_vault = list()
laser_room_ip_address = "192.168.169.49"
lase_room_password = "lab"


def get_argparser():
    parser = argparse.ArgumentParser(description="ARTIQ Dashboard")
    parser.add_argument(
        "-s", "--server", default="::1",
        help="hostname or IP of the master to connect to")
    parser.add_argument(
        "--port-notify", default=3250, type=int,
        help="TCP port to connect to for notifications")
    parser.add_argument(
        "--port-control", default=3251, type=int,
        help="TCP port to connect to for control")
    parser.add_argument(
        "--port-broadcast", default=1067, type=int,
        help="TCP port to connect to for broadcasts")
    parser.add_argument(
        "--db-file", default=None,
        help="database file for local GUI settings")
    add_common_args(parser)
    return parser


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self):
        QtWidgets.QTabWidget.__init__(self)
        self.setFocusPolicy(0)
        self.exit_request = asyncio.Event()
        self.setObjectName("MainTabs")

    def closeEvent(self, event):
        event.ignore()
        self.exit_request.set()

    def save_state(self):
        return {"tab_index": int(self.currentIndex())}

    def restore_state(self, state):
        self.setCurrentIndex(int(state["tab_index"]))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, server):
        QtWidgets.QMainWindow.__init__(self)

        icon = QtGui.QIcon(os.path.join(artiq_dir, "gui", "lettice.svg"))
        self.setWindowIcon(icon)
        self.setWindowTitle("ARTIQ Dashboard - {}".format(server))

        qfm = QtGui.QFontMetrics(self.font())
        self.resize(140*qfm.averageCharWidth(), 38*qfm.lineSpacing())

        self.exit_request = asyncio.Event()

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

class MdiArea(QtWidgets.QMdiArea):
    def __init__(self):
        QtWidgets.QMdiArea.__init__(self)
        self.pixmap = QtGui.QPixmap(os.path.join(
            artiq_dir, "gui", "logo_ver.svg"))

    def paintEvent(self, event):
        QtWidgets.QMdiArea.paintEvent(self, event)
        painter = QtGui.QPainter(self.viewport())
        x = (self.width() - self.pixmap.width())//2
        y = (self.height() - self.pixmap.height())//2
        painter.setOpacity(1)
        painter.drawPixmap(x, y, self.pixmap)

@inlineCallbacks
def parameter_vault_connect(*args):
    for widget in needs_parameter_vault:
        yield widget.setup_listeners()
        widget.setDisabled(False)

def parameter_vault_disconnect(*args):
    for widget in needs_parameter_vault:
        widget.setDisabled(True)

def main():
    # connect to labrad
    acxn = connection()
    acxn.connect(host='localhost', password='lab')
    acxn.add_on_connect("ParameterVault", parameter_vault_connect)
    acxn.add_on_disconnect("ParameterVault", parameter_vault_disconnect)
    # connect to laser room labrad
    laser_room_acxn = connection()
    laser_room_acxn.connect(host=laser_room_ip_address,
                                 password=lase_room_password,
                                 tls_mode="off")
    # initialize application
    args = get_argparser().parse_args()
    widget_log_handler = log.init_log(args, "dashboard")

    if args.db_file is None:
        args.db_file = os.path.join(get_user_config_dir(),
                           "artiq_dashboard_{server}_{port}.pyon".format(
                            server=args.server.replace(":","."),
                            port=args.port_notify))

    app = QtWidgets.QApplication(["ARTIQ Dashboard"])
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    atexit.register(loop.close)
    smgr = state.StateManager(args.db_file)

    # create connections to master
    rpc_clients = dict()
    for target in "schedule", "experiment_db", "dataset_db":
        client = AsyncioClient()
        loop.run_until_complete(client.connect_rpc(
            args.server, args.port_control, "master_" + target))
        atexit.register(client.close_rpc)
        rpc_clients[target] = client

    config = Client(args.server, args.port_control, "master_config")
    try:
        server_name = config.get_name()
    finally:
        config.close_rpc()

    disconnect_reported = False
    def report_disconnect():
        nonlocal disconnect_reported
        if not disconnect_reported:
            logging.error("connection to master lost, "
                          "restart dashboard to reconnect")
        disconnect_reported = True

    sub_clients = dict()
    for notifier_name, modelf in (("explist", explorer.Model),
                                  ("explist_status", explorer.StatusUpdater),
                                  ("datasets", datasets.Model),
                                  ("schedule", schedule.Model)):
        subscriber = ModelSubscriber(notifier_name, modelf,
            report_disconnect)
        loop.run_until_complete(subscriber.connect(
            args.server, args.port_notify))
        atexit_register_coroutine(subscriber.close)
        sub_clients[notifier_name] = subscriber

    broadcast_clients = dict()
    for target in "log", "ccb":
        client = Receiver(target, [], report_disconnect)
        loop.run_until_complete(client.connect(
            args.server, args.port_broadcast))
        atexit_register_coroutine(client.close)
        broadcast_clients[target] = client

    # initialize main window
    tabs = TabWidget()
    main_main_window = MainWindow(args.server if server_name is None else server_name)
    main_window = MainWindow(args.server if server_name is None else server_name)
    main_main_window.setCentralWidget(tabs)
    smgr.register(tabs)
    smgr.register(main_main_window)
    smgr.register(main_window, "sortoflikeamainwindowbutnotquite")
    mdi_area = MdiArea()
    mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    main_window.setCentralWidget(mdi_area)

    # create UI components
    expmgr = experiments.ExperimentManager(main_window,
                                           sub_clients["explist"],
                                           sub_clients["schedule"],
                                           rpc_clients["schedule"],
                                           rpc_clients["experiment_db"],
                                           )
    smgr.register(expmgr)
    # d_shortcuts = shortcuts.ShortcutsDock(main_window, expmgr)
    # smgr.register(d_shortcuts)
    d_pmt = pmt_control.PMTControlDock(acxn)
    smgr.register(d_pmt)
    d_parameter_editor = parameter_editor.ParameterEditorDock(acxn=acxn)
    smgr.register(d_parameter_editor)
    needs_parameter_vault.append(d_parameter_editor)
    d_explorer = explorer.ExplorerDock(expmgr, None,
                                       sub_clients["explist"],
                                       sub_clients["explist_status"],
                                       rpc_clients["schedule"],
                                       rpc_clients["experiment_db"])
    smgr.register(d_explorer)

    d_datasets = datasets.DatasetsDock(sub_clients["datasets"],
                                       rpc_clients["dataset_db"])
    smgr.register(d_datasets)

    d_applets = applets_ccb.AppletsCCBDock(main_window, sub_clients["datasets"])
    atexit_register_coroutine(d_applets.stop)
    smgr.register(d_applets)
    broadcast_clients["ccb"].notify_cbs.append(d_applets.ccb_notify)

    d_ttl_dds = moninj.MonInj()
    loop.run_until_complete(d_ttl_dds.start(args.server, args.port_notify))
    atexit_register_coroutine(d_ttl_dds.stop)

    d_schedule = schedule.ScheduleDock(
        rpc_clients["schedule"], sub_clients["schedule"])
    smgr.register(d_schedule)

    logmgr = log.LogDockManager(main_window)
    smgr.register(logmgr)
    broadcast_clients["log"].notify_cbs.append(logmgr.append_message)
    widget_log_handler.callback = logmgr.append_message

    # lay out docks
    right_docks = [
        d_explorer, d_pmt, d_parameter_editor,
        d_ttl_dds.ttl_dock, #d_ttl_dds.dds_dock,
        d_ttl_dds.dac_dock,
        d_datasets, d_applets
    ]
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, right_docks[0])
    for d1, d2 in zip(right_docks, right_docks[1:]):
        main_window.tabifyDockWidget(d1, d2)
    main_window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d_schedule)

    tabs.addTab(main_window, "Control")
    laser_room_tab =  LaserRoomTab()
    smgr.register(laser_room_tab)
    tabs.addTab(laser_room_tab, "Laser Room")
    # temp_controller_tab = TempControllerTab()
    # smgr.register(temp_controller_tab)
    # tabs.addTab(temp_controller_tab, "Temperature Controller")
    histograms_tab = ReadoutHistograms(acxn, smgr)
    smgr.register(histograms_tab)
    needs_parameter_vault.append(histograms_tab)
    tabs.addTab(histograms_tab, "Readout")
    # TEMP: disable drift tracker tab to avoid noisy logging
    #drift_tracker_tab = DriftTracker(laser_room_acxn)
    #smgr.register(drift_tracker_tab)
    #tabs.addTab(drift_tracker_tab, "Drift Tracker")

    smgr.load()
    smgr.start()
    atexit_register_coroutine(smgr.stop)

    # load/initialize state
    if os.name == "nt":
        # HACK: show the main window before creating applets.
        # Otherwise, the windows of those applets that are in detached
        # QDockWidgets fail to be embedded.
        main_window.show()

    # work around for https://github.com/m-labs/artiq/issues/1307
    d_ttl_dds.ttl_dock.show()
    d_ttl_dds.dds_dock.show()

    # create first log dock if not already in state
    d_log0 = logmgr.first_log_dock()
    if d_log0 is not None:
        main_window.tabifyDockWidget(d_schedule, d_log0)


    if server_name is not None:
        server_description = server_name + " ({})".format(args.server)
    else:
        server_description = args.server
    logging.info("ARTIQ dashboard %s connected to %s",
                 artiq_version, server_description)

    main_main_window.show()
    loop.run_until_complete(main_main_window.exit_request.wait())

if __name__ == "__main__":
    main()
