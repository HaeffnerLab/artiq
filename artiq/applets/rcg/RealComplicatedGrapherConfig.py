import pickle
from collections import namedtuple
from artiq.applets.rcg.fitting.fit_functions import __all__ as fit_models
import os


# Host / port to run server on
host = "::1"
port = 3286


# Data directory location
data_dir = os.path.join(os.path.expanduser("~"), "data")


# Tuple. First value True, then load all compatible hdf5 data  presently saved in
# the ~/data directory under today's date. Second value True, then all items
# are loaded in their 'checked' state.
auto_load = True, True


# Options to pass to pyqtgraph.setConfigOptions()
opts = {"foreground": "w"}


# Plot Line Colors
default_colors = [(47,126,243), (250,138,39), (96,233,128), (255,77,77),
                  (255,51,153), (128,255,0), (255,241,102), (255,128,128),
                  (255,255,192), (255,255,64), (0,255,0), (64, 255,255),
                  (0,128, 255), (192,64,192), (255,255,255), (64, 0, 255),
                  (128,128,128)]


# Custom Line colors. Ideally prodcued by QColorDialog in RealComplicatedGraphy.py
try:
    custom_colors = pickle.load(open("custom_colors.pkl", "rb"))
except:
    custom_colors = default_colors


# Plotter tabs configuration
graphConfig = namedtuple("graphConfig",
                  "title name row col rowspan colspan show_points ylims")
graphConfig.__new__.__defaults__ = ("", "Current", 0, 0, 1, 1, True, [0, 1])

tab_configs = [
    ("Current", [graphConfig(name="Current", ylims=[-2, 2])]),
    ("Spectrum", [graphConfig(name="Spectrum")]),
    ("CalibLines", [graphConfig(name="CalibLine1", title="Line1"),
                    graphConfig(name="CalibLine2", title="Line2", col=1)]),
    ("Rabi", [graphConfig(name="Rabi", ylims=[0, 1])]),
    ("Molmer-Sorensen", [graphConfig(name="Molmer-Sorensen")]),
    ("VAET", [graphConfig(name="vaet_parity", title="VAET Parity", ylims=[-1, 1]),
             graphConfig(name="scan_nu_eff", title="Scan Nu_eff", col=1),
              graphConfig(name="vaet_time", title="VAET Time", row=1, colspan=2, ylims=[0, 2000])]),
    ("Parity", [graphConfig(name="Parity", ylims=[-1, 1])]),
    ("Ramsey", [graphConfig(name="Ramsey", ylims=[0, 1])]),
    ("CalibSidebands", [graphConfig(name="nbar", title="nbar", colspan=2),
                        graphConfig(name="CalibRed", title="CalibRed", row=1),
                        graphConfig(name="CalibBlue", title="CalibBlue", row=1, col=1)]),
    ("DriftTrackerRamsey", [graphConfig(name="DriftTrackerRamsey1", title="DriftTrackerRamsey1"),
                            graphConfig(name="DriftTrackerRamsey2", title="DriftTrackerRamsey2", col=1)]),
    ("Benchmarking", [graphConfig(name="Benchmarking", ylims=[0, 1])]),
]


# Declare Fit Models
fit_models_dir = "/home/lattice/artiq/artiq/applets/rcg/fitting"
