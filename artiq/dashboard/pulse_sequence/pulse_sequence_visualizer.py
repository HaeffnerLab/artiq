from artiq.dashboard.pulse_sequence.sequence_analyzer import SequenceAnalyzer
from artiq.protocols.pc_rpc import Server
import asyncio
import logging
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets, QtGui
import time

logger = logging.getLogger(__name__)
simulation_logger = logging.getLogger("** SIMULATION **")
        
class PulseSequenceVisualizerServer:
    def __init__(self, psv):
        self.psv = psv

    def plot_simulated_pulses(self, dds, ttl, channels):
        try:
            self.psv.on_new_seq(dds, ttl, channels, signal_time=time.localtime())
        except:
            logger.warning("Failed to plot pulse sequence visualization", exc_info=True)
            raise

class PulseSequenceVisualizer(QtWidgets.QDockWidget):
    def __init__(self):
        QtWidgets.QDockWidget.__init__(self, "Pulse Sequence")
        self.setObjectName("PulseSequenceDock")
        self.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        # Initialize
        self.last_seq_data = None
        self.last_plot = None
        self.subscribed = False
        self.current_box = None
        self.mpl_connection = None
        self.main_widget = QtWidgets.QWidget()
        self.setWidget(self.main_widget)
        self.create_layout()
        self.connect_asyncio_server()

    def connect_asyncio_server(self):
        self.loop = asyncio.get_event_loop()
        self.asyncio_server = Server({
            "pulse_sequence_visualizer": PulseSequenceVisualizerServer(self),
            "simulation_logger": simulation_logger
            }, None, True)
        self.task = self.loop.create_task(self.asyncio_server.start("::1", 3289))
    
    def create_layout(self):
        # Creates GUI layout
        layout = QtGui.QVBoxLayout()
        plot_layout = self.create_plot_layout()
        layout.addLayout(plot_layout)
        self.main_widget.setLayout(layout)
   
    def create_plot_layout(self):
        # Creates empty matplotlib plot layout
        layout = QtGui.QVBoxLayout()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        self.axes = self.fig.add_subplot(111)
        self.axes.legend(loc = 'best')
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        self.axes.set_title('Most Recent Pulse Sequence', fontsize = 22)
        self.axes.set_xlabel('Time (ms)')
        self.fig.tight_layout()
        # Create an empty an invisible annotation, which will be moved around and set to visible later when needed
        self.annot = self.axes.annotate("", xy=(0,0), xytext=(-0.5,0.5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"), horizontalalignment='center', 
            multialignment='left', 
            verticalalignment='center')
        self.annot.get_bbox_patch().set_alpha(0.8)
        self.annot.set_visible(False)
        # Add the canvas to the GUI widget.
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas)
        return layout

    def on_new_seq(self, dds, ttl, channels, signal_time):
        # Temporary stop tracking mouse movement
        if self.mpl_connection:
            self.canvas.mpl_disconnect(self.mpl_connection)
        self.last_seq_data = {'DDS':dds, 'TTL':ttl, 'channels':channels}
        # Create SequenceAnalyzer object instance
        self.sequence = SequenceAnalyzer(ttl, dds, channels)
        # Clear the plot of all drawn objects
        self.clear_plot()
        # Call the SequenceAnalyzer object's create_full_plot method to draw the plot on the GUI's axes.
        self.sequence.create_full_plot(self.axes)
        self.axes.set_title('Most Recent Pulse Sequence, ' + time.strftime('%Y-%m-%d %H:%M:%S', signal_time))
        # Draw and reconnect to mouse hover events
        self.canvas.draw_idle()
        self.mpl_connection = self.canvas.mpl_connect("motion_notify_event", self.hover)

    def clear_plot(self):
        # Remove all lines, boxes, and annotations, except for the hover annotation
        for child in self.axes.get_children():
            if isinstance(child, (matplotlib.lines.Line2D, matplotlib.text.Annotation, matplotlib.collections.PolyCollection)):
                if child is not self.annot:
                    child.remove()

    def format_starttime(self, t):
        # Function for formatting times in the hover annotation
        if round(1e6*t) < 1000:
            return '{:.1f} $\mu$s'.format(1e6*t)
        else:
            return '{:.3f} ms'.format(1e3*t)

    def format_duration(self, t):
        # Function for formatting times in the hover annotation
        if round(1e6*t) < 1000:
            return '%#.4g $\mu$s' % (1e6*t)
        else:
            return '%#.4g ms' % (1e3*t)

    def update_annot(self, dds_box):
        # This function updates the text of the hover annotation.
        drawx = 1e3*(dds_box.starttime() + dds_box.duration()/2.0)
        drawy = dds_box.offset + dds_box.scale/2.0
        self.annot.xy = (drawx, drawy)
        text = '{0}\nStart: {1}\nDuration: {2}\n{3:.4f} MHz\n{4:.2f} amp w/att'.format(dds_box.channel,
                                                                                self.format_starttime(dds_box.starttime()),
                                                                                self.format_duration(dds_box.duration()),
                                                                                dds_box.frequency(),
                                                                                dds_box.amplitude())
        self.annot.set_text(text)

    def hover(self, event):
        # This function is called when the mouse moves
        # It updates the hover annotation if necessary.
        (self.last_mouse_x, self.last_mouse_y) = (event.x, event.y)
        vis = self.annot.get_visible()
        if event.inaxes == self.axes:
            for dds_box in self.sequence.dds_boxes:
                if dds_box.box.contains(event)[0]:
                    if dds_box is not self.current_box:
                        self.current_box = dds_box
                        self.update_annot(dds_box)
                        self.annot.set_visible(True)
                        self.canvas.draw_idle()
                    break
                else:
                    self.current_box = None
                    if vis:
                        self.annot.set_visible(False)
                        self.canvas.draw_idle()
        else:
            self.current_box = None

    def closeEvent(self, event):
        self.loop.create_task(self.asyncio_server.stop())
        super(PulseSequenceVisualizer, self).closeEvent(event)
    
if __name__=="__main__":
    a = QtGui.QApplication( [] )
    widget = pulse_sequence_visualizer()
    widget.show()