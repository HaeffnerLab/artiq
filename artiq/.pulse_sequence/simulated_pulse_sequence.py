from artiq.dashboard.drift_tracker import client_config as dt_config
from artiq.language import core as core_language
from artiq.protocols.pc_rpc import Client
from datetime import datetime
from easydict import EasyDict as edict
import importlib
import importlib.machinery
import json
import labrad
from labrad.units import WithUnit
import logging
import numpy as np
import os
import traceback
import sys

logger = logging.getLogger(__name__)

#
# Entry point to trigger a simulation of a particular experiment
#
def run_simulation(file_path, class_, argument_values):

    # define a function to import a modified source file
    def modify_and_import(module_name, path, modification_func):
        # adapted from https://stackoverflow.com/questions/41858147/how-to-modify-imported-source-code-on-the-fly
        loader = importlib.machinery.SourceFileLoader(module_name, path)
        source = loader.get_source(module_name)
        new_source = modification_func(source)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        codeobj = compile(new_source, module.__spec__.origin, 'exec')
        exec(codeobj, module.__dict__)
        sys.modules[module_name] = module
        return module

    # import all of the subsequences and strip out the @kernel decorators
    subsequences_folder = os.path.join(os.path.expanduser("~"), "artiq-work", "subsequences")
    for path, subdirs, files in os.walk(subsequences_folder):
        for filename in files:
            filename_without_extension, extension = os.path.splitext(filename)
            if extension == ".py":
                try:
                    module_name = "simulated_subsequences." + filename_without_extension
                    experiment_file_full_path = os.path.join(path, filename)
                    modify_and_import(module_name, experiment_file_full_path, lambda src:
                        src.replace("@kernel", ""))
                except:
                    logger.error(traceback.print_exc())
                    continue

    # load the experiment source and make the necessary modifications
    file_path = os.path.join(os.path.expanduser("~"), "artiq-work", file_path)
    mod = modify_and_import(class_, file_path, lambda src: 
        src.replace("from pulse_sequence", "from simulated_pulse_sequence")
        .replace("from subsequences.", "from simulated_subsequences.")
        .replace("@kernel", ""))

    # execute the simulated pulse sequence
    pulse_sequence = getattr(mod, class_)()
    pulse_sequence.set_submission_arguments(argument_values)
    pulse_sequence.simulate()

class SimulatedDDSSwitch:
    def __init__(self, dds):
        self.dds = dds
        self.is_on = False
    def on(self):
        if not self.is_on:
            self.is_on = True
            self.dds._switched_on()
    def off(self):
        if self.is_on:
            self.is_on = False
            self.dds._switched_off()
    def toggle(self):
        if self.is_on:
            self.off()
        else:
            self.on()

class SimulatedDDS:

    def __init__(self, name, pulse_sequence):
        self.name = name
        self.pulse_sequence = pulse_sequence
        self.sw = SimulatedDDSSwitch(self)
        self.freq = 0.0
        self.amplitude = 0.0
        self.phase = 0.0
        self.ref_time_mu = 0
        self.att = 8.0

    def _switched_on(self):
        self.time_switched_on = self.pulse_sequence.time_manager.get_time()

    def _switched_off(self):
        if self.time_switched_on is not None:
            time_switched_off = self.pulse_sequence.time_manager.get_time()
            self.pulse_sequence.report_pulse(self, self.time_switched_on, time_switched_off)
            self.time_switched_on = None

    def set(self, freq, amplitude=None, phase=None, ref_time_mu=None):
        if isinstance(freq, WithUnit):
            freq = freq['Hz']

        self.freq = freq
        if amplitude:
            self.amplitude = amplitude
        if phase:
            self.phase = phase
        if ref_time_mu:
            self.ref_time_mu = ref_time_mu

    def set_att(self, att):
        self.att = att

class _FakeCore:
    def seconds_to_mu(self, time):
        return time

class SimulationScheduler:
    def submit(self, scheduler_name, expid, priority=None):
        run_simulation(expid["file"], expid["class_name"], expid["arguments"])

class FitError(Exception):
    pass

class PulseSequence:

    scan_params = dict()

    def __init__(self):
        self.p = None
        self.set_subsequence = dict()
        self.run_after = dict()
        self.selected_scan = dict()
        self.time_manager = None
        self.parameter_dict = None
        self.simulated_pulses = None
        self.core = _FakeCore()
        self.data = edict()
        self.grapher = None
        self.scheduler = SimulationScheduler()
        self.rcg_tabs = dict()

        self.sequence_name = type(self).__name__
        self.rcg_tabs[self.sequence_name] = dict()
        self.timestamp = datetime.now().strftime("%H%M_%S")
        self.dir = os.path.join(os.path.expanduser("~"), "data", "simulation",
                                datetime.now().strftime("%Y-%m-%d"), self.sequence_name)
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)

        logger.info("Simulation timestamp " + self.timestamp + ", output files saved to " + self.dir)

    def set_submission_arguments(self, submission_arguments):
        self.submission_arguments = submission_arguments
    
    def write_line(self, output_file, line):
        # writes a line to both the file and the logger (if uncommented)
        output_file.write(line + "\n")
        #logger.info(line)

    def load_parameters(self):
        if not self.p:
            self.p = self.load_parameter_vault()

        self.parameter_dict = {}

        # add accessed params to parameter_dict
        for collection_name in self.p:
            for param_name in self.p[collection_name]:
                self.parameter_dict[collection_name + "." + param_name] = self.p[collection_name][param_name]

        # update trap frequencies in member variables
        self.trap_frequency_names = list()
        self.trap_frequency_values = list()
        for name, value in self.p.TrapFrequencies.items():
            self.trap_frequency_names.append(name)
            self.trap_frequency_values.append(value)
    
    def write_parameters_for_scan(self, scan_name):
        if not self.p:
            self.load_parameters()

        # add scan params to parameter_dict
        self.parameter_dict["Scan.sequence_name"] = self.sequence_name
        self.parameter_dict["Scan.scan_name"] = scan_name
        self.parameter_dict["Scan.parameter_name"] = self.scan_parameter_name
        for k,v in self.scan_settings.items():
            self.parameter_dict["Scan." + k] = v

        # write all the parameters to a file
        filename = self.timestamp + "_params_" + scan_name + ".txt"
        with open(filename, "w") as param_file:
            self.write_line(param_file, str(self.parameter_dict))
        print("** SIMULATION ** parameters written to " + os.path.join(self.dir, filename))

    def report_pulse(self, dds, time_switched_on, time_switched_off):
        simulated_pulse = {
            "dds_name": dds.name,
            "time_on": time_switched_on,
            "time_off": time_switched_off,
            "freq": dds.freq,
            "amp": dds.amplitude,
            "att": dds.att,
            "phase": dds.phase,
        }
        self.simulated_pulses.append(simulated_pulse)

    def simulate_with_ion_sim(self):
        path_to_simulate_jl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulate.jl")
        from julia import Main
        Main.include(path_to_simulate_jl)
        return Main.simulate_with_ion_sim(self.parameter_dict, self.simulated_pulses, self.num_ions)

    def simulate(self):
        self.load_parameters()
        self.setup_grapher()
        self.setup_carriers()
        self.setup_dds()
        
        self.N = int(self.p.StateReadout.repeat_each_measurement)

        self.num_ions = int(self.p.IonsOnCamera.ion_number)
        
        for scan_name in PulseSequence.scan_params:
            self.data[scan_name] = edict(x=[], y=[])
            x_data = np.array([], dtype=float)
            y_data = {}
            
            # Look up the settings for the current scan.
            self.scan_parameter_name = self.submission_arguments[scan_name + "-Scan_Selection"]
            self.scan_settings = self.submission_arguments[scan_name + ":" + self.scan_parameter_name]
            self.selected_scan[scan_name] = self.scan_parameter_name
            
            # Output parameters to be used for the current scan.
            self.write_parameters_for_scan(scan_name)

            # Call run_initially, but only if this is the first scan.
            if scan_name == list(PulseSequence.scan_params.keys())[0]:
                self.run_initially()
            
            # Determine the list of scan points.
            variable_param_name = self.scan_parameter_name.replace(".", "_")
            scan_points = [self.get_variable_parameter(variable_param_name)]
            if self.scan_settings["ty"] == "RangeScan":
                scan_points = np.linspace(self.scan_settings["start"], self.scan_settings["stop"], self.scan_settings["npoints"])

            # Iterate over the scan points and simulate one pulse sequence per point.
            for scan_idx, scan_point in enumerate(scan_points):
                # Reset the timer and stored pulse sequence.
                self.setup_time_manager()
                self.simulated_pulses = []

                # Overwrite the scan parameter value with the current scan point.
                setattr(self, variable_param_name, scan_point)
                self.parameter_dict[self.scan_parameter_name] = scan_point

                self.set_subsequence[scan_name]()

                # Run the pulse sequence function to generate the pulse sequence.
                current_sequence = getattr(self, scan_name)
                current_sequence()

                # Write the generated pulse sequences to a file.
                filename = self.timestamp + "_pulses_" + str(scan_idx) + ".txt"
                with open(filename, "w") as pulses_file:
                    self.write_line(pulses_file, str(self.simulated_pulses))
                print("** SIMULATION ** pulse sequence written to " + os.path.join(self.dir, filename))
                
                # Call IonSim code to simulate the dynamics.
                print("** SIMULATION ** Calling IonSim with num_ions=" + str(self.num_ions) + ", " + self.scan_parameter_name + "=" + str(scan_point))
                result_data = self.simulate_with_ion_sim()
                
                # Record and plot the result.
                x_data = np.append(x_data, scan_point)
                for result_name, result_value in result_data.items():
                    if not result_name in y_data:
                        y_data[result_name] = np.array([], dtype=float)
                    y_data[result_name] = np.append(y_data[result_name], result_value)
                    if self.grapher:
                        plot_title = self.timestamp + " - " + scan_name + " - state:" + result_name
                        self.grapher.plot(x_data, y_data[result_name], tab_name="IonSim",
                            plot_title=plot_title, append=True,
                            file_="", range_guess=(scan_points[0], scan_points[-1]))
        
            # Add the results to self.data and output to file.
            self.data[scan_name]["x"] = x_data
            for y_name, y_data in y_data.items():
                self.data[scan_name]["y"].append(y_data)
            filename = self.timestamp + "_results.txt"
            with open(filename, "w") as results_file:
                self.write_line(results_file, str(self.data[scan_name]))
            print("** SIMULATION ** results written to " + os.path.join(self.dir, filename))
            
            if scan_name in self.run_after:
                try:
                    self.run_after[scan_name]()
                except FitError:
                    logger.error("FitError encountered in run_after for " + scan_name, exc_info=True)
                    raise

        try:
            self.run_finally()
        except FitError:
            logger.error("FitError encountered in run_finally", exc_info=True)
            raise

    def setup_grapher(self):
        if not self.grapher:
            try:
                self.grapher = Client("::1", 3286, "rcg")
            except:
                logger.warning("Failed to connect to RCG grapher", exc_info=True)
                self.grapher = None

    def setup_carriers(self):
        self.carrier_names = [
            "S+1/2D-3/2", "S-1/2D-5/2", "S+1/2D-1/2", "S-1/2D-3/2", "S+1/2D+1/2",
            "S-1/2D-1/2", "S+1/2D+3/2", "S-1/2D+1/2", "S+1/2D+5/2", "S-1/2D+3/2"]
        self.carrier_dict = {}
        for idx, name in enumerate(self.carrier_names):
            self.carrier_dict[name] = idx
        self.carrier_values = self.get_carriers_from_sd_tracker()

    def get_carriers_from_sd_tracker(self):
        global_cxn = labrad.connect(dt_config.global_address,
                                    password=dt_config.global_password,
                                    tls_mode="off")
        sd_tracker = global_cxn.sd_tracker_global
        current_lines = sd_tracker.get_current_lines(dt_config.client_name)
        _list = [0.] * 10

        for carrier, frequency in current_lines:
            frequency = frequency.inBaseUnits()
            abs_freq = frequency[frequency.units]
            for i in range(10):
                if carrier == self.carrier_names[i]:
                    _list[i] = abs_freq
                    break
        return _list

    def get_trap_frequency(self, name):
        freq = 0.
        for i in range(len(self.trap_frequency_names)):
            if self.trap_frequency_names[i] == name:
                freq = self.trap_frequency_values[i]
                if isinstance(freq, WithUnit):
                    freq = freq['Hz']
                return freq
        return 0.

    def make_dds(self, name):
        return SimulatedDDS(name, self)

    def setup_dds(self):
        self.dds_729G = self.make_dds("729G")
        self.dds_729L1 = self.make_dds("729L1")
        self.dds_729L2 = self.make_dds("729L2")
        self.dds_SP_729G = self.make_dds("SP_729G")
        self.dds_SP_729L1 = self.make_dds("SP_729L1")
        self.dds_SP_729L2 = self.make_dds("SP_729L2")
        self.dds_SP_729G_bichro = self.make_dds("SP_729G_bichro")
        self.dds_SP_729L1_bichro = self.make_dds("SP_729L1_bichro")
        self.dds_SP_729L2_bichro = self.make_dds("SP_729L2_bichro")
        self.dds_397 = self.make_dds("397")
        self.dds_854 = self.make_dds("854")
        self.dds_866 = self.make_dds("866")

    def setup_time_manager(self):
        class _FakeTimeManager:
            def __init__(self):
                self.time = 0.

            def _noop(self, *args, **kwargs):
                pass

            def _take_time(self, duration):
                if isinstance(duration, WithUnit):
                    duration = duration['s']
                self.time += duration

            def _get_time(self):
                return self.time

            enter_sequential = _noop
            enter_parallel = _noop
            exit = _noop
            set_time_mu = _noop
            get_time_mu = _get_time
            get_time = _get_time
            take_time_mu = _take_time
            take_time = _take_time

        self.time_manager = _FakeTimeManager()
        core_language.set_time_manager(self.time_manager)
                
    def get_729_dds(self, name="729G", id=0):
        if id == 0:
            self.dds_729 =           self.dds_729G if name == "729G" else self.dds_729L1 if name == "729L1" else self.dds_729L2
            self.dds_729_SP =        self.dds_SP_729G if name == "729G" else self.dds_SP_729L1 if name == "729L1" else self.dds_SP_729L2
            self.dds_729_SP_bichro = self.dds_SP_729G_bichro if name == "729G" else self.dds_SP_729L1_bichro if name == "729L1" else self.dds_SP_729L2_bichro
        elif id == 1:
            self.dds_7291 =           self.dds_729G if name == "729G" else self.dds_729L1 if name == "729L1" else self.dds_729L2
            self.dds_729_SP1 =        self.dds_SP_729G if name == "729G" else self.dds_SP_729L1 if name == "729L1" else self.dds_SP_729L2
            self.dds_729_SP_bichro1 = self.dds_SP_729G_bichro if name == "729G" else self.dds_SP_729L1_bichro if name == "729L1" else self.dds_SP_729L2_bichro
        elif id == 2:
            self.dds_729 =           self.dds_729G
            self.dds_729_SP_line1 =        self.dds_SP_729G 
            self.dds_729_SP_line1_bichro = self.dds_SP_729G_bichro 
            self.dds_729_SP_line2 =        self.dds_SP_729L2 
            self.dds_729_SP_line2_bichro = self.dds_SP_729L2_bichro 

    def make_random_list(self, n, mean, std, min=None, max=None):
        #
        # Returns a list of n values pulled from a Gaussian distribution
        # with the given mean and standard deviation, in the range [min, max].
        #
        values = (std * np.random.randn(n) + mean).tolist()
        for i in range(len(values)):
            # make sure the values are between min and max
            if min:
                values[i] = max(min, amps[i])
            if max:
                values[i] = min(max, amps[i])
        return values

    def make_random_amplitudes(self, n, mean, std):
        #
        # Returns a list of n amplitudes pulled from a Gaussian distribution
        # with the given mean and standard deviation, in the range [0,1].
        #
        return self.make_random_list(n, mean, std, min=0.0, max=1.0)

    def make_random_frequencies(self, n, mean, std):
        #
        # Returns a list of n frequencies pulled from a Gaussian distribution
        # with the given mean and standard deviation, in the range [0,].
        #
        return self.make_random_list(n, mean, std, min=0.0)

    def generate_single_pass_noise_waveform(self, mean, std, freq_noise=False):
        pass
    
    def prepare_noisy_single_pass(self, freq_noise=False, id=0):
        pass

    def start_noisy_single_pass(self, phase_ref_time, freq_noise=False,
        freq_sp=WithUnit(80, 'MHz'), amp_sp=1.0, att_sp=8.0, phase_sp=0.,
        use_bichro=False, freq_sp_bichro=WithUnit(80, 'MHz'), amp_sp_bichro=1.0, att_sp_bichro=8.0, phase_sp_bichro=0.,
        id=0):
        # TODO: this doesn't add any noise right now.
        dds = self.dds_729_SP
        dds_bichro = self.dds_729_SP_bichro
        if id == 1:
            dds = self.dds_729_SP1
            dds_bichro = self.dds_729_SP_bichro1

        dds.set(freq_sp, amplitude=amp_sp, phase=phase_sp, ref_time_mu=phase_ref_time)
        dds.set_att(att_sp)
        dds.sw.on()
        if use_bichro:
            dds_bichro.set(freq_sp_bichro, amplitude=amp_sp_bichro, phase=phase_sp_bichro, ref_time_mu=phase_ref_time)
            dds_bichro.set_att(att_sp_bichro)
            dds_bichro.sw.on()

    def stop_noisy_single_pass(self, use_bichro=False, id=0):
        dds = self.dds_729_SP
        dds_bichro = self.dds_729_SP_bichro
        if id == 1:
            dds = self.dds_729_SP1
            dds_bichro = self.dds_729_SP_bichro1

        # Turn off the DDS outputs.
        dds.sw.off()
        if use_bichro:
            dds_bichro.sw.off()

    def prepare_pulse_with_amplitude_ramp(self, pulse_duration, ramp_duration, dds1_amp=0., use_dds2=False, dds2_amp=0.):
        self.pulse_duration = pulse_duration
        self.ramp_duration = ramp_duration
        self.dds1_amp = dds1_amp
        if use_dds2:
            self.dds2_amp = dds2_amp
        
    def execute_pulse_with_amplitude_ramp(self, dds1_att=8.0, dds1_freq=0.,
                                          use_dds2=False, dds2_att=8.0, dds2_freq=0.):
        # TODO: currently this doesn't actually do any ramping
        self.dds_729.set(dds1_freq, self.dds1_amp)
        self.dds_729.set_att(dds1_att)
        self.dds_729.sw.on()
        if use_dds2:
            self.dds_7291.set(dds2_freq, self.dds2_amp)
            self.dds_7291.set_att(dds2_att)
            self.dds_7291.sw.on()

        self.time_manager.take_time(self.pulse_duration)

        self.dds_729.sw.off()
        if use_dds2:
            self.dds_7291.sw.off()

    def add_subsequence(self, subsequence):
        self._set_subsequence_defaults(subsequence)
        subsequence.run = subsequence.subsequence
        try:
            subsequence.add_child_subsequences(self)
        except AttributeError:
            pass
        return subsequence

    def _set_subsequence_defaults(self, subsequence):
        d = subsequence.__dict__
        kwargs = dict()
        for key, value in d.items():
            if type(value) == str:
                try:
                    c, v = value.split(".")
                except AttributeError:
                    continue
                except ValueError:
                    continue
                try:
                    pv_value = self.p[c][v]
                except KeyError:
                    #TODO Ryan fix this - throw if a parameter isn't found
                    #raise Exception("Failed to find parameter: " + value)
                    continue
                try:
                    pv_value = float(pv_value)
                except:
                    pass
                kwargs[key] = pv_value
        for key, value in kwargs.items():
            setattr(subsequence, key, value)
            
    def get_offset_frequency(self, name):
        return 0.

    def calc_frequency(self, line, detuning=0.,
                    sideband="", order=0., dds="", bound_param=""):
        relative_display = self.Display_relative_frequencies
        freq = detuning
        abs_freq = 0.
        line_set = False
        sideband_set = True if sideband == "" else False
        for i in range(10):
            if line == self.carrier_names[i]:
                freq += self.carrier_values[i]
                line_set = True
            if sideband != "" and i <= len(self.trap_frequency_names) - 1:
                if sideband == self.trap_frequency_names[i]:
                    freq += self.trap_frequency_values[i] * order
                    sideband_set = True
            if line_set and sideband_set:
                abs_freq = freq
                break
        return freq

    def get_variable_parameter(self, name):
        # All params are fixed in simulation mode for now
        return getattr(self, name)

    def load_parameter_vault(self):
        # Grab parametervault params:
        cxn = labrad.connect()
        p = cxn.parametervault
        collections = p.get_collections()
        D = dict()
        for collection in collections:
            d = dict()
            names = p.get_parameter_names(collection)
            for name in names:
                try:
                    param = p.get_parameter([collection, name])
                    try:
                        if isinstance(param, WithUnit):
                            param = param.inBaseUnits()
                            param = param[param.units]
                    except KeyError:
                        if (units == "dBm" or
                            units == "deg" or
                            units == ""):
                            param = param[units]
                    d[name] = param
                    setattr(self, collection + "_" + name, param)
                except:
                    # broken parameter
                    continue
            D[collection] = d
        return edict(D)

    def run_initially(self):
        # individual pulse sequences should override this
        pass

    def sequence(self):
        # individual pulse sequences should override this
        raise NotImplementedError

    def run_finally(self):
        # individual pulse sequences should override this
        pass
