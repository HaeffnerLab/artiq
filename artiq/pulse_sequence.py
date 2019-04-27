from artiq.language import scan
from artiq.experiment import *
from easydict import EasyDict as edict
from datetime import datetime
import labrad
import numpy as np
import h5py as h5
import os
import csv
from artiq.protocols.pc_rpc import Client
from bisect import bisect


class PulseSequence(EnvExperiment):
    fixed_params = list()
    rcg_tab = "Current"
    scan_params = dict()
    show_params = list()

    def build(self):
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_device("pmt")
        self.setattr_device("LTriggerIN")

        if self.scan_params:
            self.multi_scannables = dict()
            for seq_name, scan_list in self.scan_params.items():
                self.multi_scannables[seq_name] = [self.get_argument(seq_name + ":" + scan_param[0],
                    scan.Scannable(default=scan.RangeScan(*scan_param[1:])), group=seq_name) for scan_param in scan_list]
        else:
            pass
            # print("\n"*10, self.scan_params)
            # raise Exception("No scan window specified.")

        self.setup()

        # Load all AD9910 and AD9912 DDS channels specified in device_db:
        for key, val in self.get_device_db().items():
            if isinstance(val, dict) and "class" in val:
                if val["class"] == "AD9910" or val["class"] == "AD9912":
                    setattr(self, "dds_" + key, self.get_device(key))
        self.cpld_list = [self.get_device("urukul{}_cpld".format(i)) for i in range(3)]

    def prepare(self):
        # Grab parametervault params:
        G = globals()
        cxn = labrad.connect()
        p = cxn.parametervault
        collections = p.get_collections()
        # Takes over 1 second to do this. We should move away from using labrad units
        # in registry. Would be nice if parametervault was not a labrad server.
        D = dict()
        for collection in collections:
            d = dict()
            names = p.get_parameter_names(collection)
            for name in names:
                try:
                    param = p.get_parameter([collection, name])
                    try:
                        units = param.units
                        if units == "":
                            param = param[units]
                        else:
                            param = param[units] * G[units]
                    except AttributeError:
                        pass
                    except KeyError:
                        if (units == "dBm" or
                            units == "deg" or
                            units == ""):
                            param = param[units]
                    d[name] = param
                except:
                    #broken parameter
                    continue
            D[collection] = d
        for item in self.fixed_params:
            collection, param = item[0].split(".")
            D[collection].update({param: item[1]})
        self.p = edict(D)
        cxn.disconnect()

        # Grab cw parameters:
        # Because parameters are grabbed in prepare stage, loaded dds cw parameters
        # may not be the most current.
        self.dds_list = list()
        self.freq_list = list()
        self.amp_list = list()
        self.att_list = list()
        self.state_list = list()

        for key, settings in self.p.dds_cw_parameters.items():
            self.dds_list.append(getattr(self, "dds_" + key))
            self.freq_list.append(float(settings[1][0]) * 1e6)
            self.amp_list.append(float(settings[1][1]))
            self.att_list.append(float(settings[1][3]))
            self.state_list.append(bool(float(settings[1][2])))
       
        # Try to make rcg/hist connection
        try:
            self.rcg = Client("::1", 3286, "rcg")
        except:
            self.rcg = None
        try:
            self.pmt_hist = Client("::1", 3287, "pmt_histogram")
        except:
            self.pmt_hist = None

        # Make scan object for repeating the experiment
        N = int(self.p.StateReadout.repeat_each_measurement)
        self.N = N

        # Create datasets and setup readout
        self.x_label = dict()
        scan_specs = dict()
        for seq_name, scan_list in self.multi_scannables.items():
            scan_specs[seq_name] = [len(scan) for scan in scan_list]

        self.rm = self.p.StateReadout.readout_mode
        
        if self.rm == "pmt":
            self.n_ions = len(self.p.StateReadout.threshold_list)
            for seq_name, dims in scan_specs.items():
                if len(dims) == 1:
                # Currently not supporting any default plotting for (n>1)-dim scans
                    for i in range(self.n_ions):
                        setattr(self, "{}-pmt_ion{}".format(seq_name, i), np.full(dims, np.nan))
                    # for i, scan in enumerate(self.multi_scannables[seq_name]):
                    x_array = np.array(list(self.multi_scannables[seq_name][0]))
                    self.x_label[seq_name] = self.scan_params[seq_name][0][0]
                    f = seq_name + "-" if len(self.scan_params) > 1 else ""
                    f += self.x_label[seq_name]
                    # print("FFFF: ", f)
                    setattr(self, f, x_array)
                    dims.append(N)
                self.set_dataset("{}-raw_data".format(seq_name), np.full(dims, np.nan))

        # Setup for saving data
        self.timestamp = None
        self.dir = os.path.join(os.path.expanduser("~"), "data",
                                datetime.now().strftime("%Y-%m-%d"), type(self).__name__)
        os.makedirs(self.dir, exist_ok=True)
        os.chdir(self.dir)

        self.run_initially()

    def run(self):
        line_trigger = self.p.line_trigger_settings.enabled
        line_offset = float(self.p.line_trigger_settings.offset_duration)
        line_offset = self.core.seconds_to_mu((16 + line_offset)*ms)

        for seq_name, scan_list in self.multi_scannables.items():
            self.sequence = getattr(self, seq_name)
            if len(self.multi_scannables) > 1:
                dir_ = seq_name
            else:
                dir_ = ""

            collection, param = self.scan_params[seq_name][0][0].split(".")
            for i, scan_value in enumerate(scan_list[0]):
                self.p[collection].update({param: scan_value})
                
                print("point: ", i)
                
                for j in range(self.N):
                    raw_run_data = []
                    # print("\nRun nos: ", i, j, "\n")
                    if self.scheduler.check_pause():
                        try:
                            self.scheduler.pause()
                        except TerminationRequested:
                            break
                    self.single_run(line_trigger, line_offset)
            
                    # readout
                    if "pmt" in self.rm:
                        duration = self.p.StateReadout.pmt_readout_duration
                        # print("Duration: ", duration)
                        pmt_count = self.pmt_readout(duration)
                        raw_run_data.append(pmt_count)
                
                self.record_result("{}-raw_data".format(seq_name), i, raw_run_data)

                if self.rm == "pmt":
                    name = "{}-pmt_ion{}"
                    data = sorted(self.get_dataset("{}-raw_data".format(seq_name))[i])
                    idxs = [0]
                    for threshold in self.p.StateReadout.threshold_list:
                        idxs.append(bisect(data, threshold))
                    idxs.append(self.N)
                    for k in range(self.n_ions):
                        dataset = getattr(self, name.format(seq_name, k))
                        if idxs[k + 1] == idxs[k]:
                            dataset[i] = 0
                        else:
                            dataset[i] = idxs[k + 1] - idxs[k]
                        self.save_and_send_to_rcg(
                            getattr(self, seq_name + "-" + self.x_label[seq_name] if dir_ else self.x_label[seq_name])[:i+1],
                            dataset[:i + 1], name.format(seq_name, k), dir_=dir_)
            
                    rem = (i + 1) % 5
                    if rem == 0:
                        self.save_result(self.x_label[seq_name], 
                            getattr(self, seq_name + "-" + self.x_label[seq_name] if dir_ else self.x_label[seq_name])[i - 4:i + 1], 
                            dir_=dir_, xdata=True)
                        for k in range(self.n_ions):
                            self.save_result(name.format(seq_name, k), 
                                getattr(self, name.format(seq_name, k))[i - 4:i + 1], dir_=dir_)
                        hist_data = self.get_dataset(seq_name + "-raw_data")
                        try:
                            hist_data = hist_data[i - 4:i + 1]
                        except IndexError:
                            hist_data = hist_data[i - 4:]
                        self.send_to_hist(hist_data.flatten())

            else:
                rem = (i + 1) % 5
                if self.rm == "pmt":
                    self.save_result(self.x_label[seq_name], 
                        getattr(self, seq_name + "-" + self.x_label[seq_name] if dir_ else self.x_label[seq_name])[-rem:], 
                        xdata=True, dir_=dir_)
                    for i in range(self.n_ions):
                        self.save_result(name.format(seq_name, i), getattr(self, name.format(seq_name, i))[-rem:], dir_=dir_)
                        self.send_to_hist(self.get_dataset(seq_name + "-raw_data")[-rem:].flatten())             

        self.reset_cw_settings(self.dds_list, self.freq_list,
                                self.amp_list, self.state_list, self.att_list)

    @kernel
    def pmt_readout(self, duration) -> TInt32:
        self.core.break_realtime()
        t_count = self.pmt.gate_rising(duration)
        return self.pmt.count(t_count)
    
    @kernel
    def single_run(self, line_trigger, line_offset):
        if line_trigger:
            self.line_trigger(line_offset)
        else:
            self.core.reset()
        self.pmt_readout(1*us)

    @kernel
    def rep_run(self):
        pass

    @kernel
    def line_trigger(self, offset):
        # Phase lock to mains
        self.core.reset()
        t_gate = self.LTriggerIN.gate_rising(16*ms)
        trigger_time = self.LTriggerIN.timestamp_mu(t_gate)
        at_mu(trigger_time + offset)
    
    def analyze(self):
        # Is this necessary?
        try:
            self.rcg.close_rpc()
            self.pmt_hist.close_rpc()
        except:
            pass

        self.run_finally()

    @rpc(flags={"async"})
    def save_and_send_to_rcg(self, x, y, name, dir_=""):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%H%M_%S")
            self.filename = self.timestamp + ".h5"
            if dir_:
                os.makedirs(dir_, exist_ok=True)
                file_ = os.path.join(dir_, self.filename)
            else:
                file_ = self.filename
            with h5.File(file_, "w") as f:
                datagrp = f.create_group("scan_data")
                datagrp.attrs["plot_show"] = self.rcg_tab
                f.create_dataset("time", data=[], maxshape=(None,))
                params = f.create_group("parameters")
                for collection in self.p.keys():
                    collectiongrp = params.create_group(collection)
                    for key, val in self.p[collection].items():
                        collectiongrp.create_dataset(key, data=str(val))
            
            with open("../scan_list", "a+") as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=",")
                seq_name = type(self).__name__
                if dir_:
                    seq_name += "_" + dir_ 
                csvwriter.writerow([self.timestamp, seq_name,
                                    os.path.join(self.dir, self.filename)])

            if len(self.x_label) == 1:
                x_label = list(self.x_label.values())[0]
            else:            
                try:
                    x_label = self.x_label[dir_] 
                except Exception as e:
                    # print("\n3\n", e, "\nseq_name: ", seq_name)
                    x_label = "x"
            
            x_file = dir_ + "-" if dir_ else dir_
            self.save_result(x_label, list(getattr(self, x_file +  x_label)), 
                             dir_=dir_, xdata=True)

        if self.rcg is None:
            try:
                self.rcg = Client("::1", 3286, "rcg")
            except:
                return
        try:
            self.rcg.plot(x, y, tab_name=self.rcg_tab,
                          plot_title=self.timestamp + " - " + name, append=True,
                          file_=os.path.join(os.getcwd(), self.filename))
        except:
            return

    @kernel
    def reset_cw_settings(self, dds_list, freq_list, amp_list, 
                          state_list, att_list):
        # Return the CW settings to what they were when prepare
        # stage was run
        self.core.reset()
        for cpld in self.cpld_list:
            cpld.init()
        with parallel:
            for i in range(len(dds_list)):
                dds_list[i].init()
                dds_list[i].set(freq_list[i], amplitude=amp_list[i])
                dds_list[i].set_att(att_list[i]*dB)
                if state_list[i]:
                    dds_list[i].sw.on()
                else:
                    dds_list[i].sw.off()

    @rpc(flags={"async"})
    def record_result(self, dataset, idx, val):
        self.mutate_dataset(dataset, idx, val)

    @rpc(flags={"async"})
    def save_result(self, dataset, data, dir_="", xdata=False):
        if dir_:
            dir_ += "/"
        print("final file: ", dir_ + self.filename)
        with h5.File(dir_ + self.filename, "a") as f:
            datagrp = f["scan_data"]
            try:
                datagrp[dataset]
            except KeyError:
                data = datagrp.create_dataset(dataset, data=data, maxshape=(None,))
                if xdata:
                    data.attrs["x-axis"] = True
                return
            datagrp[dataset].resize(datagrp[dataset].shape[0] + data.shape[0], axis=0)
            datagrp[dataset][-data.shape[0]:] = data

    @rpc(flags={"async"})
    def send_to_hist(self, data):
        self.pmt_hist.plot(data)

    def add_sequence(self, subsequence, replacement_parameters={}):
        new_parameters = self.p.copy()
        for key, val in replacement_parameters.items():
            collection, parameter = key.split(".")
            new_parameters[collection].update({parameter: val})
        subsequence.p = edict(new_parameters)
        subsequence(self).run()

    def setup(self):
        pass
    
    def run_initially(self):
        pass

    @kernel 
    def sequence(self):
        raise NotImplementedError

    def run_finally(self):
        pass