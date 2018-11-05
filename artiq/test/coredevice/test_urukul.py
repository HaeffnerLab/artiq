import unittest

from artiq.experiment import *
from artiq.test.hardware_testbench import ExperimentCase
from artiq.coredevice import urukul


class UrukulExp(EnvExperiment):
    def build(self, runner):
        self.setattr_device("core")
        self.dev = self.get_device("urukul_cpld")
        self.runner = runner

    def run(self):
        getattr(self, self.runner)()

    @kernel
    def instantiate(self):
        pass

    @kernel
    def init(self):
        self.core.break_realtime()
        self.dev.init()

    @kernel
    def cfg_write(self):
        self.core.break_realtime()
        self.dev.init()
        self.dev.cfg_write(self.dev.cfg_reg)

    @kernel
    def sta_read(self):
        self.core.break_realtime()
        self.dev.init()
        sta = self.dev.sta_read()
        self.set_dataset("sta", sta)

    @kernel
    def switches(self):
        self.core.break_realtime()
        self.dev.init()
        self.dev.io_rst()
        self.dev.cfg_sw(0, 0)
        self.dev.cfg_sw(0, 1)
        self.dev.cfg_sw(3, 1)
        self.dev.cfg_switches(0b1010)

    @kernel
    def switch_speed(self):
        self.core.break_realtime()
        self.dev.init()
        n = 10
        t0 = self.core.get_rtio_counter_mu()
        for i in range(n):
            self.dev.cfg_sw(3, i & 1)
        self.set_dataset("dt", self.core.mu_to_seconds(
            self.core.get_rtio_counter_mu() - t0)/n)

    @kernel
    def switches_readback(self):
        self.core.reset()  # clear switch TTLs
        self.dev.init()
        sw_set = 0b1010
        self.dev.cfg_switches(sw_set)
        sta_get = self.dev.sta_read()
        self.set_dataset("sw_set", sw_set)
        self.set_dataset("sta_get", sta_get)

    @kernel
    def att(self):
        self.core.break_realtime()
        self.dev.init()
        att_set = 0x12345678
        self.dev.set_all_att_mu(att_set)
        att_get = self.dev.get_att_mu()
        self.set_dataset("att_set", att_set)
        self.set_dataset("att_get", att_get)

    @kernel
    def att_speed(self):
        self.core.break_realtime()
        self.dev.init()
        n = 10
        t0 = self.core.get_rtio_counter_mu()
        for i in range(n):
            self.dev.set_att(3, 30*dB)
        self.set_dataset("dt", self.core.mu_to_seconds(
            self.core.get_rtio_counter_mu() - t0)/n)

    @kernel
    def io_update(self):
        self.core.break_realtime()
        self.dev.init()
        self.dev.io_update.pulse_mu(8)

    @kernel
    def sync(self):
        self.core.break_realtime()
        self.dev.init()
        self.dev.set_sync_div(2)


class UrukulTest(ExperimentCase):
    def test_instantiate(self):
        self.execute(UrukulExp, "instantiate")

    def test_init(self):
        self.execute(UrukulExp, "init")

    def test_cfg_write(self):
        self.execute(UrukulExp, "cfg_write")

    def test_sta_read(self):
        self.execute(UrukulExp, "sta_read")
        sta = self.dataset_mgr.get("sta")
        print(hex(sta))
        self.assertEqual(urukul.urukul_sta_ifc_mode(sta), 0b0001)

    def test_switches(self):
        self.execute(UrukulExp, "switches")

    def test_switch_speed(self):
        self.execute(UrukulExp, "switch_speed")
        self.assertLess(self.dataset_mgr.get("dt"), 3*us)

    def test_switches_readback(self):
        self.execute(UrukulExp, "switches_readback")
        sw_get = urukul.urukul_sta_rf_sw(self.dataset_mgr.get("sta_get"))
        sw_set = self.dataset_mgr.get("sw_set")
        self.assertEqual(sw_get, sw_set)

    def test_att(self):
        self.execute(UrukulExp, "att")
        att_set = self.dataset_mgr.get("att_set")
        att_get = self.dataset_mgr.get("att_get")
        self.assertEqual(att_set, att_get)

    def test_att_speed(self):
        self.execute(UrukulExp, "att_speed")
        self.assertLess(self.dataset_mgr.get("dt"), 3*us)

    def test_io_update(self):
        self.execute(UrukulExp, "io_update")

    def test_sync(self):
        self.execute(UrukulExp, "sync")
