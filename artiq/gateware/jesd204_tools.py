from collections import namedtuple

from migen import *
from migen.genlib.cdc import MultiReg, BusSynchronizer
from migen.genlib.resetsync import AsyncResetSynchronizer
from misoc.interconnect.csr import *

from jesd204b.common import (JESD204BTransportSettings,
                             JESD204BPhysicalSettings,
                             JESD204BSettings)
from jesd204b.phy.gth import GTHChannelPLL as JESD204BGTHChannelPLL
from jesd204b.phy import JESD204BPhyTX
from jesd204b.core import JESD204BCoreTX
from jesd204b.core import JESD204BCoreTXControl


class UltrascaleCRG(Module, AutoCSR):
    linerate = int(6e9)
    refclk_freq = int(150e6)
    fabric_freq = int(125e6)

    def __init__(self, platform, use_rtio_clock=False):
        self.ibuf_disable = CSRStorage(reset=1)
        self.jreset = CSRStorage(reset=1)
        self.jref = Signal()
        self.refclk = Signal()
        self.clock_domains.cd_jesd = ClockDomain()

        refclk2 = Signal()
        refclk_pads = platform.request("dac_refclk", 0)
        platform.add_period_constraint(refclk_pads.p, 1e9/self.refclk_freq)
        self.specials += [
            Instance("IBUFDS_GTE3", i_CEB=self.ibuf_disable.storage, p_REFCLK_HROW_CK_SEL=0b00,
                     i_I=refclk_pads.p, i_IB=refclk_pads.n,
                     o_O=self.refclk, o_ODIV2=refclk2),
            AsyncResetSynchronizer(self.cd_jesd, self.jreset.storage),
        ]

        if use_rtio_clock:
            self.cd_jesd.clk.attr.add("keep")
            self.comb += self.cd_jesd.clk.eq(ClockSignal("rtio"))
        else:
            self.specials += Instance("BUFG_GT", i_I=refclk2, o_O=self.cd_jesd.clk)

        jref = platform.request("dac_sysref")
        jref_se = Signal()
        jref_r = Signal()
        self.specials += [
            Instance("IBUFDS_IBUFDISABLE",
                p_USE_IBUFDISABLE="TRUE", p_SIM_DEVICE="ULTRASCALE",
                i_IBUFDISABLE=self.ibuf_disable.storage,
                i_I=jref.p, i_IB=jref.n,
                o_O=jref_se),
            # SYSREF normally meets s/h at the FPGA, except during margin
            # scan and before full initialization.
            # Be paranoid and use a double-register anyway.
            Instance("FD", i_C=ClockSignal("jesd"), i_D=jref_se, o_Q=jref_r,
                     attr={("IOB", "TRUE")}),
            Instance("FD", i_C=ClockSignal("jesd"), i_D=jref_r, o_Q=self.jref)
        ]


PhyPads = namedtuple("PhyPads", "txp txn")


class UltrascaleTX(Module, AutoCSR):
    def __init__(self, platform, sys_crg, jesd_crg, dac):
        ps = JESD204BPhysicalSettings(l=8, m=4, n=16, np=16)
        ts = JESD204BTransportSettings(f=2, s=2, k=16, cs=0)
        settings = JESD204BSettings(ps, ts, did=0x5a, bid=0x5)

        jesd_pads = platform.request("dac_jesd", dac)
        phys = []
        for i in range(len(jesd_pads.txp)):
            cpll = JESD204BGTHChannelPLL(
                    jesd_crg.refclk, jesd_crg.refclk_freq, jesd_crg.linerate)
            self.submodules += cpll
            phy = JESD204BPhyTX(
                    cpll, PhyPads(jesd_pads.txp[i], jesd_pads.txn[i]),
                    jesd_crg.fabric_freq, transceiver="gth")
            platform.add_period_constraint(phy.transmitter.cd_tx.clk,
                    40*1e9/jesd_crg.linerate)
            platform.add_false_path_constraints(
                sys_crg.cd_sys.clk,
                jesd_crg.cd_jesd.clk,
                phy.transmitter.cd_tx.clk)
            phys.append(phy)

        self.submodules.core = JESD204BCoreTX(
            phys, settings, converter_data_width=64)
        self.submodules.control = JESD204BCoreTXControl(self.core)
        self.core.register_jsync(platform.request("dac_sync", dac))
        self.core.register_jref(jesd_crg.jref)


# See "Digital femtosecond time difference circuit for CERN's timing system"
# by P. Moreira and I. Darwazeh
class DDMTD(Module, AutoCSR):
    def __init__(self, input_pads, rtio_clk_freq=150e6):
        N = 64
        self.dt = CSRStatus(N.bit_length())

        # # #

        self.clock_domains.cd_helper = ClockDomain(reset_less=True)
        helper_fb = Signal()
        helper_output = Signal()

        input_se = Signal()
        beat1 = Signal()
        beat2 = Signal()
        self.specials += [
            Instance("MMCME2_BASE",
                p_CLKIN1_PERIOD=1e9/rtio_clk_freq,
                i_CLKIN1=ClockSignal("rtio"),
                i_RST=ResetSignal("rtio"),

                # VCO at 1200MHz with 150MHz RTIO frequency
                p_CLKFBOUT_MULT_F=8.0,
                p_DIVCLK_DIVIDE=1,

                o_CLKFBOUT=helper_fb, i_CLKFBIN=helper_fb,

                # helper PLL ratio: 64/65 (N=64)
                p_CLKOUT0_DIVIDE_F=8.125,
                o_CLKOUT0=helper_output,
            ),
            Instance("BUFG", i_I=helper_output, o_O=self.cd_helper.clk),
            Instance("IBUFDS", i_I=input_pads.p, i_IB=input_pads.n, o_O=input_se),
            Instance("FD", i_C=self.cd_helper.clk, i_D=input_se, o_Q=beat1),
            Instance("FD", i_C=self.cd_helper.clk, i_D=ClockSignal("rtio"), o_Q=beat2),
        ]

        counting = Signal()
        counter = Signal(N.bit_length())

        beat1_r = Signal()
        beat2_r = Signal()
        result = Signal.like(counter)

        self.sync.helper += [
            If(counting,
                counter.eq(counter + 1)
            ).Else(
                result.eq(counter)
            ),

            beat1_r.eq(beat1),
            If(beat1 & ~beat1_r, counting.eq(1), counter.eq(0)),
            beat2_r.eq(beat2),
            If(beat2 & ~beat2_r, counting.eq(0))
        ]

        bsync = BusSynchronizer(len(result), "helper", "sys")
        self.submodules += bsync
        self.comb += [
            bsync.i.eq(result),
            self.dt.status.eq(bsync.o)
        ]
