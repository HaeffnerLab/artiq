eem_fmc_connections = {
    0: [0, 8, 2, 3, 4, 5, 6, 7],
    1: [1, 9, 10, 11, 12, 13, 14, 15],
    2: [17, 16, 24, 19, 20, 21, 22, 23],
    3: [18, 25, 26, 27, 28, 29, 30, 31],
}


def shiftreg_bits(eem, out_pins):
    r = 0
    for i in range(8):
        if i in out_pins:
            shift = eem_fmc_connections[eem][i]
            r |= 1 << shift
    return r


dio_bank0_out_pins = set(range(4))
dio_bank1_out_pins = set(range(4, 8))
urukul_out_pins = {
    0,       # clk
    1,       # mosi
    3, 4, 5, # cs_n
    6,       # io_update
    7,       # dds_reset
}
urukul_out_pins_aux = {
    4,       # sw0
    5,       # sw1
    6,       # sw2
    7,       # sw3
}
zotino_out_pins = {
    0,       # clk
    1,       # mosi
    3, 4,    # cs_n
    5,       # ldac_n
    7,       # clr_n
}
