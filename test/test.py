# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer, with_timeout

# 48 MHz clock — has to be an even integer in ps for cocotb 2.x.
CLK_PERIOD_PS = 20834  # ~47.999 MHz, close enough to the 48 MHz design target.
BAUD = 115200

# In gate-level simulation we can only inspect the top-level pins; internal
# RTL hierarchy has been flattened away.
GL_TEST = os.environ.get("GATES") == "yes"


async def reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 10)


async def uart_send_byte(dut, byte):
    """Drive ui_in[3] with a 115200-baud UART frame: start, 8 data LSB-first, stop."""
    bit_time = Timer(int(1e12 / BAUD), unit="ps")
    ui = int(dut.ui_in.value)

    def set_rx(level):
        dut.ui_in.value = (ui & ~(1 << 3)) | ((level & 1) << 3)

    set_rx(0)  # start
    await bit_time
    for i in range(8):
        set_rx((byte >> i) & 1)
        await bit_time
    set_rx(1)  # stop
    await bit_time


@cocotb.test()
async def test_reset_outputs(dut):
    """After reset, the design exposes the expected static output state."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_PS, unit="ps").start())
    await reset(dut)

    # Wait long enough for one UART bit period so any reset-time glitching settles.
    await Timer(int(1e12 / BAUD), unit="ps")

    uio_oe = int(dut.uio_oe.value)
    uo_out = int(dut.uo_out.value)

    # uio[2] (dp_pu_o) is always driven as output.
    assert (uio_oe >> 2) & 1, f"uio_oe={uio_oe:#04x}"

    # No USB host has configured us yet — uo_out[7] (configured) should be low.
    assert (uo_out >> 7) & 1 == 0, f"uo_out={uo_out:#04x}"

    # UART TX line idles high (uo_out[4]).
    assert (uo_out >> 4) & 1 == 1, f"uo_out={uo_out:#04x}"


@cocotb.test(skip=GL_TEST)
async def test_uart_rx_decodes_byte(dut):
    """Drive a real 115200-baud frame on the RX pin; check the UART RX decoder catches it."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_PS, unit="ps").start())
    await reset(dut)

    # Idle UART line is high.
    dut.ui_in.value = 1 << 3
    await ClockCycles(dut.clk, 50)

    rx = dut.user_project.u_uart_rx
    expected = 0xA5

    async def wait_for_valid():
        while True:
            await RisingEdge(dut.clk)
            if rx.uart_rx_valid.value == 1:
                return

    cocotb.start_soon(uart_send_byte(dut, expected))
    await with_timeout(wait_for_valid(), 200, "us")

    got = int(rx.uart_rx_data.value)
    assert got == expected, f"uart_rx_data={got:#04x}, expected={expected:#04x}"
