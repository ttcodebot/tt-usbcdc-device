# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer, with_timeout

from usb_host import UsbHost

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


@cocotb.test(timeout_time=300, timeout_unit="ms")
async def test_device_descriptor(dut):
    """Drive a (very) minimal USB host through enumeration and read the Device Descriptor."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_PS, unit="ps").start())
    await reset(dut)

    # Idle the USB lines in J (D+ high, D- low) before the device's TSIGATT timer expires.
    dut.uio_in.value = 0b01  # dp=1, dn=0

    host = UsbHost(dut)

    # Wait for the device to assert dp_pu_o (uio_out[2]) — the phy_rx requires
    # ~16 ms TSIGATT before pull-up. This is the bulk of the simulation time.
    async def wait_for_pull_up():
        while True:
            await RisingEdge(dut.clk)
            if (int(dut.uio_out.value) >> 2) & 1:
                return

    await with_timeout(wait_for_pull_up(), 25, "ms")

    # The phy_rx then needs another ~64 us before rx_en goes high. Give it some slack.
    await Timer(200, unit="us")

    # Drive a USB bus reset (SE0) for >=10 us, then return to J.
    await host.bus_reset(microseconds=12.0)

    # Recovery — give the device a moment to come out of reset (well below 10ms).
    await Timer(200, unit="us")

    desc = await host.get_device_descriptor(length=18, max_packet=8)
    assert desc is not None, "did not receive device descriptor"
    assert len(desc) == 18, f"wrong length: {len(desc)}"

    bLength = desc[0]
    bDescriptorType = desc[1]
    bcdUSB = desc[2] | (desc[3] << 8)
    bDeviceClass = desc[4]
    bMaxPacketSize0 = desc[7]
    idVendor = desc[8] | (desc[9] << 8)
    idProduct = desc[10] | (desc[11] << 8)

    dut._log.info(
        "device descriptor: bLength=%d bDescriptorType=0x%02x bcdUSB=0x%04x "
        "bDeviceClass=0x%02x bMaxPacketSize0=%d idVendor=0x%04x idProduct=0x%04x",
        bLength, bDescriptorType, bcdUSB, bDeviceClass, bMaxPacketSize0, idVendor, idProduct,
    )

    assert bLength == 0x12, f"bLength={bLength:#04x}"
    assert bDescriptorType == 0x01, f"bDescriptorType={bDescriptorType:#04x}"
    assert bcdUSB in (0x0100, 0x0110, 0x0200, 0x0210), f"bcdUSB={bcdUSB:#06x}"
    assert idVendor == 0x1209, f"idVendor={idVendor:#06x}"
    assert idProduct == 0x5454, f"idProduct={idProduct:#06x}"
