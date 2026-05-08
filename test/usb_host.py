# SPDX-FileCopyrightText: (c) 2026 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
"""
A tiny USB 1.1 / 2.0 full-speed host model for the ulixxe usb_cdc core
running inside `tt_um_urish_usb_cdc`.

The model lives entirely in Python and drives the device's D+/D- pins
through `uio_in[0]`/`uio_in[1]`. It samples the device's transmissions
via `uio_out[0]`/`uio_out[1]` while the device asserts `usb_tx_en`
(`uio_oe[0]==1`).

It implements only what's needed to coax a Device Descriptor out of the
DUT:
    - NRZI encode/decode with bit stuffing.
    - Sync + EOP framing.
    - CRC5 for tokens, CRC16 for data payloads.
    - SETUP / IN / ACK packets and DATA0/DATA1 emit + decode.
    - A high-level `get_device_descriptor()` helper that follows the
      standard control-read sequence.
"""

from __future__ import annotations

from cocotb.triggers import RisingEdge

# BIT_SAMPLES inside the design. 4 clk_i cycles == 1 USB FS bit.
BIT_SAMPLES = 4

# USB PIDs (4-bit identifier, low nibble of the PID byte).
PID_OUT = 0x1
PID_IN = 0x9
PID_SOF = 0x5
PID_SETUP = 0xD
PID_DATA0 = 0x3
PID_DATA1 = 0xB
PID_ACK = 0x2
PID_NAK = 0xA
PID_STALL = 0xE


def pid_byte(pid4: int) -> int:
    """USB PID byte = (~pid << 4) | pid (low nibble)."""
    return ((~pid4) & 0xF) << 4 | (pid4 & 0xF)


def crc5(value: int, nbits: int) -> int:
    """USB token CRC5 (polynomial 0x05, init 0x1F, residue 0x0C, output ~ then bit-reversed for tx)."""
    crc = 0x1F
    for i in range(nbits):
        bit = (value >> i) & 1
        if (bit ^ ((crc >> 4) & 1)) == 1:
            crc = ((crc << 1) & 0x1F) ^ 0x05
        else:
            crc = (crc << 1) & 0x1F
    return crc & 0x1F


def crc16(data: bytes) -> int:
    """USB data CRC16 (polynomial 0x8005, init 0xFFFF, residue 0x800D)."""
    crc = 0xFFFF
    for byte in data:
        for i in range(8):
            bit = (byte >> i) & 1
            if (bit ^ ((crc >> 15) & 1)) == 1:
                crc = ((crc << 1) & 0xFFFF) ^ 0x8005
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _bit_reverse(value: int, nbits: int) -> int:
    out = 0
    for i in range(nbits):
        if value & (1 << i):
            out |= 1 << (nbits - 1 - i)
    return out


def make_token(pid4: int, addr: int, endp: int) -> bytes:
    """Build a 3-byte token packet (PID + 11-bit addr/endp + CRC5)."""
    addr &= 0x7F
    endp &= 0xF
    token11 = (addr & 0x7F) | ((endp & 0xF) << 7)  # addr in [6:0], endp in [10:7]
    crc = crc5(token11, 11) ^ 0x1F  # invert
    crc_rev = _bit_reverse(crc, 5)
    word = token11 | (crc_rev << 11)  # 16 bits, two bytes
    return bytes([pid_byte(pid4), word & 0xFF, (word >> 8) & 0xFF])


def make_data(pid4: int, payload: bytes) -> bytes:
    """Build a data packet: PID + payload + CRC16."""
    crc = crc16(payload) ^ 0xFFFF
    crc_lo = _bit_reverse(crc & 0xFF, 8)
    crc_hi = _bit_reverse((crc >> 8) & 0xFF, 8)
    return bytes([pid_byte(pid4)]) + payload + bytes([crc_hi, crc_lo])


def make_handshake(pid4: int) -> bytes:
    return bytes([pid_byte(pid4)])


def nrzi_encode_with_sync_eop(packet: bytes) -> list[int]:
    """Encode a packet (raw bytes, LSB first) into a list of line states.

    Each list entry is one bit time. Line state values:
        'J' = D+ high, D- low
        'K' = D+ low,  D- high
        '0' = SE0 (both low) -- only used for EOP.

    The output starts with the SYNC (KJKJKJKK as NRZI bits = 0x80 LSB first),
    then the bytes in order, then 2 SE0s + 1 J (EOP).

    Bit stuffing is applied: after 6 consecutive 1s, a 0 is inserted.
    NRZI: 0 -> toggle, 1 -> hold.
    """
    # Build the raw bit stream (LSB first within each byte).
    # The SYNC pattern in USB is 0x80 LSB-first = 0,0,0,0,0,0,0,1
    bits: list[int] = []
    sync_byte = 0x80  # canonical SYNC
    full = bytes([sync_byte]) + packet
    for byte in full:
        for i in range(8):
            bits.append((byte >> i) & 1)

    # Apply bit stuffing
    stuffed: list[int] = []
    ones = 0
    for b in bits:
        stuffed.append(b)
        if b == 1:
            ones += 1
            if ones == 6:
                stuffed.append(0)  # stuff bit
                ones = 0
        else:
            ones = 0

    # NRZI encode. Start in J. 0 -> flip, 1 -> hold.
    states: list[str] = []
    cur = "J"
    for b in stuffed:
        if b == 0:
            cur = "K" if cur == "J" else "J"
        states.append(cur)

    # EOP: SE0, SE0, J
    states.append("0")
    states.append("0")
    states.append("J")
    return states


class UsbHost:
    """Minimal USB host that drives the DUT's D+/D- through uio_in."""

    def __init__(self, dut, bit_samples: int = BIT_SAMPLES, addr: int = 0):
        self.dut = dut
        self.bit_samples = bit_samples
        self.addr = addr
        self._toggle_in = 0  # next IN expected DATA0/DATA1 ; ctrl IN starts DATA1
        # Idle the bus in J: dp=1, dn=0 — set the upper bits to 0.
        self._idle()

    # ---- low-level line driving --------------------------------------

    def _set_line(self, dp: int, dn: int):
        """Drive uio_in[1:0] = {dn, dp}. Preserve other ui bits (currently 0)."""
        cur = int(self.dut.uio_in.value) & ~0b11
        self.dut.uio_in.value = cur | (dp & 1) | ((dn & 1) << 1)

    def _set_state(self, st: str):
        if st == "J":
            self._set_line(1, 0)
        elif st == "K":
            self._set_line(0, 1)
        elif st == "0":
            self._set_line(0, 0)
        else:
            raise ValueError(st)

    def _idle(self):
        self._set_state("J")

    async def _wait_bit(self):
        # 1 USB FS bit = bit_samples clk_i cycles.
        for _ in range(self.bit_samples):
            await RisingEdge(self.dut.clk)

    # ---- packet TX --------------------------------------------------

    async def send_packet(self, packet: bytes):
        states = nrzi_encode_with_sync_eop(packet)
        for st in states:
            self._set_state(st)
            await self._wait_bit()
        # Return to idle J (already J after EOP), hold for some cycles
        self._idle()
        # Inter-packet gap: at least a couple of bit times of J
        for _ in range(self.bit_samples * 2):
            await RisingEdge(self.dut.clk)

    async def bus_reset(self, microseconds: float = 12.0):
        """Hold SE0 for >=10us, then return to J idle."""
        # Sim has 48MHz clock => 1us = 48 cycles
        cycles = int(48 * microseconds)
        self._set_state("0")
        for _ in range(cycles):
            await RisingEdge(self.dut.clk)
        self._idle()

    # ---- packet RX --------------------------------------------------

    def _read_lines(self) -> tuple[int, int]:
        """Sample the current dp/dn the host sees (device drives if uio_oe[0]==1)."""
        oe = int(self.dut.uio_oe.value)
        if oe & 1:
            uo = int(self.dut.uio_out.value)
            return uo & 1, (uo >> 1) & 1
        # otherwise the host's drive is what's on the line
        ui = int(self.dut.uio_in.value)
        return ui & 1, (ui >> 1) & 1

    async def _wait_device_drive(self, timeout_cycles: int) -> bool:
        """Wait for the device to start driving (uio_oe[0]==1)."""
        for _ in range(timeout_cycles):
            await RisingEdge(self.dut.clk)
            if int(self.dut.uio_oe.value) & 1:
                return True
        return False

    async def receive_packet(self, timeout_cycles: int = 200000) -> bytes | None:
        """Wait for device to transmit, sample its packet, decode and return raw bytes (PID + payload + CRC)."""
        if not await self._wait_device_drive(timeout_cycles):
            return None
        # Device is now driving. Sample at the centre of each bit.
        # Walk the line until we lock onto SYNC. We sample once per bit time.
        # The phy_tx starts in J (idle), then begins SYNC by toggling to K.
        # Since the device just started driving, current state is likely K (first SYNC bit).
        # Strategy: sample once per bit (bit_samples cycles). Detect transitions.
        # For decoding: we look for SYNC pattern (0x80 LSB-first = K,J,K,J,K,J,K,K),
        # after NRZI decoding that's bits 0,0,0,0,0,0,0,1.
        # Easiest approach: continuously sample, build bit history, then decode once
        # we see EOP (>=2 SE0s).

        # Sample center: align by waiting half a bit time first.
        for _ in range(self.bit_samples // 2):
            await RisingEdge(self.dut.clk)

        samples: list[str] = []  # 'J', 'K', or '0'
        # We must keep sampling until we see the EOP (SE0 SE0).
        # Hard limit on packet length to avoid hangs.
        max_bits = 2048
        eop_se0_count = 0
        for _ in range(max_bits):
            dp, dn = self._read_lines()
            if dp == 1 and dn == 0:
                samples.append("J")
                eop_se0_count = 0
            elif dp == 0 and dn == 1:
                samples.append("K")
                eop_se0_count = 0
            elif dp == 0 and dn == 0:
                samples.append("0")
                eop_se0_count += 1
                if eop_se0_count >= 2:
                    # EOP detected, advance one more bit (J recovery) then break
                    for _ in range(self.bit_samples):
                        await RisingEdge(self.dut.clk)
                    break
            else:
                samples.append("J")  # SE1 shouldn't happen
            for _ in range(self.bit_samples):
                await RisingEdge(self.dut.clk)
        else:
            return None

        # Decode samples: NRZI -> bits, drop SYNC, un-bit-stuff, group to bytes.
        # Find SYNC start: the device's idle-before-sync is J (we're sampling
        # while device drives — first sample may already be K).
        # Trim trailing SE0s.
        while samples and samples[-1] == "0":
            samples.pop()
        # Find the K transition that marks the SYNC start: scan from the front.
        i = 0
        while i < len(samples) and samples[i] == "J":
            i += 1
        if i == len(samples):
            return None
        bitstream = samples[i:]

        # NRZI decode: previous state J. 0 = transition, 1 = no-transition.
        prev = "J"
        bits: list[int] = []
        for s in bitstream:
            if s == "0":
                break
            if s == prev:
                bits.append(1)
            else:
                bits.append(0)
            prev = s

        # Un-bit-stuff: remove the 0 inserted after every six 1s.
        unstuffed: list[int] = []
        ones = 0
        i = 0
        while i < len(bits):
            b = bits[i]
            unstuffed.append(b)
            if b == 1:
                ones += 1
                if ones == 6:
                    # next bit must be the stuff 0; skip it
                    i += 1
                    if i < len(bits):
                        # ignore the stuff bit
                        pass
                    ones = 0
            else:
                ones = 0
            i += 1

        # First 8 bits are SYNC (0x80 LSB first). Strip them.
        if len(unstuffed) < 8:
            return None
        unstuffed = unstuffed[8:]

        # Group to bytes (LSB first).
        out = bytearray()
        for b in range(0, len(unstuffed) - (len(unstuffed) % 8), 8):
            byte = 0
            for j in range(8):
                byte |= unstuffed[b + j] << j
            out.append(byte)
        return bytes(out)

    # ---- high-level transactions -----------------------------------

    @staticmethod
    def _decode_pid(byte: int) -> int | None:
        lo = byte & 0xF
        hi = (byte >> 4) & 0xF
        if (lo ^ hi) != 0xF:
            return None
        return lo

    async def setup_transaction(self, addr: int, endp: int, setup8: bytes) -> bool:
        """Issue SETUP token + DATA0(setup8) and read ACK."""
        await self.send_packet(make_token(PID_SETUP, addr, endp))
        # small inter-packet gap
        for _ in range(self.bit_samples * 2):
            await RisingEdge(self.dut.clk)
        await self.send_packet(make_data(PID_DATA0, setup8))
        resp = await self.receive_packet(timeout_cycles=20000)
        if not resp:
            return False
        pid = self._decode_pid(resp[0])
        return pid == PID_ACK

    async def in_transaction(self, addr: int, endp: int, expected_toggle: int) -> bytes | None:
        """Issue IN token, read DATA0/DATA1, send ACK. Returns payload bytes, or None on NAK/timeout."""
        await self.send_packet(make_token(PID_IN, addr, endp))
        resp = await self.receive_packet(timeout_cycles=20000)
        if not resp:
            return None
        pid = self._decode_pid(resp[0])
        if pid in (PID_NAK, PID_STALL):
            return None
        if pid not in (PID_DATA0, PID_DATA1):
            return None
        # Strip PID + CRC16 (last 2 bytes)
        if len(resp) < 3:
            return None
        payload = bytes(resp[1:-2])
        # Send ACK
        for _ in range(self.bit_samples * 2):
            await RisingEdge(self.dut.clk)
        await self.send_packet(make_handshake(PID_ACK))
        return payload

    async def get_device_descriptor(self, length: int = 18, max_packet: int = 8) -> bytes | None:
        """Run the standard GET_DESCRIPTOR(DEVICE) control read."""
        # Standard SETUP packet for GET_DESCRIPTOR(DEVICE, length)
        # bmRequestType=0x80, bRequest=0x06 (GET_DESCRIPTOR),
        # wValue=0x0100 (DEVICE descriptor index 0),
        # wIndex=0, wLength=length
        setup = bytes([0x80, 0x06, 0x00, 0x01, 0x00, 0x00, length & 0xFF, (length >> 8) & 0xFF])
        ok = await self.setup_transaction(self.addr, 0, setup)
        if not ok:
            return None

        # Now fan out IN tokens until we have all bytes.
        got = bytearray()
        # Try at most a generous number of attempts (NAKs allowed).
        attempts_left = 64
        while len(got) < length and attempts_left > 0:
            attempts_left -= 1
            payload = await self.in_transaction(self.addr, 0, 0)
            if payload is None:
                # NAK: retry after a short pause
                for _ in range(self.bit_samples * 4):
                    await RisingEdge(self.dut.clk)
                continue
            got += payload
            if len(payload) < max_packet:
                break
        if len(got) < length:
            return None
        # Status stage: send OUT-DATA1 ZLP. (Skipped here — not required for the
        # descriptor verification, and the test does not perform a second
        # control transfer.)
        return bytes(got[:length])
