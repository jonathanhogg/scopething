
"""
vm
==

Package capturing BitScope VM specification, including registers, enumerations, flags,
commands and logic for encoding and decoding virtual machine instructions and data.

All names and descriptions copyright BitScope and taken from their [VM specification
document][VM01B].

[VM01B]: https://docs.google.com/document/d/1cZNRpSPAMyIyAvIk_mqgEByaaHzbFTX8hWglAMTlnHY

"""

import array
import asyncio
from collections import namedtuple
from enum import IntEnum
import logging
import struct


LOG = logging.getLogger('vm')


class Register(namedtuple('Register', ['base', 'dtype', 'description'])):
    def encode(self, value):
        sign = self.dtype[0]
        if '.' in self.dtype:
            whole, fraction = map(int, self.dtype[1:].split('.', 1))
            width = whole + fraction
            value = int(round(value * (1 << fraction)))
        else:
            width = int(self.dtype[1:])
        if sign == 'U':
            n = 1 << width
            value = max(0, min(value, n-1))
            bs = struct.pack('<I', value)
        elif sign == 'S':
            n = 1 << (width - 1)
            value = max(-n, min(value, n-1))
            bs = struct.pack('<i', value)
        else:
            raise TypeError("Unrecognised dtype")
        return bs[:width//8]
    def decode(self, bs):
        if len(bs) < 4:
            bs = bs + bytes(4 - len(bs))
        sign = self.dtype[0]
        if sign == 'U':
            value = struct.unpack('<I', bs)[0]
        elif sign == 'S':
            value = struct.unpack('<i', bs)[0]
        else:
            raise TypeError("Unrecognised dtype")
        if '.' in self.dtype:
            whole, fraction = map(int, self.dtype[1:].split('.', 1))
            value = value / (1 << fraction)
        return value
    def calculate_width(self):
        if '.' in self.dtype:
            return sum(map(int, self.dtype[1:].split('.', 1))) // 8
        else:
            return int(self.dtype[1:]) // 8


Registers = {
    "TriggerLogic": Register(0x05, 'U8', "Trigger Logic, one bit per channel (0 => Low, 1 => High)"),
    "TriggerMask": Register(0x06, 'U8', "Trigger Mask, one bit per channel (0 => Don’t Care, 1 => Active)"),
    "SpockOption": Register(0x07, 'U8', "Spock Option Register (see bit definition table for details)"),
    "SampleAddress": Register(0x08, 'U24', "Sample address (write) 24 bit"),
    "SampleCounter": Register(0x0b, 'U24', "Sample address (read) 24 bit"),
    "TriggerIntro": Register(0x32, 'U16', "Edge trigger intro filter counter (samples/2)"),
    "TriggerOutro": Register(0x34, 'U16', "Edge trigger outro filter counter (samples/2)"),
    "TriggerValue": Register(0x44, 'S0.16', "Digital (comparator) trigger (signed)"),
    "TriggerTime": Register(0x40, 'U32', "Stopwatch trigger time (ticks)"),
    "ClockTicks": Register(0x2e, 'U16', "Master Sample (clock) period (ticks)"),
    "ClockScale": Register(0x14, 'U16', "Clock divide by N (low byte)"),
    "TraceOption": Register(0x20, 'U8', "Trace Mode Option bits"),
    "TraceMode": Register(0x21, 'U8', "Trace Mode (see Trace Mode Table)"),
    "TraceIntro": Register(0x26, 'U16', "Pre-trigger capture count (samples)"),
    "TraceDelay": Register(0x22, 'U32', "Delay period (uS)"),
    "TraceOutro": Register(0x2a, 'U16', "Post-trigger capture count (samples)"),
    "Timeout": Register(0x2c, 'U16', "Auto trace timeout (auto-ticks)"),
    "Prelude": Register(0x3a, 'U16', "Buffer prefill value"),
    "BufferMode": Register(0x31, 'U8', "Buffer mode"),
    "DumpMode": Register(0x1e, 'U8', "Dump mode"),
    "DumpChan": Register(0x30, 'U8', "Dump (buffer) Channel (0..127,128..254,255)"),
    "DumpSend": Register(0x18, 'U16', "Dump send (samples)"),
    "DumpSkip": Register(0x1a, 'U16', "Dump skip (samples)"),
    "DumpCount": Register(0x1c, 'U16', "Dump size (samples)"),
    "DumpRepeat": Register(0x16, 'U16', "Dump repeat (iterations)"),
    "StreamIdent": Register(0x36, 'U8', "Stream data token"),
    "StampIdent": Register(0x3c, 'U8', "Timestamp token"),
    "AnalogEnable": Register(0x37, 'U8', "Analog channel enable (bitmap)"),
    "DigitalEnable": Register(0x38, 'U8', "Digital channel enable (bitmap)"),
    "SnoopEnable": Register(0x39, 'U8', "Frequency (snoop) channel enable (bitmap)"),
    "Cmd": Register(0x46, 'U8', "Command Vector"),
    "Mode": Register(0x47, 'U8', "Operation Mode (per command)"),
    "Option": Register(0x48, 'U16', "Command Option (bits fields per command)"),
    "Size": Register(0x4a, 'U16', "Operation (unit/block) size"),
    "Index": Register(0x4c, 'U16', "Operation index (eg, P Memory Page)"),
    "Address": Register(0x4e, 'U16', "General purpose address"),
    "Clock": Register(0x50, 'U16', "Sample (clock) period (ticks)"),
    "Modulo": Register(0x52, 'U16', "Modulo Size (generic)"),
    "Level": Register(0x54, 'U0.16', "Output (analog) attenuation (unsigned)"),
    "Offset": Register(0x56, 'S0.16', "Output (analog) offset (signed)"),
    "Mask": Register(0x58, 'U16', "Translate source modulo mask"),
    "Ratio": Register(0x5a, 'U16.16', "Translate command ratio (phase step)"),
    "Mark": Register(0x5e, 'U16', "Mark count/phase (ticks/step)"),
    "Space": Register(0x60, 'U16', "Space count/phase (ticks/step)"),
    "Rise": Register(0x82, 'U16', "Rising edge clock (channel 1) phase (ticks)"),
    "Fall": Register(0x84, 'U16', "Falling edge clock (channel 1) phase (ticks)"),
    "Control": Register(0x86, 'U8', "Clock Control Register (channel 1)"),
    "Rise2": Register(0x88, 'U16', "Rising edge clock (channel 2) phase (ticks)"),
    "Fall2": Register(0x8a, 'U16', "Falling edge clock (channel 2) phase (ticks)"),
    "Control2": Register(0x8c, 'U8', "Clock Control Register (channel 2)"),
    "Rise3": Register(0x8e, 'U16', "Rising edge clock (channel 3) phase (ticks)"),
    "Fall3": Register(0x90, 'U16', "Falling edge clock (channel 3) phase (ticks)"),
    "Control3": Register(0x92, 'U8', "Clock Control Register (channel 3)"),
    "EepromData": Register(0x10, 'U8', "EE Data Register"),
    "EepromAddress": Register(0x11, 'U8', "EE Address Register"),
    "ConverterLo": Register(0x64, 'U0.16', "VRB ADC Range Bottom (D Trace Mode)"),
    "ConverterHi": Register(0x66, 'U0.16', "VRB ADC Range Top (D Trace Mode)"),
    "TriggerLevel": Register(0x68, 'U0.16', "Trigger Level (comparator, unsigned)"),
    "LogicControl": Register(0x74, 'U8', "Logic Control"),
    "Rest": Register(0x78, 'U16', "DAC (rest) level"),
    "KitchenSinkA": Register(0x7b, 'U8', "Kitchen Sink Register A"),
    "KitchenSinkB": Register(0x7c, 'U8', "Kitchen Sink Register B"),
    "Map0": Register(0x94, 'U8', "Peripheral Pin Select Channel 0"),
    "Map1": Register(0x95, 'U8', "Peripheral Pin Select Channel 1"),
    "Map2": Register(0x96, 'U8', "Peripheral Pin Select Channel 2"),
    "Map3": Register(0x97, 'U8', "Peripheral Pin Select Channel 3"),
    "Map4": Register(0x98, 'U8', "Peripheral Pin Select Channel 4"),
    "Map5": Register(0x99, 'U8', "Peripheral Pin Select Channel 5"),
    "Map6": Register(0x9a, 'U8', "Peripheral Pin Select Channel 6"),
    "Map7": Register(0x9b, 'U8', "Peripheral Pin Select Channel 7"),
    "MasterClockN": Register(0xf7, 'U8', "PLL prescale (DIV N)"),
    "MasterClockM": Register(0xf8, 'U16', "PLL multiplier (MUL M)"),
}

class TraceMode(IntEnum):
    Analog         =  0
    Mixed          =  1
    AnalogChop     =  2
    MixedChop      =  3
    AnalogFast     =  4
    MixedFast      =  5
    AnalogFastChop =  6
    MixedFastChop  =  7
    AnalogShot     = 11
    MixedShot      = 12
    LogicShot      = 13
    Logic          = 14
    LogicFast      = 15
    AnalogShotChop = 16
    MixedShotChop  = 17
    Macro          = 18
    MacroChop      = 19

class BufferMode(IntEnum):
    Single    = 0
    Chop      = 1
    Dual      = 2
    ChopDual  = 3
    Macro     = 4
    MacroChop = 5

class DumpMode(IntEnum):
    Raw    = 0
    Burst  = 1
    Summed = 2
    MinMax = 3
    AndOr  = 4
    Native = 5
    Filter = 6
    Span   = 7

class SpockOption(IntEnum):
    TriggerInvert                 = 0x40
    TriggerSourceA                = 0x04 * 0
    TriggerSourceB                = 0x04 * 1
    TriggerSwap                   = 0x02
    TriggerTypeSampledAnalog      = 0x01 * 0
    TriggerTypeHardwareComparator = 0x01 * 1

class KitchenSinkA(IntEnum):
    ChannelAComparatorEnable = 0x80
    ChannelBComparatorEnable = 0x40

class KitchenSinkB(IntEnum):
    AnalogFilterEnable       = 0x80
    WaveformGeneratorEnable  = 0x40

class TraceStatus(IntEnum):
    Done = 0x00
    Auto = 0x01
    Wait = 0x02
    Stop = 0x03

CaptureMode = namedtuple('CaptureMode', ('clock_low', 'clock_high', 'clock_max', 'analog_channels', 'sample_width',
                                         'logic_channels', 'clock_divide', 'trace_mode', 'buffer_mode'))

CaptureModes = [
    CaptureMode(40, 65535, None, 1, 2, False, False, TraceMode.Macro,          BufferMode.Macro),
    CaptureMode(40, 65535, None, 2, 2, False, False, TraceMode.MacroChop,      BufferMode.MacroChop),
    CaptureMode(15,    40, None, 1, 1, False, True,  TraceMode.Analog,         BufferMode.Single),
    CaptureMode(13,    40, None, 2, 1, False, True,  TraceMode.AnalogChop,     BufferMode.Chop),
    CaptureMode( 8,    14, None, 1, 1, False, False, TraceMode.AnalogFast,     BufferMode.Single),
    CaptureMode( 8,    40, None, 2, 1, False, False, TraceMode.AnalogFastChop, BufferMode.Chop),
    CaptureMode( 2,     7, 5,    1, 1, False, False, TraceMode.AnalogShot,     BufferMode.Single),
    CaptureMode( 4,     7, 5,    2, 1, False, False, TraceMode.AnalogShotChop, BufferMode.Chop),
    CaptureMode( 5, 16384, None, 0, 1, True,  False, TraceMode.Logic,          BufferMode.Single),
    CaptureMode( 4,     4, None, 0, 1, True,  False, TraceMode.LogicFast,      BufferMode.Single),
    CaptureMode( 1,     3, None, 0, 1, True,  False, TraceMode.LogicShot,      BufferMode.Single),
    CaptureMode(15,    40, None, 1, 1, True,  True,  TraceMode.Mixed,          BufferMode.Dual),
    CaptureMode(13,    40, None, 2, 1, True,  True,  TraceMode.MixedChop,      BufferMode.ChopDual),
    CaptureMode( 8,    14, None, 1, 1, True,  False, TraceMode.MixedFast,      BufferMode.Dual),
    CaptureMode( 8,    40, None, 2, 1, True,  False, TraceMode.MixedFastChop,  BufferMode.ChopDual),
    CaptureMode( 2,     7, 5,    1, 1, True,  False, TraceMode.MixedShot,      BufferMode.Dual),
    CaptureMode( 4,     7, 5,    2, 1, True,  False, TraceMode.MixedShotChop,  BufferMode.ChopDual),
]


class VirtualMachine:

    class Transaction:
        def __init__(self, vm):
            self._vm = vm
        def append(self, cmd):
            self._data += cmd
        async def __aenter__(self):
            self._data = b''
            self._vm._transactions.append(self)
            return self
        async def __aexit__(self, exc_type, exc_value, traceback):
            if self._vm._transactions.pop() != self:
                raise RuntimeError("Mis-ordered transactions")
            if exc_type is None:
                await self._vm.issue(self._data)
            return False

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer
        self._transactions = []
        self._reply_buffer = b''

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._reader = None

    __del__ = close

    def transaction(self):
        return self.Transaction(self)

    async def issue(self, cmd):
        if isinstance(cmd, str):
            cmd = cmd.encode('ascii')
        if not self._transactions:
            LOG.debug(f"Issue: {cmd!r}")
            self._writer.write(cmd)
            await self._writer.drain()
            echo = await self._reader.readexactly(len(cmd))
            if echo != cmd:
                raise RuntimeError("Mismatched response")
        else:
            self._transactions[-1].append(cmd)

    async def read_replies(self, n):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        replies = []
        data, self._reply_buffer = self._reply_buffer, b''
        while len(replies) < n:
            index = data.find(b'\r')
            if index >= 0:
                reply = data[:index]
                LOG.debug(f"Read reply: {reply!r}")
                replies.append(reply)
                data = data[index+1:]
            else:
                data += await self._reader.read()
        if data:
            self._reply_buffer = data
        return replies

    async def reset(self):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        LOG.debug("Issue reset")
        self._writer.write(b'!')
        await self._writer.drain()
        while not (await self._reader.read()).endswith(b'!'):
            pass
        self._reply_buffer = b''
        LOG.debug("Reset complete")

    async def set_registers(self, **kwargs):
        cmd = ''
        r0 = r1 = None
        for base, name in sorted((Registers[name].base, name) for name in kwargs):
            register = Registers[name]
            bs = register.encode(kwargs[name])
            LOG.debug(f"{name} = 0x{''.join(f'{b:02x}' for b in reversed(bs))}")
            for i, byte in enumerate(bs):
                if cmd:
                    cmd += 'z'
                    r1 += 1
                address = base + i
                if r1 is None or address > r1 + 3:
                    cmd += f'{address:02x}@'
                    r0 = r1 = address
                else:
                    cmd += 'n' * (address - r1)
                    r1 = address
                if byte != r0:
                    cmd += '[' if byte == 0 else f'{byte:02x}'
                    r0 = byte
        if cmd:
            await self.issue(cmd + 's')

    async def get_register(self, name):
        register = Registers[name]
        await self.issue(f'{register.base:02x}@p')
        values = []
        width = register.calculate_width()
        for i in range(width):
            values.append(int((await self.read_replies(2))[1], 16))
            if i < width-1:
                await self.issue(b'np')
        return register.decode(bytes(values))

    async def issue_get_revision(self):
        await self.issue(b'?')

    async def issue_capture_spock_registers(self):
        await self.issue(b'<')

    async def issue_program_spock_registers(self):
        await self.issue(b'>')

    async def issue_configure_device_hardware(self):
        await self.issue(b'U')

    async def issue_streaming_trace(self):
        await self.issue(b'T')

    async def issue_triggered_trace(self):
        await self.issue(b'D')

    async def read_analog_samples(self, n, sample_width):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        if sample_width == 2:
            data = await self._reader.readexactly(2*n)
            return array.array('f', ((value+32768)/65535 for (value,) in struct.iter_unpack('>h', data)))
        elif sample_width == 1:
            data = await self._reader.readexactly(n)
            return array.array('f', (value/255 for value in data))
        else:
            raise ValueError(f"Bad sample width: {sample_width}")

    async def read_logic_samples(self, n):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        return await self._reader.readexactly(n)

    async def issue_cancel_trace(self):
        await self.issue(b'K')

    async def issue_sample_dump_csv(self):
        await self.issue(b'S')

    async def issue_analog_dump_binary(self):
        await self.issue(b'A')

    async def issue_wavetable_read(self):
        await self.issue(b'R')

    async def wavetable_read_bytes(self, n):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        return await self._reader.readexactly(n)

    async def wavetable_write_bytes(self, bs):
        cmd = ''
        last_byte = None
        for byte in bs:
            if byte != last_byte:
                cmd += f'{byte:02x}'
            cmd += 'W'
            last_byte = byte
        if cmd:
            await self.issue(cmd)

    async def issue_synthesize_wavetable(self):
        await self.issue(b'Y')

    async def issue_translate_wavetable(self):
        await self.issue(b'X')

    async def issue_control_waveform_generator(self):
        await self.issue(b'Z')

    async def issue_read_eeprom(self):
        await self.issue(b'r')

    async def issue_write_eeprom(self):
        await self.issue(b'w')


