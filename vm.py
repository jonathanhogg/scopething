"""
vm
==

Package capturing BitScope VM specification, including registers, enumerations, flags,
commands and logic for encoding and decoding virtual machine instructions and data.

All names and descriptions copyright BitScope and taken from their [VM specification
document][VM01B] (with slight changes).

[VM01B]: https://docs.google.com/document/d/1cZNRpSPAMyIyAvIk_mqgEByaaHzbFTX8hWglAMTlnHY

"""

# pylama:ignore=E221,C0326,R0904,W1203

import array
from collections import namedtuple
from enum import IntEnum
import logging
import struct

from utils import DotDict


Log = logging.getLogger(__name__)


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
            max_value = (1 << width) - 1
            value = min(max(0, value), max_value)
            data = struct.pack('<I', value)
        elif sign == 'S':
            max_value = (1 << (width - 1))
            value = min(max(-max_value, value), max_value - 1)
            data = struct.pack('<i', value)
        else:
            raise TypeError("Unrecognised dtype")
        return data[:width//8]

    def decode(self, data):
        if len(data) < 4:
            data = data + bytes(4 - len(data))
        sign = self.dtype[0]
        if sign == 'U':
            value = struct.unpack('<I', data)[0]
        elif sign == 'S':
            value = struct.unpack('<i', data)[0]
        else:
            raise TypeError("Unrecognised dtype")
        if '.' in self.dtype:
            fraction = int(self.dtype.split('.', 1)[1])
            value /= (1 << fraction)
        return value

    @property
    def maximum_value(self):
        if '.' in self.dtype:
            whole, fraction = map(int, self.dtype[1:].split('.', 1))
        else:
            whole, fraction = int(self.dtype[1:]), 0
        if self.dtype[0] == 'S':
            whole -= 1
        max_value = (1 << (whole+fraction)) - 1
        return max_value / (1 << fraction) if fraction else max_value

    @property
    def width(self):
        if '.' in self.dtype:
            return sum(map(int, self.dtype[1:].split('.', 1))) // 8
        return int(self.dtype[1:]) // 8


Registers = DotDict({
    "TriggerLogic": Register(0x05, 'U8', "Trigger Logic, one bit per channel (0 => Low, 1 => High)"),
    "TriggerMask": Register(0x06, 'U8', "Trigger Mask, one bit per channel (0 => Don’t Care, 1 => Active)"),
    "SpockOption": Register(0x07, 'U8', "Spock Option Register (see bit definition table for details)"),
    "SampleAddress": Register(0x08, 'U24', "Sample address (write) 24 bit"),
    "SampleCounter": Register(0x0b, 'U24', "Sample address (read) 24 bit"),
    "TriggerIntro": Register(0x32, 'U16', "Edge trigger intro filter counter (samples/2)"),
    "TriggerOutro": Register(0x34, 'U16', "Edge trigger outro filter counter (samples/2)"),
    "TriggerValue": Register(0x44, 'S0.16', "Digital (comparator) trigger (signed)"),
    "TriggerTime": Register(0x40, 'U32', "Stopwatch trigger time (ticks)"),
    "ClockTicks": Register(0x2e, 'U16', "Sample period (ticks)"),
    "ClockScale": Register(0x14, 'U16', "Sample clock divider"),
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
    "PrimaryClockN": Register(0xf7, 'U8', "PLL prescale (DIV N)"),
    "PrimaryClockM": Register(0xf8, 'U16', "PLL multiplier (MUL M)"),
})


class TraceMode(IntEnum):
    Analog         = 0
    Mixed          = 1
    AnalogChop     = 2
    MixedChop      = 3
    AnalogFast     = 4
    MixedFast      = 5
    AnalogFastChop = 6
    MixedFastChop  = 7
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


CaptureMode = namedtuple('CaptureMode', ('trace_mode', 'clock_low', 'clock_high', 'clock_divide',
                                         'analog_channels', 'sample_width', 'logic_channels', 'buffer_mode'))


CaptureModes = [
    CaptureMode(TraceMode.Macro,          40, 16384,     1, 1, 2, False, BufferMode.Macro),
    CaptureMode(TraceMode.MacroChop,      40, 16384,     1, 2, 2, False, BufferMode.MacroChop),
    CaptureMode(TraceMode.Analog,         15,    40, 16383, 1, 1, False, BufferMode.Single),
    CaptureMode(TraceMode.AnalogChop,     13,    40, 16383, 2, 1, False, BufferMode.Chop),
    CaptureMode(TraceMode.AnalogFast,      8,    14,     1, 1, 1, False, BufferMode.Single),
    CaptureMode(TraceMode.AnalogFastChop,  8,    40,     1, 2, 1, False, BufferMode.Chop),
    CaptureMode(TraceMode.AnalogShot,      2,     5,     1, 1, 1, False, BufferMode.Single),
    CaptureMode(TraceMode.AnalogShotChop,  4,     5,     1, 2, 1, False, BufferMode.Chop),
    CaptureMode(TraceMode.Logic,           5, 16384,     1, 0, 1, True,  BufferMode.Single),
    CaptureMode(TraceMode.LogicFast,       4,     4,     1, 0, 1, True,  BufferMode.Single),
    CaptureMode(TraceMode.LogicShot,       1,     3,     1, 0, 1, True,  BufferMode.Single),
    CaptureMode(TraceMode.Mixed,          15,    40, 16383, 1, 1, True,  BufferMode.Dual),
    CaptureMode(TraceMode.MixedChop,      13,    40, 16383, 2, 1, True,  BufferMode.ChopDual),
    CaptureMode(TraceMode.MixedFast,       8,    14,     1, 1, 1, True,  BufferMode.Dual),
    CaptureMode(TraceMode.MixedFastChop,   8,    40,     1, 2, 1, True,  BufferMode.ChopDual),
    CaptureMode(TraceMode.MixedShot,       2,     5,     1, 1, 1, True,  BufferMode.Dual),
    CaptureMode(TraceMode.MixedShotChop,   4,     5,     1, 2, 1, True,  BufferMode.ChopDual),
]


class VirtualMachine:

    class Transaction:
        def __init__(self, vm):
            self._vm = vm
            self._data = b''

        def append(self, cmd):
            self._data += cmd

        async def __aenter__(self):
            self._vm._transactions.append(self)
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            if self._vm._transactions.pop() != self:
                raise RuntimeError("Mis-ordered transactions")
            if exc_type is None:
                await self._vm.issue(self._data)
            return False

    def __init__(self, reader=None, writer=None):
        self._reader = reader
        self._writer = writer
        self._transactions = []
        self._reply_buffer = b''
        self.serial_number = ""

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._reader = None
            return True
        return False

    __del__ = close

    def transaction(self):
        return self.Transaction(self)

    async def issue(self, cmd):
        if isinstance(cmd, str):
            cmd = cmd.encode('ascii')
        if not self._transactions:
            Log.debug(f"Issue: {cmd!r}")
            self._writer.write(cmd)
            await self._writer.drain()
            echo = await self._reader.readexactly(len(cmd))
            if echo != cmd:
                raise RuntimeError("Mismatched response")
        else:
            self._transactions[-1].append(cmd)

    async def read_replies(self, nreplies):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        replies = []
        data, self._reply_buffer = self._reply_buffer, b''
        while len(replies) < nreplies:
            index = data.find(b'\r')
            if index >= 0:
                reply = data[:index]
                Log.debug(f"Read reply: {reply!r}")
                replies.append(reply)
                data = data[index+1:]
            else:
                data += await self._reader.read(100)
        if data:
            self._reply_buffer = data
        return replies

    async def issue_reset(self):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        Log.debug("Issue reset")
        self._writer.write(b'!')
        await self._writer.drain()
        while not (await self._reader.read(1000)).endswith(b'!'):
            pass
        self._reply_buffer = b''
        Log.debug("Reset complete")

    async def set_registers(self, **kwargs):
        cmd = ''
        register0 = register1 = None
        for base, name in sorted((Registers[name].base, name) for name in kwargs):
            register = Registers[name]
            data = register.encode(kwargs[name])
            Log.debug(f"{name} = 0x{''.join(f'{b:02x}' for b in reversed(data))}")
            for i, byte in enumerate(data):
                if cmd:
                    cmd += 'z'
                    register1 += 1
                address = base + i
                if register1 is None or address > register1 + 3:
                    cmd += f'{address:02x}@'
                    register0 = register1 = address
                else:
                    cmd += 'n' * (address - register1)
                    register1 = address
                if byte != register0:
                    cmd += '[' if byte == 0 else f'{byte:02x}'
                    register0 = byte
        if cmd:
            await self.issue(cmd + 's')

    async def get_register(self, name):
        register = Registers[name]
        await self.issue(f'{register.base:02x}@p')
        values = []
        width = register.width
        for i in range(width):
            values.append(int((await self.read_replies(2))[1], 16))
            if i < width-1:
                await self.issue(b'np')
        return register.decode(bytes(values))

    async def issue_get_revision(self):
        await self.issue(b'?')

    async def issue_capture_spock_registers(self):
        await self.issue(b'<')

    async def issue_get_serial_number(self):
        """
        Get serial number stocked in bitscope EEPROM
        read from (R17) register, one byte by one byte
        """
        cmds = [
            b'[11]@[30]sr',
            b'[11]@[31]sr',
            b'[11]@[32]sr',
            b'[11]@[33]sr',
            b'[11]@[34]sr',
            b'[11]@[35]sr',
            b'[11]@[36]sr',
            b'[11]@[37]sr',
        ]
        serial_number = ""

        for cmd in cmds:
            await self.issue(cmd)
            result = await self.read_replies(2)
            serial_number += bytes([int(result[1], 16)]).decode("ascii")

        return serial_number

    async def issue_program_spock_registers(self):
        await self.issue(b'>')

    async def issue_configure_device_hardware(self):
        await self.issue(b'U')

    async def issue_streaming_trace(self):
        await self.issue(b'T')

    async def issue_triggered_trace(self):
        await self.issue(b'D')

    async def read_analog_samples(self, nsamples, sample_width):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        if sample_width == 2:
            data = await self._reader.readexactly(2 * nsamples)
            return array.array('f', ((value+32768)/65536 for (value,) in struct.iter_unpack('>h', data)))
        if sample_width == 1:
            data = await self._reader.readexactly(nsamples)
            return array.array('f', (value/256 for value in data))
        raise ValueError(f"Bad sample width: {sample_width}")

    async def read_logic_samples(self, nsamples):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        return await self._reader.readexactly(nsamples)

    async def issue_cancel_trace(self):
        await self.issue(b'K')

    async def issue_sample_dump_csv(self):
        await self.issue(b'S')

    async def issue_analog_dump_binary(self):
        await self.issue(b'A')

    async def issue_wavetable_read(self):
        await self.issue(b'R')

    async def wavetable_read_bytes(self, nbytes):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        return await self._reader.readexactly(nbytes)

    async def wavetable_write_bytes(self, data):
        cmd = ''
        last_byte = None
        for byte in data:
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

    async def issue_control_clock_generator(self):
        await self.issue(b'Z')

    async def issue_read_eeprom(self):
        await self.issue(b'r')

    async def issue_write_eeprom(self):
        await self.issue(b'w')
