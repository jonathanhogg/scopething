
import asyncio
import numpy as np
import struct


Registers = {
    "TriggerLogic": (0x05, 'U8', "Trigger Logic, one bit per channel (0 => Low, 1 => High)"),
    "TriggerMask": (0x06, 'U8', "Trigger Mask, one bit per channel (0 => Don’t Care, 1 => Active)"),
    "SpockOption": (0x07, 'U8', "Spock Option Register (see bit definition table for details)"),
    "SampleAddress": (0x08, 'U24', "Sample address (write) 24 bit"),
    "SampleCounter": (0x0b, 'U24', "Sample address (read) 24 bit"),
    "TriggerIntro": (0x32, 'U24', "Edge trigger intro filter counter (samples/2)"),
    "TriggerOutro": (0x34, 'U16', "Edge trigger outro filter counter (samples/2)"),
    "TriggerValue": (0x44, 'S16', "Digital (comparator) trigger (signed)"),
    "TriggerTime": (0x40, 'U32', "Stopwatch trigger time (ticks)"),
    "ClockTicks": (0x2e, 'U16', "Master Sample (clock) period (ticks)"),
    "ClockScale": (0x14, 'U16', "Clock divide by N (low byte)"),
    "TraceOption": (0x20, 'U8', "Trace Mode Option bits"),
    "TraceMode": (0x21, 'U8', "Trace Mode (see Trace Mode Table)"),
    "TraceIntro": (0x26, 'U16', "Pre-trigger capture count (samples)"),
    "TraceDelay": (0x22, 'U32', "Delay period (uS)"),
    "TraceOutro": (0x2a, 'U16', "Post-trigger capture count (samples)"),
    "Timeout": (0x2c, 'U16', "Auto trace timeout (auto-ticks)"),
    "Prelude": (0x3a, 'U16', "Buffer prefill value"),
    "BufferMode": (0x31, 'U8', "Buffer mode"),
    "DumpMode": (0x1e, 'U8', "Dump mode"),
    "DumpChan": (0x30, 'U8', "Dump (buffer) Channel (0..127,128..254,255)"),
    "DumpSend": (0x18, 'U16', "Dump send (samples)"),
    "DumpSkip": (0x1a, 'U16', "Dump skip (samples)"),
    "DumpCount": (0x1c, 'U16', "Dump size (samples)"),
    "DumpRepeat": (0x16, 'U16', "Dump repeat (iterations)"),
    "StreamIdent": (0x36, 'U8', "Stream data token"),
    "StampIdent": (0x3c, 'U8', "Timestamp token"),
    "AnalogEnable": (0x37, 'U8', "Analog channel enable (bitmap)"),
    "DigitalEnable": (0x38, 'U8', "Digital channel enable (bitmap)"),
    "SnoopEnable": (0x39, 'U8', "Frequency (snoop) channel enable (bitmap)"),
    "Cmd": (0x46, 'U8', "Command Vector"),
    "Mode": (0x47, 'U8', "Operation Mode (per command)"),
    "Option": (0x48, 'U16', "Command Option (bits fields per command)"),
    "Size": (0x4a, 'U16', "Operation (unit/block) size"),
    "Index": (0x4c, 'U16', "Operation index (eg, P Memory Page)"),
    "Address": (0x4e, 'U16', "General purpose address"),
    "Clock": (0x50, 'U16', "Sample (clock) period (ticks)"),
    "Modulo": (0x52, 'U16', "Modulo Size (generic)"),
    "Level": (0x54, 'U0.16', "Output (analog) attenuation (unsigned)"),
    "Offset": (0x56, 'S1.15', "Output (analog) offset (signed)"),
    "Mask": (0x58, 'U16', "Translate source modulo mask"),
    "Ratio": (0x5a, 'U16.16', "Translate command ratio (phase step)"),
    "Mark": (0x5e, 'U16', "Mark count/phase (ticks/step)"),
    "Space": (0x60, 'U16', "Space count/phase (ticks/step)"),
    "Rise": (0x82, 'U16', "Rising edge clock (channel 1) phase (ticks)"),
    "Fall": (0x84, 'U16', "Falling edge clock (channel 1) phase (ticks)"),
    "Control": (0x86, 'U8', "Clock Control Register (channel 1)"),
    "Rise2": (0x88, 'U16', "Rising edge clock (channel 2) phase (ticks)"),
    "Fall2": (0x8a, 'U16', "Falling edge clock (channel 2) phase (ticks)"),
    "Control2": (0x8c, 'U8', "Clock Control Register (channel 2)"),
    "Rise3": (0x8e, 'U16', "Rising edge clock (channel 3) phase (ticks)"),
    "Fall3": (0x90, 'U16', "Falling edge clock (channel 3) phase (ticks)"),
    "Control3": (0x92, 'U8', "Clock Control Register (channel 3)"),
    "EepromData": (0x10, 'U8', "EE Data Register"),
    "EepromAddress": (0x11, 'U8', "EE Address Register"),
    "ConverterLo": (0x64, 'U0.16', "VRB ADC Range Bottom (D Trace Mode)"),
    "ConverterHi": (0x66, 'U0.16', "VRB ADC Range Top (D Trace Mode)"),
    "TriggerLevel": (0x68, 'U16', "Trigger Level (comparator, unsigned)"),
    "LogicControl": (0x74, 'U8', "Logic Control"),
    "Rest": (0x78, 'U16', "DAC (rest) level"),
    "KitchenSinkA": (0x7b, 'U8', "Kitchen Sink Register A"),
    "KitchenSinkB": (0x7c, 'U8', "Kitchen Sink Register B"),
    "Map0": (0x94, 'U8', "Peripheral Pin Select Channel 0"),
    "Map1": (0x95, 'U8', "Peripheral Pin Select Channel 1"),
    "Map2": (0x96, 'U8', "Peripheral Pin Select Channel 2"),
    "Map3": (0x97, 'U8', "Peripheral Pin Select Channel 3"),
    "Map4": (0x98, 'U8', "Peripheral Pin Select Channel 4"),
    "Map5": (0x99, 'U8', "Peripheral Pin Select Channel 5"),
    "Map6": (0x9a, 'U8', "Peripheral Pin Select Channel 6"),
    "Map7": (0x9b, 'U8', "Peripheral Pin Select Channel 7"),
    "MasterClockN": (0xf7, 'U8', "PLL prescale (DIV N)"),
    "MasterClockM": (0xf8, 'U16', "PLL multiplier (MUL M)"),
}


class TraceMode:
    Analog         =  0
    AnalogFast     =  4
    AnalogShot     = 11
    Mixed          =  1
    MixedFast      =  5
    MixedShot      = 12
    Logic          = 14
    LogicFast      = 15
    LogicShot      = 13
    AnalogChop     =  2
    AnalogFastChop =  6
    AnalogShotChop = 16
    MixedChop      =  3
    MixedFastChop  =  7
    MixedShotChop  = 17
    Macro          = 18
    MacroChop      = 19


class BufferMode:
    Single    = 0
    Chop      = 1
    Dual      = 2
    ChopDual  = 3
    Macro     = 4
    MacroChop = 5

class DumpMode:
    Raw    = 0
    Burst  = 1
    Summed = 2
    MinMax = 3
    AndOr  = 4
    Native = 5
    Filter = 6
    Span   = 7

class SpockOption:
    TriggerInvert                 = 0x40
    TriggerSourceA                = 0x04 * 0
    TriggerSourceB                = 0x04 * 1
    TriggerSwap                   = 0x02
    TriggerTypeSampledAnalog      = 0x01 * 0
    TriggerTypeHardwareComparator = 0x01 * 1

class KitchenSinkA:
    ChannelAComparatorEnable = 0x80
    ChannelBComparatorEnable = 0x40

class KitchenSinkB:
    AnalogFilterEnable       = 0x80
    WaveformGeneratorEnable  = 0x40


def encode(value, dtype):
    sign = dtype[0]
    if '.' in dtype:
        whole, fraction = map(int, dtype[1:].split('.', 1))
        width = whole + fraction
        value = int(round(value * (1 << fraction)))
    else:
        width = int(dtype[1:])
    if sign == 'U':
        n = 1 << width
        value = max(0, min(value, n-1))
        bs = struct.pack('<I', value)
    elif sign == 'S':
        n = 1 << (width - 1)
        value = max(-n, min(value, n-1))
        bs = struct.pack('<i', value)
    else:
        raise TypeError("Unrecognised type")
    return bs[:width//8]

def decode(bs, dtype):
    if len(bs) < 4:
        bs = bs + bytes(4 - len(bs))
    sign = dtype[0]
    if sign == 'U':
        value = struct.unpack('<I', bs)[0]
    elif sign == 'S':
        value = struct.unpack('<i', bs)[0]
    if '.' in dtype:
        whole, fraction = map(int, dtype[1:].split('.', 1))
        value = value / (1 << fraction)
    return value

def calculate_width(dtype):
    if '.' in dtype:
        return sum(map(int, dtype[1:].split('.', 1))) // 8
    else:
        return int(dtype[1:]) // 8


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
            self._vm._transactions.pop()
            if exc_type is None:
                await self._vm.issue(self._data)
            return False

    def __init__(self, stream):
        self._stream = stream
        self._transactions = []

    def transaction(self):
        return self.Transaction(self)

    async def issue(self, cmd):
        if isinstance(cmd, str):
            cmd = cmd.encode('ascii')
        if not self._transactions:
            await self._stream.write(cmd)
            echo = await self._stream.readexactly(len(cmd))
            if echo != cmd:
                raise RuntimeError("Mismatched response")
        else:
            self._transactions[-1].append(cmd)

    async def read_replies(self, n):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        replies = []
        for i in range(n):
            replies.append((await self._stream.readuntil(b'\r'))[:-1])
        return replies

    async def reset(self):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        await self._stream.write(b'!')
        await self._stream.readuntil(b'!')

    async def set_registers(self, **kwargs):
        cmd = ''
        r0 = r1 = None
        for base, name in sorted((Registers[name][0], name) for name in kwargs):
            base, dtype, desc = Registers[name]
            for i, byte in enumerate(encode(kwargs[name], dtype)):
                if cmd:
                    cmd += 'z'
                    r1 += 1
                address = base + i
                if r1 is None or address > r1 + 3:
                    cmd += '{:02x}@'.format(address)
                    r0 = r1 = address
                else:
                    cmd += 'n' * (address - r1)
                    r1 = address
                if byte != r0:
                    cmd += '[' if byte == 0 else '{:02x}'.format(byte)
                    r0 = byte
        if cmd:
            await self.issue(cmd + 's')

    async def get_register(self, name):
        base, dtype, desc = Registers[name]
        await self.issue('{:02x}@p'.format(base))
        values = []
        width = calculate_width(dtype)
        for i in range(width):
            values.append(int(await self.read_reply(), 16))
            if i < width-1:
                await self.issue(b'np')
        return decode(bytes(values), dtype)

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

    async def issue_cancel_trace(self):
        await self.issue(b'K')

    async def issue_sample_dump_csv(self):
        await self.issue(b'S')

    async def issue_analog_dump_binary(self):
        await self.issue(b'A')

    async def issue_wavetable_read(self):
        await self.issue(b'R')

    async def wavetable_write_bytes(self, bs):
        cmd = ''
        last_byte = None
        for byte in bs:
            if byte != last_byte:
                cmd += '{:02x}'.format(byte)
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


