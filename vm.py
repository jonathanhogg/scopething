
import asyncio
import numpy as np
import struct

Registers = {
    "vrTriggerLogic": (0x05, 'U8', "Trigger Logic, one bit per channel (0 => Low, 1 => High)"),
    "vrTriggerMask": (0x06, 'U8', "Trigger Mask, one bit per channel (0 => Don’t Care, 1 => Active)"),
    "vrSpockOption": (0x07, 'U8', "Spock Option Register (see bit definition table for details)"),
    "vrSampleAddress": (0x08, 'U24', "Sample address (write) 24 bit"),
    "vrSampleCounter": (0x0b, 'U24', "Sample address (read) 24 bit"),
    "vrTriggerIntro": (0x32, 'U24', "Edge trigger intro filter counter (samples/2)"),
    "vrTriggerOutro": (0x34, 'U16', "Edge trigger outro filter counter (samples/2)"),
    "vrTriggerValue": (0x44, 'S16', "Digital (comparator) trigger (signed)"),
    "vrTriggerTime": (0x40, 'U32', "Stopwatch trigger time (ticks)"),
    "vrClockTicks": (0x2e, 'U16', "Master Sample (clock) period (ticks)"),
    "vrClockScale": (0x14, 'U16', "Clock divide by N (low byte)"),
    "vrTraceOption": (0x20, 'U8', "Trace Mode Option bits"),
    "vrTraceMode": (0x21, 'U8', "Trace Mode (see Trace Mode Table)"),
    "vrTraceIntro": (0x26, 'U16', "Pre-trigger capture count (samples)"),
    "vrTraceDelay": (0x22, 'U32', "Delay period (uS)"),
    "vrTraceOutro": (0x2a, 'U16', "Post-trigger capture count (samples)"),
    "vrTimeout": (0x2c, 'U16', "Auto trace timeout (auto-ticks)"),
    "vrPrelude": (0x3a, 'U16', "Buffer prefill value"),
    "vrBufferMode": (0x31, 'U8', "Buffer mode"),
    "vrDumpMode": (0x1e, 'U8', "Dump mode"),
    "vrDumpChan": (0x30, 'U8', "Dump (buffer) Channel (0..127,128..254,255)"),
    "vrDumpSend": (0x18, 'U16', "Dump send (samples)"),
    "vrDumpSkip": (0x1a, 'U16', "Dump skip (samples)"),
    "vrDumpCount": (0x1c, 'U16', "Dump size (samples)"),
    "vrDumpRepeat": (0x16, 'U16', "Dump repeat (iterations)"),
    "vrStreamIdent": (0x36, 'U8', "Stream data token"),
    "vrStampIdent": (0x3c, 'U8', "Timestamp token"),
    "vrAnalogEnable": (0x37, 'U8', "Analog channel enable (bitmap)"),
    "vrDigitalEnable": (0x38, 'U8', "Digital channel enable (bitmap)"),
    "vrSnoopEnable": (0x39, 'U8', "Frequency (snoop) channel enable (bitmap)"),
    "vpCmd": (0x46, 'U8', "Command Vector"),
    "vpMode": (0x47, 'U8', "Operation Mode (per command)"),
    "vpOption": (0x48, 'U16', "Command Option (bits fields per command)"),
    "vpSize": (0x4a, 'U16', "Operation (unit/block) size"),
    "vpIndex": (0x4c, 'U16', "Operation index (eg, P Memory Page)"),
    "vpAddress": (0x4e, 'U16', "General purpose address"),
    "vpClock": (0x50, 'U16', "Sample (clock) period (ticks)"),
    "vpModulo": (0x52, 'U16', "Modulo Size (generic)"),
    "vpLevel": (0x54, 'U0.16', "Output (analog) attenuation (unsigned)"),
    "vpOffset": (0x56, 'S1.15', "Output (analog) offset (signed)"),
    "vpMask": (0x58, 'U16', "Translate source modulo mask"),
    "vpRatio": (0x5a, 'U16.16', "Translate command ratio (phase step)"),
    "vpMark": (0x5e, 'U16', "Mark count/phase (ticks/step)"),
    "vpSpace": (0x60, 'U16', "Space count/phase (ticks/step)"),
    "vpRise": (0x82, 'U16', "Rising edge clock (channel 1) phase (ticks)"),
    "vpFall": (0x84, 'U16', "Falling edge clock (channel 1) phase (ticks)"),
    "vpControl": (0x86, 'U8', "Clock Control Register (channel 1)"),
    "vpRise2": (0x88, 'U16', "Rising edge clock (channel 2) phase (ticks)"),
    "vpFall2": (0x8a, 'U16', "Falling edge clock (channel 2) phase (ticks)"),
    "vpControl2": (0x8c, 'U8', "Clock Control Register (channel 2)"),
    "vpRise3": (0x8e, 'U16', "Rising edge clock (channel 3) phase (ticks)"),
    "vpFall3": (0x90, 'U16', "Falling edge clock (channel 3) phase (ticks)"),
    "vpControl3": (0x92, 'U8', "Clock Control Register (channel 3)"),
    "vrEepromData": (0x10, 'U8', "EE Data Register"),
    "vrEepromAddress": (0x11, 'U8', "EE Address Register"),
    "vrConverterLo": (0x64, 'U16', "VRB ADC Range Bottom (D Trace Mode)"),
    "vrConverterHi": (0x66, 'U16', "VRB ADC Range Top (D Trace Mode)"),
    "vrTriggerLevel": (0x68, 'U16', "Trigger Level (comparator, unsigned)"),
    "vrLogicControl": (0x74, 'U8', "Logic Control"),
    "vrRest": (0x78, 'U16', "DAC (rest) level"),
    "vrKitchenSinkA": (0x7b, 'U8', "Kitchen Sink Register A"),
    "vrKitchenSinkB": (0x7c, 'U8', "Kitchen Sink Register B"),
    "vpMap0": (0x94, 'U8', "Peripheral Pin Select Channel 0"),
    "vpMap1": (0x95, 'U8', "Peripheral Pin Select Channel 1"),
    "vpMap2": (0x96, 'U8', "Peripheral Pin Select Channel 2"),
    "vpMap3": (0x97, 'U8', "Peripheral Pin Select Channel 3"),
    "vpMap4": (0x98, 'U8', "Peripheral Pin Select Channel 4"),
    "vpMap5": (0x99, 'U8', "Peripheral Pin Select Channel 5"),
    "vpMap6": (0x9a, 'U8', "Peripheral Pin Select Channel 6"),
    "vpMap7": (0x9b, 'U8', "Peripheral Pin Select Channel 7"),
    "vrMasterClockN": (0xf7, 'U8', "PLL prescale (DIV N)"),
    "vrMasterClockM": (0xf8, 'U16', "PLL multiplier (MUL M)"),
    "vrLedLevelRED": (0xfa, 'U8', "Red LED Intensity (VM10 only)"),
    "vrLedLevelGRN": (0xfb, 'U8', "Green LED Intensity (VM10 only)"),
    "vrLedLevelYEL": (0xfc, 'U8', "Yellow LED Intensity (VM10 only)"),
    "vcBaudHost": (0xfe, 'U16', "baud rate (host side)"),
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


class BufferModes:
    Single    = 0
    Chop      = 1
    Dual      = 2
    ChopDual  = 3
    Macro     = 4
    MacroChop = 5


class SpockOptions:
    TriggerInvert        = 0x40
    TriggerSourceA       = 0x00
    TriggerSourceB       = 0x04
    TriggerSwap          = 0x02
    TriggerSampledAnalog = 0x00
    TriggerHardware      = 0x01


class KitchenSinkA:
    ChannelAComparatorEnable = 0x80
    ChannelBComparatorEnable = 0x40

class KitchenSinkB:
    AnalogFilterEnable       = 0x80
    WaveformGeneratorEnable  = 0x40


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
            await self._stream.readuntil(cmd)
        else:
            self._transactions[-1].append(cmd)

    async def read_reply(self):
        return (await self.read_replies(1))[0]

    async def read_replies(self, n):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        await self._stream.readuntil(b'\r')
        replies = []
        for i in range(n):
            replies.append((await self._stream.readuntil(b'\r'))[:-1])
        return replies

    async def reset(self):
        if self._transactions:
            raise TypeError("Command transaction in progress")
        await self._stream.write(b'!')
        await self._stream.readuntil(b'!')

    def encode(self, value, dtype):
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

    def decode(self, bs, dtype):
        if dtype == 'int':
            return struct.unpack('<i', bs)[0]
        elif dtype == 'uint':
            return struct.unpack('<I', bs)[0]
        elif dtype.startswith('fixed.'):
            fraction_bits = int(dtype[6:])
            return struct.unpack('<i', bs)[0] / (1<<fraction_bits)
        elif dtype.startswith('ufixed.'):
            fraction_bits = int(dtype[7:])
            return struct.unpack('<I', bs)[0] / (1<<fraction_bits)
        else:
            raise TypeError("Unrecognised type")

    def calculate_width(self, dtype):
        if '.' in dtype:
            return sum(map(int, dtype[1:].split('.', 1))) // 8
        else:
            return int(dtype[1:]) // 8

    async def set_registers(self, **kwargs):
        cmd = ''
        r0 = r1 = None
        for base, name in sorted((Registers[name][0], name) for name in kwargs):
            base, dtype, desc = Registers[name]
            for i, byte in enumerate(self.encode(kwargs[name], dtype)):
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
        width = self.calculate_width(dtype)
        for i in range(width):
            values.append(int(await self.read_reply(), 16))
            if i < width-1:
                await self.issue(b'np')
        return self.decode(bytes(values), dtype)

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
        await self.issue(b'S')

    async def issue_wavetable_read(self):
        await self.issue(b'R')

    async def wavetable_write(self, byte):
        await self.issue('{:02x}W'.format(byte))

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


