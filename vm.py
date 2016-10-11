

import asyncio
import numpy as np
import struct


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

    Registers = {
        "vrTriggerLogic": (1, 0x05, '''Trigger Logic, one bit per channel (0 => Low, 1 => High)''', 'uint'),
        "vrTriggerMask": (1, 0x06, '''Trigger Mask, one bit per channel (0 => Don’t Care, 1 => Active)''', 'uint'),
        "vrSpockOption": (1, 0x07, '''Spock Option Register (see bit definition table for details)''', 'uint'),
        "vrSampleAddress": (3, 0x08, '''Sample address (write) 24 bit''', 'uint'),
        "vrSampleCounter": (3, 0x0b, '''Sample address (read) 24 bit''', 'uint'),
        "vrTriggerIntro": (2, 0x32, '''Edge trigger intro filter counter (samples/2)''', 'uint'),
        "vrTriggerOutro": (2, 0x34, '''Edge trigger outro filter counter (samples/2)''', 'uint'),
        "vrTriggerValue": (2, 0x44, '''Digital (comparator) trigger (signed)''', 'int'),
        "vrTriggerTime": (4, 0x40, '''Stopwatch trigger time (ticks)''', 'uint'),
        "vrClockTicks": (2, 0x2e, '''Master Sample (clock) period (ticks)''', 'uint'),
        "vrClockScale": (2, 0x14, '''Clock divide by N (low byte)''', 'uint'),
        "vrTraceOption": (1, 0x20, '''Trace Mode Option bits''', 'uint'),
        "vrTraceMode": (1, 0x21, '''Trace Mode (see Trace Mode Table)''', 'uint'),
        "vrTraceIntro": (2, 0x26, '''Pre-trigger capture count (samples)''', 'uint'),
        "vrTraceDelay": (4, 0x22, '''Delay period (uS)''', 'uint'),
        "vrTraceOutro": (2, 0x2a, '''Post-trigger capture count (samples)''', 'uint'),
        "vrTimeout": (2, 0x2c, '''Auto trace timeout (auto-ticks)''', 'uint'),
        "vrPrelude": (2, 0x3a, '''Buffer prefill value''', 'uint'),
        "vrBufferMode": (1, 0x31, '''Buffer mode''', 'uint'),
        "vrDumpMode": (1, 0x1e, '''Dump mode''', 'uint'),
        "vrDumpChan": (1, 0x30, '''Dump (buffer) Channel (0..127,128..254,255)''', 'uint'),
        "vrDumpSend": (2, 0x18, '''Dump send (samples)''', 'uint'),
        "vrDumpSkip": (2, 0x1a, '''Dump skip (samples)''', 'uint'),
        "vrDumpCount": (2, 0x1c, '''Dump size (samples)''', 'uint'),
        "vrDumpRepeat": (2, 0x16, '''Dump repeat (iterations)''', 'uint'),
        "vrStreamIdent": (1, 0x36, '''Stream data token''', 'uint'),
        "vrStampIdent": (1, 0x3c, '''Timestamp token''', 'uint'),
        "vrAnalogEnable": (1, 0x37, '''Analog channel enable (bitmap)''', 'uint'),
        "vrDigitalEnable": (1, 0x38, '''Digital channel enable (bitmap)''', 'uint'),
        "vrSnoopEnable": (1, 0x39, '''Frequency (snoop) channel enable (bitmap)''', 'uint'),
        "vpCmd": (1, 0x46, '''Command Vector''', 'uint'),
        "vpMode": (1, 0x47, '''Operation Mode (per command)''', 'uint'),
        "vpOption": (2, 0x48, '''Command Option (bits fields per command)''', 'uint'),
        "vpSize": (2, 0x4a, '''Operation (unit/block) size''', 'uint'),
        "vpIndex": (2, 0x4c, '''Operation index (eg, P Memory Page)''', 'uint'),
        "vpAddress": (2, 0x4e, '''General purpose address''', 'uint'),
        "vpClock": (2, 0x50, '''Sample (clock) period (ticks)''', 'uint'),
        "vpModulo": (2, 0x52, '''Modulo Size (generic)''', 'uint'),
        "vpLevel": (2, 0x54, '''Output (analog) attenuation (unsigned)''', 'uint'),
        "vpOffset": (2, 0x56, '''Output (analog) offset (signed)''', 'int'),
        "vpMask": (2, 0x58, '''Translate source modulo mask''', 'uint'),
        "vpRatio": (4, 0x5a, '''Translate command ratio (phase step)''', 'uint'),
        "vpMark": (2, 0x5e, '''Mark count/phase (ticks/step)''', 'uint'),
        "vpSpace": (2, 0x60, '''Space count/phase (ticks/step)''', 'uint'),
        "vpRise": (2, 0x82, '''Rising edge clock (channel 1) phase (ticks)''', 'uint'),
        "vpFall": (2, 0x84, '''Falling edge clock (channel 1) phase (ticks)''', 'uint'),
        "vpControl": (1, 0x86, '''Clock Control Register (channel 1)''', 'uint'),
        "vpRise2": (2, 0x88, '''Rising edge clock (channel 2) phase (ticks)''', 'uint'),
        "vpFall2": (2, 0x8a, '''Falling edge clock (channel 2) phase (ticks)''', 'uint'),
        "vpControl2": (1, 0x8c, '''Clock Control Register (channel 2)''', 'uint'),
        "vpRise3": (2, 0x8e, '''Rising edge clock (channel 3) phase (ticks)''', 'uint'),
        "vpFall3": (2, 0x90, '''Falling edge clock (channel 3) phase (ticks)''', 'uint'),
        "vpControl3": (1, 0x92, '''Clock Control Register (channel 3)''', 'uint'),
        "vrEepromData": (1, 0x10, '''EE Data Register''', 'uint'),
        "vrEepromAddress": (1, 0x11, '''EE Address Register''', 'uint'),
        "vrConverterLo": (2, 0x64, '''VRB ADC Range Bottom (D Trace Mode)''', 'uint'),
        "vrConverterHi": (2, 0x66, '''VRB ADC Range Top (D Trace Mode)''', 'uint'),
        "vrTriggerLevel": (2, 0x68, '''Trigger Level (comparator, unsigned)''', 'uint'),
        "vrLogicControl": (1, 0x74, '''Logic Control''', 'uint'),
        "vrRest": (2, 0x78, '''DAC (rest) level''', 'uint'),
        "vrKitchenSinkA": (1, 0x7b, '''Kitchen Sink Register A''', 'uint'),
        "vrKitchenSinkB": (1, 0x7c, '''Kitchen Sink Register B''', 'uint'),
    }

    def __init__(self, stream):
        self._stream = stream
        self._transactions = []

    def new_transaction(self):
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
        await self._stream.readuntil(b'\r')
        replies = []
        for i in range(n):
            replies.append((await self._stream.readuntil(b'\r'))[:-1])
        return replies

    async def reset(self):
        await self._stream.write(b'!')
        await self._stream.readuntil(b'!')

    async def set_register(self, name, value):
        width, base, desc, dtype = self.Registers[name]
        bs = struct.pack('<i' if dtype == 'int' else '<I', value)
        cmd = '{:02x}@'.format(base) + 'z'.join('{:02x}'.format(bs[i]) for i in range(width)) + 's'
        await self.issue(cmd)

    async def set_registers(self, **kwargs):
        async with self.new_transaction():
            for name, value in kwargs.items():
                await self.set_register(name, value)

    async def get_register(self, name):
        width, base, desc, dtype = self.Registers[name]
        await self.issue('{:02x}@p'.format(base))
        bs = []
        for i in range(width):
            bs.append(int(await self.read_reply(), 16))
            if i < width-1:
                await self.issue(b'np')
        for i in range(4 - width):
            bs.append(0)
        value = struct.unpack('<i' if dtype == 'int' else '<I', bytes(bs))[0]
        return value

    async def get_revision(self):
        await self.issue(b'?')
        return await self.read_reply()

    async def capture_spock_registers(self):
        await self.issue(b'<')

    async def program_spock_registers(self):
        await self.issue(b'>')

    async def configure_device_hardware(self):
        await self.issue(b'U')

    async def streaming_trace(self):
        await self.issue(b'T')

    async def triggered_trace(self):
        await self.issue(b'D')

    async def cancel_trace(self):
        await self.issue(b'K')

    async def sample_dump_csv(self):
        await self.issue(b'S')

    async def analog_dump_binary(self):
        await self.issue(b'S')

    async def read_wavetable(self, size=1024, address=0):
        async with self.new_transaction():
            await self.set_registers(vpSize=size, vpAddress=address)
            await self.issue(b'R')
        return await self._stream.readexactly(size)

    async def write_wavetable(self, data, address=0):
        async with self.new_transaction():
            await self.set_registers(vpSize=1, vpAddress=address)
            for byte in data:
                await self.issue('{:02x}W'.format(byte))

    async def synthesize_wavetable(self, mode='sine', ratio=0.5):
        mode = {'sine': 0, 'sawtooth': 1, 'exponential': 2, 'square': 3}[mode.lower()]
        async with self.new_transaction():
            await self.set_registers(vpCmd=0, vpMode=mode, vpRatio=int(max(0, min(ratio, 1))*65535))
            await self.issue(b'Y')

    async def translate_wavetable(self, ratio, level=1, offset=0, size=0, index=0, address=0):
        async with self.new_transaction():
            await self.set_registers(vpCmd=0, vpMode=0, vpLevel=int(65535*level), vpOffset=int(65535*offset), vpRatio=ratio,
                                     vpSize=size, vpIndex=index, vpAddress=address)
            await self.issue(b'X')

    async def stop_waveform_generator(self):
        async with self.new_transaction():
            await self.set_registers(vpCmd=1, vpMode=0)
            await self.issue(b'Z')

    async def start_waveform_generator(self, clock, modulo, mark, space, rest, option):
        async with self.new_transaction():
            await self.set_registers(vpCmd=2, vpMode=0, vpClock=clock, vpModulo=modulo, 
                                     vpMark=mark, vpSpace=space, vrRest=rest, vpOption=option)
            await self.issue(b'Z')

    async def start_clock_generator(self):
        async with self.new_transaction():
            await self.set_registers(vpCmd=3, vpMode=0)
            await self.issue(b'Z')

    async def read_eeprom(self, address):
        async with self.new_transaction():
            await self.set_registers(vrEepromAddress=address)
            await self.issue(b'r')
        return int(await self.read_reply(), 16)

    async def write_eeprom(self, address, data):
        async with self.new_transaction():
            await self.set_registers(vrEepromAddress=address, vrEepromData=data)
            await self.issue(b'w')
        return int(await self.read_reply(), 16)


async def main():
    from streams import SerialStream
    vm = VirtualMachine(SerialStream())
    await vm.reset()
    print(await vm.get_revision())
    print(await vm.get_register('vrConverterLo'))
    print(await vm.get_register('vrTriggerLevel'))
    print(await vm.get_register('vrConverterHi'))
    await vm.set_register('vrTriggerLevel', 15000)
    print(await vm.get_register('vrTriggerLevel'))
    n = await vm.read_eeprom(0)
    print(n)
    print(await vm.write_eeprom(0, n+1))
    print(await vm.read_eeprom(0))
    async with vm.new_transaction():
        await vm.set_registers(vrKitchenSinkB=0x40)
        await vm.configure_device_hardware()
    await vm.synthesize_wavetable('sawtooth')
    #data = await vm.read_wavetable()
    #global array
    #array = np.ndarray(buffer=data, shape=(len(data),), dtype='uint8')
    #print(array)
    await vm.translate_wavetable(671088)
    await vm.start_waveform_generator(clock=40, modulo=1000, mark=10, space=1, rest=0x7f00, option=0x8004)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

