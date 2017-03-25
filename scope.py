#!/usr/bin/env python3

import argparse
import asyncio
import logging
import math
import os
import struct

import streams
import vm


Log = logging.getLogger('scope')


class Scope(vm.VirtualMachine):

    PARAMS_MAGIC = 0xb0b2

    @classmethod
    async def connect(cls, device=None):
        if device is None:
            reader = writer = streams.SerialStream.stream_matching(0x0403, 0x6001)
        elif os.path.exists(device):
            reader = writer = streams.SerialStream(device=device)
        elif ':' in device:
            host, port = device.split(':', 1)
            Log.info("Connecting to remote scope at {}:{}".format(host, port))
            reader, writer = await asyncio.open_connection(host, int(port))
        else:
            raise ValueError("Don't know what to do with '{}'".format(device))
        scope = cls(reader, writer)
        await scope.setup()
        return scope

    async def setup(self):
        Log.info("Resetting scope")
        await self.reset()
        await self.issue_get_revision()
        revision = ((await self.read_replies(2))[1]).decode('ascii')
        if revision == 'BS000501':
            self.awg_clock_period = 25e-9
            self.awg_wavetable_size = 1024
            self.awg_sample_buffer_size = 1024
            self.awg_minimum_clock = 33
            self.awg_maximum_voltage = 3.3
            self.analog_params = (18.584, -3.5073, 298.11, 18.253, 0.40815)
            self.analog_offsets = {'A': -0.011785, 'B': 0.011785}
            self.analog_min = -5.7
            self.analog_max = 8
            self.capture_clock_period = 25e-9
            self.capture_buffer_size = 12*1024
            self.trigger_timeout_tick = 6.4e-6
            self.trigger_low = -7.517
            self.trigger_high = 10.816
        # await self.load_params()  XXX switch this off until I understand EEPROM better
        self._generator_running = False
        Log.info("Initialised scope, revision: {}".format(revision))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._reader = None

    __del__ = close

    async def load_params(self):
        params = []
        for i in range(struct.calcsize('<H8fH')):
            params.append(await self.read_eeprom(i+70))
        params = struct.unpack('<H8fH', bytes(params))
        if params[0] == self.PARAMS_MAGIC and params[-1] == self.PARAMS_MAGIC:
            self.analog_params = tuple(params[1:7])
            self.analog_offsets['A'] = params[8]
            self.analog_offsets['B'] = params[9]

    async def save_params(self):
        params = struct.pack('<H8fH', self.PARAMS_MAGIC, *self.analog_params,
                             self.analog_offsets['A'], self.analog_offsets['B'], self.PARAMS_MAGIC)
        for i, byte in enumerate(params):
            await self.write_eeprom(i+70, byte)

    def calculate_lo_hi(self, low, high, params=None):
        if params is None:
            params = self.analog_params
        d, f, b, scale, offset = params
        l = low / scale + offset
        h = high / scale + offset
        al = d + f * (2*l - 1)**2
        ah = d + f * (2*h - 1)**2
        dl = (l*(2*al + b) - al*h) / b
        dh = (h*(2*ah + b) - ah*(l + 1)) / b
        return dl, dh

    async def capture(self, channels=['A'], trigger_channel=None, trigger_level=0, trigger_type='rising', hair_trigger=False,
                      period=1e-3, nsamples=1000, timeout=None, low=None, high=None, raw=False):
        if 'A' in channels and 'B' in channels:
            nsamples_multiplier = 2
            dual = True
        else:
            nsamples_multiplier = 1
            dual = False
        ticks = int(period / nsamples / nsamples_multiplier / self.capture_clock_period)
        for clock_mode in vm.ClockModes:
            if clock_mode.dual == dual and ticks in range(clock_mode.clock_low, clock_mode.clock_high + 1):
                break
        else:
            raise RuntimeError("Unsupported clock period: {}".format(ticks))
        if clock_mode.clock_max is not None and ticks > clock_mode.clock_max:
            ticks = clock_mode.clock_max
        nsamples = int(round(period / ticks / nsamples_multiplier / self.capture_clock_period))
        total_samples = nsamples * nsamples_multiplier
        buffer_width = self.capture_buffer_size // clock_mode.sample_width
        assert total_samples <= buffer_width
        
        if raw:
            lo, hi = low, high
        else:
            if low is None:
                low = self.analog_min
            if high is None:
                high = self.analog_max
            lo, hi = self.calculate_lo_hi(low, high)

        if trigger_channel is None:
            trigger_channel = channels[0]
        else:
            assert trigger_channel in channels
        spock_option = vm.SpockOption.TriggerTypeHardwareComparator
        if trigger_channel == 'A':
            kitchen_sink_a = vm.KitchenSinkA.ChannelAComparatorEnable
            spock_option |= vm.SpockOption.TriggerSourceA
        elif trigger_channel == 'B':
            kitchen_sink_a = vm.KitchenSinkA.ChannelBComparatorEnable
            spock_option |= vm.SpockOption.TriggerSourceB
        kitchen_sink_b = vm.KitchenSinkB.AnalogFilterEnable
        if self._generator_running:
            kitchen_sink_b |= vm.KitchenSinkB.WaveformGeneratorEnable
        if trigger_type.lower() in {'falling', 'below'}:
            spock_option |= vm.SpockOption.TriggerInvert
        trigger_intro = 0 if trigger_type.lower() in {'above', 'below'} else (1 if hair_trigger else 4)
        if not raw:
            trigger_level = (trigger_level - self.trigger_low) / (self.trigger_high - self.trigger_low)
        analog_enable = 0
        if 'A' in channels:
            analog_enable |= 1
        if 'B' in channels:
            analog_enable |= 2

        async with self.transaction():
            await self.set_registers(TraceMode=clock_mode.TraceMode, BufferMode=clock_mode.BufferMode,
                                     SampleAddress=0, ClockTicks=ticks, ClockScale=1,
                                     TraceIntro=total_samples//2, TraceOutro=total_samples//2, TraceDelay=0,
                                     Timeout=int(round((period*5 if timeout is None else timeout) / self.trigger_timeout_tick)),
                                     TriggerMask=0x7f, TriggerLogic=0x80, TriggerLevel=trigger_level, SpockOption=spock_option,
                                     TriggerIntro=trigger_intro, TriggerOutro=2 if hair_trigger else 4, Prelude=0,
                                     ConverterLo=lo, ConverterHi=hi,
                                     KitchenSinkA=kitchen_sink_a, KitchenSinkB=kitchen_sink_b, AnalogEnable=analog_enable)
            await self.issue_program_spock_registers()
            await self.issue_configure_device_hardware()
            await self.issue_triggered_trace()
        while True:
            code, timestamp = (int(x, 16) for x in await self.read_replies(2))
            if code != 2:
                break
        address = int((await self.read_replies(1))[0], 16) // nsamples_multiplier
        traces = {'t': [t*nsamples_multiplier*self.capture_clock_period for t in range(timestamp-nsamples*ticks, timestamp, ticks)]}
        for dump_channel, channel in enumerate(sorted(channels)):
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - nsamples) * nsamples_multiplier % buffer_width,
                                         DumpMode=vm.DumpMode.Native if clock_mode.sample_width == 2 else vm.DumpMode.Raw, 
                                         DumpChan=dump_channel, DumpCount=nsamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            data = await self._reader.readexactly(nsamples * clock_mode.sample_width)
            if clock_mode.sample_width == 2:
                data = struct.unpack('>{}h'.format(nsamples), data)
                if raw:
                    trace = [(value / 65536 + 0.5) for value in data]
                else:
                    trace = [(value / 65536 + 0.5) * (high - low) + low + self.analog_offsets[channel] for value in data]
            else:
                if raw:
                    trace = [value / 256 for value in data]
                else:
                    trace = [value / 256 * (high - low) + low + self.analog_offsets[channel] for value in data]
            traces[channel] = trace
        return traces

    async def start_generator(self, frequency, waveform='sine', wavetable=None, ratio=0.5, vpp=None, offset=0,
                              min_samples=50, max_error=1e-4):
        if vpp is None:
            vpp = self.awg_maximum_voltage
        possible_params = []
        max_clock = int(math.floor(1 / frequency / min_samples / self.awg_clock_period))
        for clock in range(self.awg_minimum_clock, max_clock+1):
            width = 1 / frequency / (clock * self.awg_clock_period)
            if width <= self.awg_sample_buffer_size:
                nwaves = int(self.awg_sample_buffer_size / width)
                size = int(round(nwaves * width))
                width = size / nwaves
                actualf = 1 / (width * clock * self.awg_clock_period)
                error = abs(frequency - actualf) / frequency
                if error < max_error:
                    possible_params.append(((error == 0, width), (size, nwaves, clock, actualf)))
        if not possible_params:
            raise ValueError("No solution to required frequency/min_samples/max_error")
        size, nwaves, clock, actualf = sorted(possible_params)[-1][1]
        async with self.transaction():
            if wavetable is None:
                mode = {'sine': 0, 'triangle': 1, 'sawtooth': 1, 'exponential': 2, 'square': 3}[waveform.lower()]
                await self.set_registers(Cmd=0, Mode=mode, Ratio=ratio)
                await self.issue_synthesize_wavetable()
            else:
                if len(wavetable) != self.awg_wavetable_size:
                    raise ValueError("Wavetable data must be {} samples".format(self.awg_wavetable_size))
                await self.set_registers(Cmd=0, Mode=1, Address=0, Size=1)
                await self.wavetable_write_bytes(wavetable)
            await self.set_registers(Cmd=0, Mode=0, Level=vpp / self.awg_maximum_voltage,
                                     Offset=offset / self.awg_maximum_voltage,
                                     Ratio=nwaves * self.awg_wavetable_size / size,
                                     Index=0, Address=0, Size=size)
            await self.issue_translate_wavetable()
            await self.set_registers(Cmd=2, Mode=0, Clock=clock, Modulo=size,
                                     Mark=10, Space=1, Rest=0x7f00, Option=0x8004)
            await self.issue_control_waveform_generator()
            await self.set_registers(KitchenSinkB=vm.KitchenSinkB.WaveformGeneratorEnable)
            await self.issue_configure_device_hardware()
            await self.issue('.')
        self._generator_running = True
        return actualf

    async def stop_generator(self):
        async with self.transaction():
            await self.set_registers(Cmd=1, Mode=0)
            await self.issue_control_waveform_generator()
            await self.set_registers(KitchenSinkB=0)
            await self.issue_configure_device_hardware()
        self._generator_running = False

    async def read_wavetable(self):
        with self.transaction():
            self.set_registers(Address=0, Size=self.awg_wavetable_size)
            self.issue_wavetable_read()
        return list(self.read_exactly(self.awg_wavetable_size))

    async def read_eeprom(self, address):
        async with self.transaction():
            await self.set_registers(EepromAddress=address)
            await self.issue_read_eeprom()
        return int((await self.read_replies(2))[1], 16)

    async def write_eeprom(self, address, byte):
        async with self.transaction():
            await self.set_registers(EepromAddress=address, EepromData=byte)
            await self.issue_write_eeprom()
        if int((await self.read_replies(2))[1], 16) != byte:
            raise RuntimeError("Error writing EEPROM byte")

    async def calibrate(self, n=33):
        import numpy as np
        from scipy.optimize import leastsq, least_squares
        items = []
        await self.start_generator(1000, waveform='square')
        for low in np.linspace(0.063, 0.4, n):
            for high in np.linspace(0.877, 0.6, n):
                data = await self.capture(channels='AB', period=2e-3, trigger_level=0.5, nsamples=1000, low=low, high=high, raw=True)
                A = np.array(data['A'])
                A.sort()
                B = np.array(data['B'])
                B.sort()
                Azero, A3v3 = A[10:490].mean(), A[510:990].mean()
                Bzero, B3v3 = B[10:490].mean(), B[510:990].mean()
                zero = (Azero + Bzero) / 2
                analog_range = 3.3 / ((A3v3 + B3v3)/2 - zero)
                analog_low = -zero * analog_range
                analog_high = analog_low + analog_range
                offset = (Azero - Bzero) / 2 * analog_range
                items.append((analog_low, analog_high, low, high, offset))
        await self.stop_generator()
        items = np.array(items)
        def f(params, analog_low, analog_high, low, high):
            lo, hi = self.calculate_lo_hi(analog_low, analog_high, params)
            return np.sqrt((low - lo) ** 2 + (high - hi) ** 2)
        result = least_squares(f, self.analog_params, args=items.T[:4], bounds=([0, -np.inf, 250, 0, 0], [np.inf, np.inf, 350, np.inf, np.inf]))
        if result.success in range(1, 5):
            self.analog_params = tuple(result.x)
            offset = items[:, 4].mean()
            self.analog_offsets = {'A': -offset, 'B': +offset}
        else:
            Log.warning("Calibration failed: {}".format(result.message))
            print(result.message)
        return result.success


import numpy as np

async def main():
    global s, x, y, data
    parser = argparse.ArgumentParser(description="scopething")
    parser.add_argument('device', nargs='?', default=None, type=str, help="Device to connect to")
    args = parser.parse_args()
    s = await Scope.connect(args.device)
    x = np.linspace(0, 2*np.pi, s.awg_wavetable_size, endpoint=False)
    y = np.round((np.sin(x)**5)*127 + 128, 0).astype('uint8')
    await s.start_generator(1000, wavetable=y)
    #if await s.calibrate():
    #    await s.save_params()

def capture(*args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(s.capture(*args, **kwargs))

def calibrate(*args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(s.calibrate(*args, **kwargs))

def generate(*args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(s.start_generator(*args, **kwargs))

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.get_event_loop().run_until_complete(main())

