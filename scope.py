#!/usr/bin/env python3

import argparse
import asyncio
import logging
import math
import os
import struct
import sys

import streams
import vm


Log = logging.getLogger('scope')


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
            Log.info(f"Connecting to remote scope at {host}:{port}")
            reader, writer = await asyncio.open_connection(host, int(port))
        else:
            raise ValueError(f"Don't know what to do with '{device}'")
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
            self.awg_maximum_voltage = 3.33
            self.analog_params = (20.17, -5.247, 299.0, 18.47, 0.4082)
            self.analog_offsets = {'A': -0.0117, 'B': 0.0117}
            self.analog_min = -5.7
            self.analog_max = 8
            self.capture_clock_period = 25e-9
            self.capture_buffer_size = 12<<10
            self.timeout_clock_period = 6.4e-6
            self.trigger_low = -7.517
            self.trigger_high = 10.816
        # await self.load_params()  XXX switch this off until I understand EEPROM better
        self._awg_running = False
        Log.info(f"Initialised scope, revision: {revision}")

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

    async def capture(self, channels=['A'], trigger=None, trigger_level=None, trigger_type='rising', hair_trigger=False,
                      period=1e-3, nsamples=1000, timeout=None, low=None, high=None, raw=False, trigger_position=0.25):
        analog_channels = set()
        logic_channels = set()
        for channel in channels:
            if channel in {'A', 'B'}:
                analog_channels.add(channel)
                if trigger is None:
                    trigger = channel
            elif channel == 'L':
                logic_channels.update(range(8))
                if trigger is None:
                    trigger = {0: 1}
            elif channel.startswith('L'):
                i = int(channel[1:])
                logic_channels.add(i)
                if trigger is None:
                    trigger = {i: 1}
            else:
                raise ValueError(f"Unrecognised channel: {channel}")
        if self._awg_running and 4 in logic_channels:
            logic_channels.remove(4)
        if 'A' in analog_channels and 7 in logic_channels:
            logic_channels.remove(7)
        if 'B' in analog_channels and 6 in logic_channels:
            logic_channels.remove(6)
        analog_enable = 0
        if 'A' in channels:
            analog_enable |= 1
        if 'B' in channels:
            analog_enable |= 2
        logic_enable = sum(1<<channel for channel in logic_channels)

        ticks = int(period / nsamples / self.capture_clock_period)
        for capture_mode in vm.CaptureModes:
            if capture_mode.analog_channels == len(analog_channels) and capture_mode.logic_channels == bool(logic_channels):
                if ticks in range(capture_mode.clock_low, capture_mode.clock_high + 1):
                    clock_scale = 1
                    break
                elif capture_mode.clock_divide and ticks > capture_mode.clock_high:
                    for clock_scale in range(2, 1<<16):
                        test_ticks = int(period / nsamples / self.capture_clock_period / clock_scale)
                        if test_ticks in range(capture_mode.clock_low, capture_mode.clock_high + 1):
                            ticks = test_ticks
                            break
                    else:
                        continue
                    break
        else:
            raise RuntimeError("Unable to find appropriate capture mode")
        if capture_mode.clock_max is not None and ticks > capture_mode.clock_max:
            ticks = capture_mode.clock_max
        if analog_channels:
            nsamples = int(round(period / ticks / self.capture_clock_period / clock_scale / len(analog_channels))) * len(analog_channels)
        else:
            nsamples = int(round(period / ticks / self.capture_clock_period / clock_scale))
        total_samples = nsamples*2 if logic_channels and analog_channels else nsamples
        buffer_width = self.capture_buffer_size // capture_mode.sample_width
        if total_samples > buffer_width:
            raise RuntimeError("Capture buffer too small for requested capture")
        
        if raw:
            lo, hi = low, high
        else:
            if low is None:
                low = self.analog_min
            if high is None:
                high = self.analog_max
            lo, hi = self.calculate_lo_hi(low, high)

        spock_option = vm.SpockOption.TriggerTypeHardwareComparator
        kitchen_sink_a = kitchen_sink_b = 0
        if self._awg_running:
            kitchen_sink_b |= vm.KitchenSinkB.WaveformGeneratorEnable
        if trigger == 'A' or 7 in logic_channels:
            kitchen_sink_a |= vm.KitchenSinkA.ChannelAComparatorEnable
        if trigger == 'B' or 6 in logic_channels:
            kitchen_sink_a |= vm.KitchenSinkA.ChannelBComparatorEnable
        if analog_channels:
            kitchen_sink_b |= vm.KitchenSinkB.AnalogFilterEnable
        if trigger_level is None:
            trigger_level = (high + low) / 2
        trigger_level = (trigger_level - self.trigger_low) / (self.trigger_high - self.trigger_low)
        if trigger == 'A' or trigger == 'B':
            if trigger == 'A':
                spock_option |= vm.SpockOption.TriggerSourceA
                trigger_logic = 0x80
            elif trigger == 'B':
                spock_option |= vm.SpockOption.TriggerSourceB
                trigger_logic = 0x40
            trigger_mask = 0xff ^ trigger_logic
        else:
            trigger_logic = 0
            trigger_mask = 0xff
            for channel, value in trigger.items():
                mask = 1<<channel
                trigger_mask &= ~mask
                if value:
                    trigger_logic |= mask
        if trigger_type.lower() in {'falling', 'below'}:
            spock_option |= vm.SpockOption.TriggerInvert
        trigger_outro = 2 if hair_trigger else 4
        trigger_intro = 0 if trigger_type.lower() in {'above', 'below'} else trigger_outro
        trigger_samples = min(max(0, int(nsamples*trigger_position)), nsamples)
        trace_outro = max(0, nsamples-trigger_samples-trigger_outro)
        trace_intro = max(0, trigger_samples-trigger_intro)
        if timeout is None:
            trigger_timeout = int(period*5/self.timeout_clock_period)
        else:
            trigger_timeout = max(1, int(math.ceil(((trigger_outro+trace_outro)*ticks*clock_scale*self.capture_clock_period
                                                    + timeout)/self.timeout_clock_period)))

        async with self.transaction():
            await self.set_registers(TraceMode=capture_mode.TraceMode, BufferMode=capture_mode.BufferMode,
                                     SampleAddress=0, ClockTicks=ticks, ClockScale=clock_scale,
                                     TriggerLevel=trigger_level, TriggerLogic=trigger_logic, TriggerMask=trigger_mask,
                                     TraceIntro=trace_intro, TraceOutro=trace_outro, TraceDelay=0, Timeout=trigger_timeout,
                                     TriggerIntro=trigger_intro, TriggerOutro=trigger_outro, Prelude=0,
                                     SpockOption=spock_option, ConverterLo=lo, ConverterHi=hi,
                                     KitchenSinkA=kitchen_sink_a, KitchenSinkB=kitchen_sink_b, 
                                     AnalogEnable=analog_enable, DigitalEnable=logic_enable)
            await self.issue_program_spock_registers()
            await self.issue_configure_device_hardware()
            await self.issue_triggered_trace()
        while True:
            code, timestamp = (int(x, 16) for x in await self.read_replies(2))
            if code != 2:
                break
        start_timestamp = timestamp - nsamples*ticks*clock_scale
        if start_timestamp < 0:
            start_timestamp += 1<<32
            timestamp += 1<<32
        address = int((await self.read_replies(1))[0], 16)
        if capture_mode.BufferMode in {vm.BufferMode.Chop, vm.BufferMode.Dual, vm.BufferMode.MacroChop}:
            address -= address % 2
        elif capture_mode.BufferMode == vm.BufferMode.ChopDual:
            address -= address % 4
        traces = DotDict()
        for dump_channel, channel in enumerate(sorted(analog_channels)):
            asamples = nsamples // len(analog_channels)
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - total_samples) % buffer_width,
                                         DumpMode=vm.DumpMode.Native if capture_mode.sample_width == 2 else vm.DumpMode.Raw, 
                                         DumpChan=dump_channel, DumpCount=asamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            data = await self._reader.readexactly(asamples * capture_mode.sample_width)
            value_multiplier, value_offset = (1, 0) if raw else ((high-low), low+self.analog_offsets[channel])
            if capture_mode.sample_width == 2:
                data = struct.unpack(f'>{asamples}h', data)
                data = [(value/65536+0.5)*value_multiplier + value_offset for value in data]
            else:
                data = [(value/256)*value_multiplier + value_offset for value in data]
            traces[channel] = {(t+dump_channel*ticks*clock_scale)*self.capture_clock_period: value
                               for (t, value) in zip(range(start_timestamp, timestamp, ticks*clock_scale*len(analog_channels)), data)}

        if logic_channels:
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - total_samples) % buffer_width,
                                         DumpMode=vm.DumpMode.Raw, DumpChan=128, DumpCount=nsamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            data = await self._reader.readexactly(nsamples)
            ts = [t*self.capture_clock_period for t in range(start_timestamp, timestamp, ticks*clock_scale)]
            for i in logic_channels:
                mask = 1<<i
                traces[f'L{i}'] = {t: 1 if value & mask else 0 for (t, value) in zip(ts, data)}

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
                    raise ValueError(f"Wavetable data must be {self.awg_wavetable_size} samples")
                await self.set_registers(Cmd=0, Mode=1, Address=0, Size=1)
                await self.wavetable_write_bytes(wavetable)
            await self.set_registers(Cmd=0, Mode=0, Level=vpp/self.awg_maximum_voltage,
                                     Offset=offset/self.awg_maximum_voltage,
                                     Ratio=nwaves*self.awg_wavetable_size/size,
                                     Index=0, Address=0, Size=size)
            await self.issue_translate_wavetable()
            await self.set_registers(Cmd=2, Mode=0, Clock=clock, Modulo=size,
                                     Mark=10, Space=1, Rest=0x7f00, Option=0x8004)
            await self.issue_control_waveform_generator()
            await self.set_registers(KitchenSinkB=vm.KitchenSinkB.WaveformGeneratorEnable)
            await self.issue_configure_device_hardware()
            await self.issue('.')
        self._awg_running = True
        return actualf

    async def stop_generator(self):
        async with self.transaction():
            await self.set_registers(Cmd=1, Mode=0)
            await self.issue_control_waveform_generator()
            await self.set_registers(KitchenSinkB=0)
            await self.issue_configure_device_hardware()
        self._awg_running = False

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

    async def calibrate(self, n=32):
        import numpy as np
        from scipy.optimize import least_squares
        items = []
        await self.start_generator(frequency=1000, waveform='square')
        i = 0
        low_min, high_max = self.calculate_lo_hi(self.analog_min, self.analog_max)
        low_max, high_min = self.calculate_lo_hi(0, self.awg_maximum_voltage)
        for low in np.linspace(low_min, low_max*0.9, n):
            for high in np.linspace(high_min*1.1, high_max, n):
                data = await self.capture(channels=['A','B'], period=2e-3 if i%2 == 0 else 1e-3, nsamples=2000, low=low, high=high, timeout=0, raw=True)
                A = np.fromiter(data['A'].values(), dtype='float')
                A.sort()
                B = np.fromiter(data['B'].values(), dtype='float')
                B.sort()
                Azero, Amax = A[10:490].mean(), A[510:990].mean()
                Bzero, Bmax = B[10:490].mean(), B[510:990].mean()
                zero = (Azero + Bzero) / 2
                analog_range = self.awg_maximum_voltage / ((Amax + Bmax)/2 - zero)
                analog_low = -zero * analog_range
                analog_high = analog_low + analog_range
                offset = (Azero - Bzero) / 2 * analog_range
                items.append((analog_low, analog_high, low, high, offset))
                i += 1
        await self.stop_generator()
        items = np.array(items)
        def f(params, analog_low, analog_high, low, high):
            lo, hi = self.calculate_lo_hi(analog_low, analog_high, params)
            return np.sqrt((low - lo) ** 2 + (high - hi) ** 2)
        result = least_squares(f, self.analog_params, args=items.T[:4], bounds=([0, -np.inf, 250, 0, 0], [np.inf, np.inf, 350, np.inf, np.inf]))
        if result.success in range(1, 5):
            self.analog_params = tuple(result.x)
            offset = items[:,4].mean()
            self.analog_offsets = {'A': -offset, 'B': +offset}
        else:
            Log.warning(f"Calibration failed: {result.message}")
        return result.success


"""
resistance$ ipython3 --pylab
Using matplotlib backend: MacOSX

In [1]: import pandas

In [2]: run scope
INFO:scope:Resetting scope
INFO:scope:Initialised scope, revision: BS000501

In [3]: generate(2000, 'triangle')
Out[3]: 2000.0

In [4]: t = pandas.DataFrame(capture(['A', 'B'], low=0, high=3.3))

In [5]: t.interpolate().plot()
Out[5]: <matplotlib.axes._subplots.AxesSubplot at 0x10db77d30>

In [6]: t = pandas.DataFrame(capture(['L'], low=0, high=3.3))

In [7]: t.plot()
Out[7]: <matplotlib.axes._subplots.AxesSubplot at 0x10d05d5f8>

In [8]: 
"""

import pandas

async def main():
    global s
    parser = argparse.ArgumentParser(description="scopething")
    parser.add_argument('device', nargs='?', default=None, type=str, help="Device to connect to")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, stream=sys.stdout)

    s = await Scope.connect(args.device)
    #await s.start_generator(2000, 'triangle')
    #x = np.linspace(0, 2*np.pi, s.awg_wavetable_size, endpoint=False)
    #y = np.round((np.sin(x)**5)*127 + 128, 0).astype('uint8')
    #await s.start_generator(1000, wavetable=y)

def await(g):
    return asyncio.get_event_loop().run_until_complete(g)

def capture(*args, **kwargs):
    return await(s.capture(*args, **kwargs))

def calibrate(*args, **kwargs):
    return await(s.calibrate(*args, **kwargs))

def generate(*args, **kwargs):
    return await(s.start_generator(*args, **kwargs))

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

