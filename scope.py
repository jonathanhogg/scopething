#!/usr/bin/env python3

import argparse
import array
import asyncio
from collections import namedtuple
import logging
import math
import os
import sys

import streams
import vm


Log = logging.getLogger('scope')


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Scope(vm.VirtualMachine):

    AnalogParams = namedtuple('AnalogParams', ['rd', 'rr', 'rt', 'rb', 'scale', 'offset'])

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
            raise ValueError(f"Don't know what to do with {device!r}")
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
            self.analog_params = self.AnalogParams(20, 300, 335, 355, 18.5, -7.585)
            self.analog_offsets = {'A': -9.5e-3, 'B': 9.5e-3}
            self.analog_default_low = -5.5
            self.analog_default_high = 8
            self.analog_lo_min = 0.07
            self.analog_hi_max = 0.88
            self.logic_low = 0
            self.logic_high = 3.3
            self.capture_clock_period = 25e-9
            self.capture_buffer_size = 12<<10
            self.timeout_clock_period = 6.4e-6
            self.timestamp_rollover = (1<<32) * self.capture_clock_period
        else:
            raise RuntimeError(f"Unsupported scope, revision: {revision}")
        self._awg_running = False
        Log.info(f"Initialised scope, revision: {revision}")

    def calculate_lo_hi(self, low, high, params=None):
        params = self.analog_params if params is None else self.AnalogParams(*params)
        l = (low - params.offset) / params.scale
        h = (high - params.offset) / params.scale
        dl = l - params.rd*(h-l)/params.rr + params.rd*l/params.rb
        dh = h + params.rd*(h-l)/params.rr - params.rd*(1-h)/params.rt
        return dl, dh

    async def capture(self, channels=['A'], trigger=None, trigger_level=None, trigger_type='rising', hair_trigger=False,
                      period=1e-3, nsamples=1000, timeout=None, low=None, high=None, raw=False, trigger_position=0.25):
        analog_channels = set()
        logic_channels = set()
        for channel in channels:
            channel = channel.upper()
            if channel in {'A', 'B'}:
                analog_channels.add(channel)
                if trigger is None:
                    trigger = channel
            elif channel == 'L':
                logic_channels.update(range(8))
                if trigger is None:
                    trigger = {0: 1}
            elif channel in {'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'}:
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
        analog_enable = sum(1<<(ord(channel)-ord('A')) for channel in analog_channels)
        logic_enable = sum(1<<channel for channel in logic_channels)

        ticks = int(round(period / nsamples / self.capture_clock_period))
        for capture_mode in vm.CaptureModes:
            if capture_mode.analog_channels == len(analog_channels) and capture_mode.logic_channels == bool(logic_channels):
                if ticks in range(capture_mode.clock_low, capture_mode.clock_high + 1):
                    clock_scale = 1
                elif capture_mode.clock_divide and ticks > capture_mode.clock_high:
                    for clock_scale in range(2, 1<<16):
                        test_ticks = int(round(period / nsamples / self.capture_clock_period / clock_scale))
                        if test_ticks in range(capture_mode.clock_low, capture_mode.clock_high + 1):
                            ticks = test_ticks
                            break
                    else:
                        continue
                else:
                    continue
                if capture_mode.clock_max is not None and ticks > capture_mode.clock_max:
                    ticks = capture_mode.clock_max
                nsamples = int(round(period / ticks / self.capture_clock_period / clock_scale))
                if len(analog_channels) == 2:
                    nsamples -= nsamples % 2
                buffer_width = self.capture_buffer_size // capture_mode.sample_width
                if logic_channels and analog_channels:
                    buffer_width //= 2
                if nsamples <= buffer_width:
                    break
        else:
            raise ValueError("Unable to find appropriate capture mode")
        
        if raw:
            lo, hi = low, high
        else:
            if low is None:
                low = self.analog_default_low if analog_channels else self.logic_low
            elif low < self.analog_default_low:
                Log.warning(f"Voltage range is below safe minimum: {low} < {self.analog_default_low}")
            if high is None:
                high = self.analog_default_high if analog_channels else self.logic_high
            elif high > self.analog_default_high:
                Log.warning(f"Voltage range is above safe maximum: {high} > {self.analog_default_high}")
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
        trigger_level = (trigger_level - self.analog_params.offset) / self.analog_params.scale
        if trigger == 'A' or trigger == 'B':
            if trigger == 'A':
                spock_option |= vm.SpockOption.TriggerSourceA
                trigger_logic = 0x80
            elif trigger == 'B':
                spock_option |= vm.SpockOption.TriggerSourceB
                trigger_logic = 0x40
            trigger_mask = 0xff ^ trigger_logic
        elif isinstance(trigger, dict):
            trigger_logic = 0
            trigger_mask = 0xff
            for channel, value in trigger.items():
                if isinstance(channel, str):
                    if channel.startswith('L'):
                        channel = int(channel[1:])
                    else:
                        raise TypeError("Unrecognised trigger value")
                if channel < 0 or channel > 7:
                    raise TypeError("Unrecognised trigger value")
                mask = 1<<channel
                trigger_mask &= ~mask
                if value:
                    trigger_logic |= mask
        else:
            raise TypeError("Unrecognised trigger value")
        if trigger_type.lower() in {'falling', 'below'}:
            spock_option |= vm.SpockOption.TriggerInvert
        trigger_outro = 4 if hair_trigger else 8
        trigger_intro = 0 if trigger_type.lower() in {'above', 'below'} else trigger_outro
        trigger_samples = min(max(0, int(nsamples*trigger_position)), nsamples)
        trace_outro = max(0, nsamples-trigger_samples-trigger_outro)
        trace_intro = max(0, trigger_samples-trigger_intro)
        if timeout is None:
            trigger_timeout = 0
        else:
            trigger_timeout = max(1, int(math.ceil(((trigger_intro+trigger_outro+trace_outro+2)*ticks*clock_scale*self.capture_clock_period
                                                    + timeout)/self.timeout_clock_period)))

        sample_period = ticks*clock_scale*self.capture_clock_period
        sample_rate = 1/sample_period
        Log.info(f"Begin {('mixed' if logic_channels else 'analogue') if analog_channels else 'logic'} signal capture "
                  f"at {sample_rate:,.0f} samples per second (trace mode {capture_mode.trace_mode.name})")
        async with self.transaction():
            await self.set_registers(TraceMode=capture_mode.trace_mode, BufferMode=capture_mode.buffer_mode,
                                     SampleAddress=0, ClockTicks=ticks, ClockScale=clock_scale,
                                     TriggerLevel=trigger_level, TriggerLogic=trigger_logic, TriggerMask=trigger_mask,
                                     TraceIntro=trace_intro, TraceOutro=trace_outro, TraceDelay=0, Timeout=trigger_timeout,
                                     TriggerIntro=trigger_intro//2, TriggerOutro=trigger_outro//2, Prelude=0,
                                     SpockOption=spock_option, ConverterLo=lo, ConverterHi=hi,
                                     KitchenSinkA=kitchen_sink_a, KitchenSinkB=kitchen_sink_b, 
                                     AnalogEnable=analog_enable, DigitalEnable=logic_enable)
            await self.issue_program_spock_registers()
            await self.issue_configure_device_hardware()
            await self.issue_triggered_trace()
        while True:
            try:
                code, timestamp = (int(x, 16) for x in await self.read_replies(2))
                if code != vm.TraceStatus.Wait:
                    break
            except asyncio.CancelledError:
                await self.issue_cancel_trace()
        cause = {vm.TraceStatus.Done: 'trigger', vm.TraceStatus.Auto: 'timeout', vm.TraceStatus.Stop: 'cancel'}[code]
        start_timestamp = timestamp - nsamples*ticks*clock_scale
        if start_timestamp < 0:
            start_timestamp += 1<<32
            timestamp += 1<<32
        address = int((await self.read_replies(1))[0], 16)
        if capture_mode.analog_channels == 2:
            address -= address % 2

        traces = DotDict()
        timestamps = array.array('d', (t*self.capture_clock_period for t in range(start_timestamp, timestamp, ticks*clock_scale)))
        start_time = start_timestamp*self.capture_clock_period
        for dump_channel, channel in enumerate(sorted(analog_channels)):
            asamples = nsamples // len(analog_channels)
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - nsamples) % buffer_width,
                                         DumpMode=vm.DumpMode.Native if capture_mode.sample_width == 2 else vm.DumpMode.Raw, 
                                         DumpChan=dump_channel, DumpCount=asamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            value_multiplier, value_offset = (1, 0) if raw else ((high-low), low+self.analog_offsets[channel])
            data = await self.read_analog_samples(asamples, capture_mode.sample_width)
            traces[channel] = DotDict({'timestamps': timestamps[dump_channel::len(analog_channels)] if len(analog_channels) > 1 else timestamps,
                                       'samples': array.array('d', (value*value_multiplier+value_offset for value in data)),
                                       'start_time': start_time+sample_period*dump_channel,
                                       'sample_period': sample_period*len(analog_channels),
                                       'sample_rate': sample_rate/len(analog_channels),
                                       'cause': cause})
        if logic_channels:
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - nsamples) % buffer_width,
                                         DumpMode=vm.DumpMode.Raw, DumpChan=128, DumpCount=nsamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            data = await self.read_logic_samples(nsamples)
            for i in logic_channels:
                mask = 1<<i
                traces[f'L{i}'] = DotDict({'timestamps': timestamps,
                                           'samples': array.array('B', (1 if value & mask else 0 for value in data)),
                                           'start_time': start_time,
                                           'sample_period': sample_period,
                                           'sample_rate': sample_rate,
                                           'cause': cause})
        Log.info(f"{nsamples} samples captured on {cause}, traces: {', '.join(traces)}")
        return traces

    async def start_generator(self, frequency, waveform='sine', wavetable=None, ratio=0.5,
                              low=0, high=None, min_samples=50, max_error=1e-4):
        if high is None:
            high = self.awg_maximum_voltage
        elif high < 0 or high > self.awg_maximum_voltage:
            raise ValueError(f"high out of range (0-{self.awg_maximum_voltage})")
        if low < 0 or low > high:
            raise ValueError("offset out of range (0-high)")
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
                wavetable = [min(max(0, int(round(y*255))),255) for y in wavetable]
                if len(wavetable) != self.awg_wavetable_size:
                    raise ValueError(f"Wavetable data must be {self.awg_wavetable_size} samples")
                await self.set_registers(Cmd=0, Mode=1, Address=0, Size=1)
                await self.wavetable_write_bytes(wavetable)
        async with self.transaction():
            offset = (high+low)/2 - self.awg_maximum_voltage/2
            await self.set_registers(Cmd=0, Mode=0, Level=(high-low)/self.awg_maximum_voltage,
                                     Offset=offset/self.awg_maximum_voltage,
                                     Ratio=nwaves*self.awg_wavetable_size/size,
                                     Index=0, Address=0, Size=size)
            await self.issue_translate_wavetable()
        async with self.transaction():
            await self.set_registers(Cmd=2, Mode=0, Clock=clock, Modulo=size,
                                     Mark=10, Space=1, Rest=0x7f00, Option=0x8004)
            await self.issue_control_waveform_generator()
        async with self.transaction():
            await self.set_registers(KitchenSinkB=vm.KitchenSinkB.WaveformGeneratorEnable)
            await self.issue_configure_device_hardware()
        self._awg_running = True
        Log.info(f"Signal generator running at {actualf:,0.1f}Hz")
        return actualf

    async def stop_generator(self):
        async with self.transaction():
            await self.set_registers(Cmd=1, Mode=0)
            await self.issue_control_waveform_generator()
            await self.set_registers(KitchenSinkB=0)
            await self.issue_configure_device_hardware()
        Log.info("Signal generator stopped")
        self._awg_running = False

    async def read_wavetable(self):
        with self.transaction():
            self.set_registers(Address=0, Size=self.awg_wavetable_size)
            self.issue_wavetable_read()
        return list(self.wavetable_read_bytes(self.awg_wavetable_size))

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
        for lo in np.linspace(self.analog_lo_min, 0.5, n, endpoint=False):
            for hi in np.linspace(0.5, self.analog_hi_max, n):
                traces = await self.capture(channels=['A','B'], period=2e-3, nsamples=2000, timeout=0, low=lo, high=hi, raw=True)
                A = np.array(traces.A.samples)
                A.sort()
                Azero, Amax = A[25:475].mean(), A[525:975].mean()
                if Azero < 0.01 or Amax > 0.99:
                    continue
                B = np.array(traces.B.samples)
                B.sort()
                Bzero, Bmax = B[25:475].mean(), B[525:975].mean()
                if Bzero < 0.01 or Bmax > 0.99:
                    continue
                zero = (Azero + Bzero) / 2
                analog_range = self.awg_maximum_voltage / ((Amax + Bmax)/2 - zero)
                low = -zero * analog_range
                high = low + analog_range
                offset = ((Amax - Bmax) + (Azero - Bzero))/2 * analog_range
                items.append((lo, hi, low, high, offset))
        await self.stop_generator()
        items = np.array(items).T
        def f(params, lo, hi, low, high, offset):
            clo, chi = self.calculate_lo_hi(low, high, params)
            return np.sqrt((lo-clo)**2 + (hi-chi)**2)
        result = least_squares(f, self.analog_params, args=items, bounds=([0, 200, 200, 200, 18, -8], [50, 400, 400, 400, 19, -7]))
        if result.success:
            Log.info(f"Calibration succeeded: {result.message}")
            params = self.analog_params = self.AnalogParams(*result.x)
            Log.info(f"Analog parameters: rd={params.rd:.1f}Ω rr={params.rr:.1f}Ω rt={params.rt:.1f}Ω rb={params.rb:.1f}Ω "
                     f"scale={params.scale:.3f}V offset={params.offset:.3f}V")
            lo, hi, low, high, offset = items
            clo, chi = self.calculate_lo_hi(low, high)
            lo_error = np.sqrt((((clo-lo)/(hi-lo))**2).mean())
            hi_error = np.sqrt((((chi-hi)/(hi-lo))**2).mean())
            Log.info(f"Mean error: lo={lo_error*10000:.1f}bps hi={hi_error*10000:.1f}bps")
            offset_mean = offset.mean()
            Log.info(f"Mean A-B offset: {offset_mean*1000:.1f}mV (+/- {100*offset.std()/offset_mean:.1f}%)")
            self.analog_offsets = {'A': -offset_mean/2, 'B': +offset_mean/2}
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

In [4]: capturep(['A', 'B'], low=0, high=3.3).interpolate().plot()
Out[4]: <matplotlib.axes._subplots.AxesSubplot at 0x10db77d30>

In [5]: capturep(['L'], low=0, high=3.3)).plot()
Out[5]: <matplotlib.axes._subplots.AxesSubplot at 0x10d05d5f8>

In [6]: 
"""

async def main():
    global s
    parser = argparse.ArgumentParser(description="scopething")
    parser.add_argument('device', nargs='?', default=None, type=str, help="Device to connect to")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug logging")
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING), stream=sys.stdout)
    s = await Scope.connect(args.device)

def await(g):
    task = asyncio.Task(g)
    while True:
        try:
            return asyncio.get_event_loop().run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()

def capture(*args, **kwargs):
    return await(s.capture(*args, **kwargs))

def capturep(*args, **kwargs):
    import pandas
    traces = capture(*args, **kwargs)
    return pandas.DataFrame({channel: pandas.Series(trace.samples, trace.timestamps) for (channel,trace) in traces.items()})

def calibrate(*args, **kwargs):
    return await(s.calibrate(*args, **kwargs))

def generate(*args, **kwargs):
    return await(s.start_generator(*args, **kwargs))

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

