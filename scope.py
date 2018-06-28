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


LOG = logging.getLogger('scope')


class UsageError(Exception):
    pass

class ConfigurationError(Exception):
    pass


class Scope(vm.VirtualMachine):

    AnalogParams = namedtuple('AnalogParams', ['la', 'lb', 'lc', 'ha', 'hb', 'hc', 'scale', 'offset', 'safe_low', 'safe_high', 'ab_offset'])

    @classmethod
    async def connect(cls, device=None):
        if device is None:
            reader = writer = streams.SerialStream.stream_matching(0x0403, 0x6001)
        elif os.path.exists(device):
            reader = writer = streams.SerialStream(device=device)
        elif ':' in device:
            host, port = device.split(':', 1)
            LOG.info(f"Connecting to remote scope at {host}:{port}")
            reader, writer = await asyncio.open_connection(host, int(port))
        else:
            raise ValueError(f"Don't know what to do with {device!r}")
        scope = cls(reader, writer)
        await scope.setup()
        return scope

    async def setup(self):
        LOG.info("Resetting scope")
        await self.reset()
        await self.issue_get_revision()
        revision = ((await self.read_replies(2))[1]).decode('ascii')
        if revision == 'BS000501':
            self.master_clock_period = 25e-9
            self.capture_buffer_size = 12<<10
            self.awg_wavetable_size = 1024
            self.awg_sample_buffer_size = 1024
            self.awg_minimum_clock = 33
            self.logic_low = 0
            self.awg_maximum_voltage = self.clock_voltage = self.logic_high = 3.3
            self.analog_params = {'x1':  self.AnalogParams(1.11, -6.57e-2, 8.46e-3, 1.11, -7.32e-2, -5.19e-2, 18.28, -7.45, -5.5, 8, 5.3e-3),
                                  'x10': self.AnalogParams(1.10, -6.11e-2, 8.61e-3, 1.10, -6.68e-2, -4.32e-2, 184.3, -90.4, -71.3, 65.7, 175e-3)}
            self.analog_lo_min = 0.07
            self.analog_hi_max = 0.88
            self.timeout_clock_period = (1<<8) * self.master_clock_period
            self.timestamp_rollover = (1<<32) * self.master_clock_period
        else:
            raise RuntimeError(f"Unsupported scope, revision: {revision}")
        self._awg_running = False
        self._clock_running = False
        LOG.info(f"Initialised scope, revision: {revision}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        super().close()
        LOG.info("Closed scope")

    def calculate_lo_hi(self, low, high, params):
        if not isinstance(params, self.AnalogParams):
            params = self.AnalogParams(*(list(params) + [None]*(11-len(params))))
        l = (low - params.offset) / params.scale
        h = (high - params.offset) / params.scale
        dl = params.la*l + params.lb*h + params.lc
        dh = params.ha*h + params.hb*l + params.hc
        return dl, dh

    async def capture(self, channels=['A'], trigger=None, trigger_level=None, trigger_type='rising', hair_trigger=False,
                      period=1e-3, nsamples=1000, timeout=None, low=None, high=None, raw=False, trigger_position=0.25, probes='x1'):
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
        if self._clock_running and 5 in logic_channels:
            logic_channels.remove(5)
        if 'A' in analog_channels and 7 in logic_channels:
            logic_channels.remove(7)
        if 'B' in analog_channels and 6 in logic_channels:
            logic_channels.remove(6)
        analog_enable = sum(1<<(ord(channel)-ord('A')) for channel in analog_channels)
        logic_enable = sum(1<<channel for channel in logic_channels)

        ticks = int(round(period / nsamples / self.master_clock_period))
        for capture_mode in vm.CaptureModes:
            if capture_mode.analog_channels == len(analog_channels) and capture_mode.logic_channels == bool(logic_channels):
                if ticks > capture_mode.clock_high and capture_mode.clock_divide:
                    for clock_scale in range(2, vm.Registers.ClockScale.maximum_value+1):
                        test_ticks = int(round(period / nsamples / self.master_clock_period / clock_scale))
                        if test_ticks in range(capture_mode.clock_low, capture_mode.clock_high + 1):
                            ticks = test_ticks
                            break
                    else:
                        continue
                    break
                elif ticks >= capture_mode.clock_low:
                    clock_scale = 1
                    if ticks > capture_mode.clock_high:
                        ticks = capture_mode.clock_high
                else:
                    continue
                n = int(round(period / ticks / self.master_clock_period / clock_scale))
                if len(analog_channels) == 2:
                    n -= n % 2
                buffer_width = self.capture_buffer_size // capture_mode.sample_width
                if logic_channels and analog_channels:
                    buffer_width //= 2
                if n <= buffer_width:
                    nsamples = n
                    break
        else:
            raise ConfigurationError("Unable to find appropriate capture mode")
        
        analog_params = self.analog_params[probes]
        if raw:
            lo, hi = low, high
        else:
            if low is None:
                low = analog_params.safe_low if analog_channels else self.logic_low
            elif low < analog_params.safe_low:
                LOG.warning(f"Voltage range is below safe minimum: {low} < {analog_params.safe_low}")
            if high is None:
                high = analog_params.safe_high if analog_channels else self.logic_high
            elif high > analog_params.safe_high:
                LOG.warning(f"Voltage range is above safe maximum: {high} > {analog_params.safe_high}")
            lo, hi = self.calculate_lo_hi(low, high, analog_params)

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
        trigger_level = (trigger_level - analog_params.offset) / analog_params.scale
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
            trigger_timeout = int(math.ceil(((trigger_intro+trigger_outro+trace_outro+2)*ticks*clock_scale*self.master_clock_period
                                              + timeout)/self.timeout_clock_period))
            if trigger_timeout > vm.Registers.Timeout.maximum_value:
                if timeout > 0:
                    raise ConfigurationError("Required trigger timeout too long")
                else:
                    raise ConfigurationError("Required trigger timeout too long, use a later trigger position")

        sample_period = ticks*clock_scale*self.master_clock_period
        sample_rate = 1/sample_period
        LOG.info(f"Begin {('mixed' if logic_channels else 'analogue') if analog_channels else 'logic'} signal capture "
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

        traces = vm.DotDict()
        timestamps = array.array('d', (t*self.master_clock_period for t in range(start_timestamp, timestamp, ticks*clock_scale)))
        start_time = start_timestamp*self.master_clock_period
        for dump_channel, channel in enumerate(sorted(analog_channels)):
            asamples = nsamples // len(analog_channels)
            async with self.transaction():
                await self.set_registers(SampleAddress=(address - nsamples) % buffer_width,
                                         DumpMode=vm.DumpMode.Native if capture_mode.sample_width == 2 else vm.DumpMode.Raw, 
                                         DumpChan=dump_channel, DumpCount=asamples, DumpRepeat=1, DumpSend=1, DumpSkip=0)
                await self.issue_program_spock_registers()
                await self.issue_analog_dump_binary()
            value_multiplier, value_offset = (1, 0) if raw else (high-low, low-analog_params.ab_offset/2*(1 if channel == 'A' else -1))
            data = await self.read_analog_samples(asamples, capture_mode.sample_width)
            traces[channel] = vm.DotDict({'timestamps': timestamps[dump_channel::len(analog_channels)] if len(analog_channels) > 1 else timestamps,
                                          'samples': array.array('f', (value*value_multiplier+value_offset for value in data)),
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
                traces[f'L{i}'] = vm.DotDict({'timestamps': timestamps,
                                              'samples': array.array('B', (1 if value & mask else 0 for value in data)),
                                              'start_time': start_time,
                                              'sample_period': sample_period,
                                              'sample_rate': sample_rate,
                                              'cause': cause})
        LOG.info(f"{nsamples} samples captured on {cause}, traces: {', '.join(traces)}")
        return traces

    async def start_waveform(self, frequency, waveform='sine', ratio=0.5, low=0, high=None, min_samples=50, max_error=1e-4):
        if self._clock_running:
            raise UsageError("Cannot start waveform generator while clock in use")
        if high is None:
            high = self.awg_maximum_voltage
        elif high < 0 or high > self.awg_maximum_voltage:
            raise ValueError(f"high out of range (0-{self.awg_maximum_voltage})")
        if low < 0 or low > high:
            raise ValueError("low out of range (0-high)")
        possible_params = []
        max_clock = int(math.floor(1 / frequency / min_samples / self.master_clock_period))
        for clock in range(self.awg_minimum_clock, max_clock+1):
            width = 1 / frequency / (clock * self.master_clock_period)
            if width <= self.awg_sample_buffer_size:
                nwaves = int(self.awg_sample_buffer_size / width)
                size = int(round(nwaves * width))
                width = size / nwaves
                actualf = 1 / (width * clock * self.master_clock_period)
                error = abs(frequency - actualf) / frequency
                if error < max_error:
                    possible_params.append((width if error == 0 else -error, (size, nwaves, clock, actualf)))
        if not possible_params:
            raise ConfigurationError("No solution to required frequency/min_samples/max_error")
        size, nwaves, clock, actualf = sorted(possible_params)[-1][1]
        async with self.transaction():
            if isinstance(waveform, str):
                mode = {'sine': 0, 'triangle': 1, 'exponential': 2, 'square': 3}[waveform.lower()]
                await self.set_registers(Cmd=0, Mode=mode, Ratio=ratio)
                await self.issue_synthesize_wavetable()
            elif len(wavetable) == self.awg_wavetable_size:
                wavetable = bytes(min(max(0, int(round(y*255))),255) for y in wavetable)
                await self.set_registers(Cmd=0, Mode=1, Address=0, Size=1)
                await self.wavetable_write_bytes(wavetable)
            else:
                raise ValueError(f"waveform must be a valid name or {self.awg_wavetable_size} samples")
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
            await self.issue_control_clock_generator()
        async with self.transaction():
            await self.set_registers(KitchenSinkB=vm.KitchenSinkB.WaveformGeneratorEnable)
            await self.issue_configure_device_hardware()
        self._awg_running = True
        LOG.info(f"Signal generator running at {actualf:0.1f}Hz")
        return actualf

    async def stop_waveform(self):
        if not self._awg_running:
            raise UsageError("Waveform generator not in use")
        async with self.transaction():
            await self.set_registers(Cmd=1, Mode=0)
            await self.issue_control_clock_generator()
            await self.set_registers(KitchenSinkB=0)
            await self.issue_configure_device_hardware()
        LOG.info("Signal generator stopped")
        self._awg_running = False

    async def start_clock(self, frequency, ratio=0.5, max_error=1e-4):
        if self._awg_running:
            raise UsageError("Cannot start clock while waveform generator in use")
        ticks = min(max(2, int(round(1 / frequency / self.master_clock_period))), vm.Registers.Clock.maximum_value)
        fall = min(max(1, int(round(ticks * ratio))), ticks-1)
        actualf, actualr = 1 / ticks / self.master_clock_period, fall / ticks
        if abs(actualf - frequency) / frequency > max_error:
            raise ConfigurationError("No solution to required frequency and max_error")
        async with self.transaction():
            await self.set_registers(Map5=0x12, Clock=ticks, Rise=0, Fall=fall, Control=0x80, Cmd=3, Mode=0)
            await self.issue_control_clock_generator()
        self._clock_running = True
        LOG.info(f"Clock generator running at {actualf:0.1f}Hz, {actualr*100:.0f}% duty cycle")
        return actualf, actualr

    async def stop_clock(self):
        if not self._clock_running:
            raise UsageError("Clock not in use")
        async with self.transaction():
            await self.set_registers(Map5=0, Cmd=1, Mode=0)
            await self.issue_control_clock_generator()
        LOG.info("Clock generator stopped")
        self._clock_running = False

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

    async def calibrate(self, probes='x1', n=32):
        import numpy as np
        from scipy.optimize import minimize
        items = []
        async def measure(lo, hi, period=2e-3, chop=True):
            if chop:
                traces = await self.capture(channels=['A','B'], period=period, nsamples=2000, timeout=0, low=lo, high=hi, raw=True)
                A = np.array(traces.A.samples)
                B = np.array(traces.B.samples)
            else:
                A = np.array((await self.capture(channels=['A'], period=period/2, nsamples=1000, timeout=0, low=lo, high=hi, raw=True)).A.samples)
                B = np.array((await self.capture(channels=['B'], period=period/2, nsamples=1000, timeout=0, low=lo, high=hi, raw=True)).B.samples)
            Amean = A.mean()
            Azero, Afull = np.median(A[A<=Amean]), np.median(A[A>=Amean])
            Bmean = B.mean()
            Bzero, Bfull = np.median(B[B<=Bmean]), np.median(B[B>=Bmean])
            return (Azero + Bzero) / 2, (Afull + Bfull) / 2, ((Afull - Bfull) + (Azero - Azero)) / 2
        await self.start_clock(frequency=2000)
        zero, full, offset = await measure(1/3, 2/3)
        zero = (zero + 1) / 3
        full = (full + 1) / 3
        analog_scale = self.clock_voltage / (full - zero)
        analog_offset = -zero * analog_scale
        LOG.info(f"Analog full range = {analog_scale:.1f}V, zero offset = {analog_offset:.1f}V")
        for lo in np.linspace(self.analog_lo_min, 0.5, n, endpoint=False):
            for hi in np.linspace(self.analog_hi_max, 0.5, n):
                period = 2e-3 if len(items) % 4 < 2 else 1e-3
                zero, full, offset = await measure(lo, hi, 2e-3 if len(items) % 4 < 2 else 1e-3, len(items) % 2 == 0)
                if zero > 0.01 and full < 0.99 and full > zero:
                    analog_range = self.clock_voltage / (full - zero)
                    items.append((lo, hi, -zero*analog_range, (1-zero)*analog_range, offset*analog_range))
        await self.stop_clock()
        items = np.array(items).T
        lo, hi, low, high, offset = items
        def f(params):
            dl, dh = self.calculate_lo_hi(low, high, self.AnalogParams(*params, analog_scale, analog_offset, None, None, None))
            return np.sqrt((lo-dl)**2 + (hi-dh)**2).mean()
        start_params = self.analog_params.get(probes, [1,0,0,1,0,0])[:6]
        result = minimize(f, start_params, method='SLSQP',
                          bounds=[(1,np.inf), (-np.inf,0), (0,np.inf), (1,np.inf), (-np.inf,0), (-np.inf,0)],
                          constraints=[{'type': 'eq', 'fun': lambda x: x[0]*1/3 + x[1]*2/3 + x[2] - 1/3},
                                       {'type': 'eq', 'fun': lambda x: x[3]*2/3 + x[4]*1/3 + x[5] - 2/3}])
        if result.success:
            LOG.info(f"Calibration succeeded: {result.message}")
            params = self.AnalogParams(*result.x, analog_scale, analog_offset, None, None, None)
            def f(x):
                lo, hi = self.calculate_lo_hi(x[0], x[1], params)
                return np.sqrt((self.analog_lo_min - lo)**2 + (self.analog_hi_max - hi)**2)
            safe_low, safe_high = minimize(f, (low[0], high[0])).x
            offset_mean = offset.mean()
            params = self.analog_params[probes] = self.AnalogParams(*result.x, analog_scale, analog_offset, safe_low, safe_high, offset_mean)
            LOG.info(f"Analog parameters: la={params.la:.3e} lb={params.lb:.3e} lc={params.lc:.3e} "
                     f"ha={params.ha:.3e} hb={params.hb:.3e} hc={params.hc:.3e} "
                     f"scale={params.scale:.3f}V offset={params.offset:.3f}V "
                     f"safe_low={params.safe_low:.1f}V safe_high={params.safe_high:.1f}V "
                     f"ab_offset={offset_mean*1000:.1f}mV (±{100*offset.std()/offset_mean:.1f}%)")
            clo, chi = self.calculate_lo_hi(low, high, params)
            lo_error = np.sqrt((((clo-lo)/(hi-lo))**2).mean())
            hi_error = np.sqrt((((chi-hi)/(hi-lo))**2).mean())
            LOG.info(f"Mean error: lo={lo_error*10000:.1f}bps hi={hi_error*10000:.1f}bps")
        else:
            LOG.warning(f"Calibration failed: {result.message}")
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

def await_(g):
    task = asyncio.Task(g)
    while True:
        try:
            return asyncio.get_event_loop().run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()

def capture(*args, **kwargs):
    return await_(s.capture(*args, **kwargs))

def capturep(*args, **kwargs):
    import pandas
    traces = capture(*args, **kwargs)
    return pandas.DataFrame({channel: pandas.Series(trace.samples, trace.timestamps) for (channel,trace) in traces.items()})

def calibrate(*args, **kwargs):
    return await_(s.calibrate(*args, **kwargs))

def start_waveform(*args, **kwargs):
    return await_(s.start_waveform(*args, **kwargs))

def start_clock(*args, **kwargs):
    return await_(s.start_clock(*args, **kwargs))

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

