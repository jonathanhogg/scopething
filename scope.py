
import asyncio
from streams import SerialStream
from vm import VirtualMachine


class Scope(VirtualMachine):

    @classmethod
    async def connect(cls, stream=None):
        scope = cls(stream if stream is not None else SerialStream())
        await scope.setup()
        return scope

    def __init__(self, stream):
        super(Scope, self).__init__(stream)

    async def setup(self):
        await self.reset()
        await self.issue_get_revision()
        revision = (await self.read_reply()).decode('ascii')
        if revision.startswith('BS0005'):
            self.awg_clock_period = 25e-9
            self.awg_wavetable_size = 1024
            self.awg_sample_buffer_size = 1024
            self.awg_minimum_clock = 33
            self.awg_maximum_voltage = 3.3

    async def generate_waveform(self, frequency, waveform='sine', ratio=0.5, vpp=None, offset=0, min_samples=40, max_error=0.0001):
        if vpp is None:
            vpp = self.awg_maximum_voltage
        best_width, best_params = None, None
        clock = self.awg_minimum_clock
        while True:
            width = 1 / frequency / (clock * self.awg_clock_period)
            if width <= self.awg_sample_buffer_size:
                nwaves = int(self.awg_sample_buffer_size / width)
                size = int(round(nwaves * width))
                width = size / nwaves
                if width < min_samples:
                    break
                actualf = 1 / (size / nwaves * clock * self.awg_clock_period)
                if abs(frequency - actualf) / frequency < max_error and (best_width is None or width > best_width):
                    best_width, best_params = width, (size, nwaves, clock, actualf)
            clock += 1
        if best_params is None:
            raise ValueError("Unable to find appropriate solution to required frequency")
        size, nwaves, clock, actualf = best_params
        async with self.transaction():
            await self.set_registers(vrKitchenSinkB=VirtualMachine.KITCHENSINKB_WAVEFORM_GENERATOR_ENABLE)
            await self.issue_configure_device_hardware()
        await self.synthesize_wavetable(waveform, ratio)
        await self.translate_wavetable(nwaves=nwaves, size=size, level=vpp/self.awg_maximum_voltage, offset=offset/self.awg_maximum_voltage)
        await self.start_waveform_generator(clock=clock, modulo=size, mark=10, space=2, rest=0x7f00, option=0x8004)
        return actualf

    async def stop_generator(self):
        await self.stop_waveform_generator()
        async with self.transaction():
            await self.set_registers(vrKitchenSinkB=0)
            await self.issue_configure_device_hardware()

    async def read_wavetable(self):
        with self.transaction():
            self.set_registers(vpAddress=0, vpSize=self.awg_wavetable_size)
            self.issue_wavetable_read()
        return list(self.read_exactly(self.awg_wavetable_size))

    async def write_wavetable(self, data):
        if len(data) != self.awg_wavetable_size:
            raise ValueError("Wavetable data must be {} samples".format(self.awg_wavetable_size))
        with self.transaction():
            self.set_registers(vpAddress=0, vpSize=1)
            for byte in data:
                self.wavetable_write(byte)

    async def synthesize_wavetable(self, waveform='sine', ratio=0.5):
        mode = {'sine': 0, 'sawtooth': 1, 'exponential': 2, 'square': 3}[waveform.lower()]
        async with self.transaction():
            await self.set_registers(vpCmd=0, vpMode=mode, vpRatio=ratio)
            await self.issue_synthesize_wavetable()

    async def translate_wavetable(self, nwaves, size, level=1, offset=0, index=0, address=0):
        async with self.transaction():
            await self.set_registers(vpCmd=0, vpMode=0, vpLevel=level, vpOffset=offset,
                                     vpRatio=nwaves * self.awg_wavetable_size / size,
                                     vpIndex=index, vpAddress=address, vpSize=size)
            await self.issue_translate_wavetable()

    async def start_waveform_generator(self, clock, modulo, mark, space, rest, option):
        async with self.transaction():
            await self.set_registers(vpCmd=2, vpMode=0, vpClock=clock, vpModulo=modulo, 
                                     vpMark=mark, vpSpace=space, vrRest=rest, vpOption=option)
            await self.issue_control_waveform_generator()

    async def read_eeprom(self, address):
        async with self.transaction():
            await self.set_registers(vrEepromAddress=address)
            await self.issue_read_eeprom()
        return int(await self.read_reply(), 16)

    async def write_eeprom(self, address, byte):
        async with self.transaction():
            await self.set_registers(vrEepromAddress=address, vrEepromData=byte)
            await self.issue_write_eeprom()
        return int(await self.read_reply(), 16)


async def main():
    global s
    s = await Scope.connect()
    print(await s.generate_waveform(440*16, 'sawtooth'))


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

