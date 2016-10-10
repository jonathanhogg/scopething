
import asyncio
import glob
import serial


class SerialStream(object):
    
    def __init__(self, device=None, loop=None, **kwargs):
        if device is None:
            device = (glob.glob('/dev/tty.usb*') + glob.glob('/dev/ttyUSB*'))[0]
        self._connection = serial.Serial(device, timeout=0, write_timeout=0, **kwargs)
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._input_buffer = b''

    async def write(self, data):
        while data:
            n = await self._write(data)
            data = data[n:]

    def _write(self, data):
        future = asyncio.Future()
        self._loop.add_writer(self._connection, self._feed_data, data, future)
        return future

    def _feed_data(self, data, future):
        future.set_result(self._connection.write(data))
        self._loop.remove_writer(self._connection)

    async def read(self, n=None):
        while True:
            if self._input_buffer:
                if n is None:
                    data, self._input_buffer = self._input__buffer, b''
                else:
                    data, self._input_buffer = self._input_buffer[:n], self._input_buffer[n:]
                return data
            if n is None:
                self._input_buffer += await self._read()
            else:
                self._input_buffer += await self._read(n - len(self._input_buffer))

    async def readexactly(self, n):
        while True:
            if len(self._input_buffer) >= n:
                data, self._input_buffer = self._input_buffer[:n], self._input_buffer[n:]
                return data
            self._input_buffer += await self._read(n - len(self._input_buffer))

    async def readuntil(self, separator):
        while True:
            index = self._input_buffer.find(separator)
            if index >= 0:
                index += len(separator)
                data, self._input_buffer = self._input_buffer[:index], self._input_buffer[index:]
                return data
            self._input_buffer += await self._read()

    def _read(self, n=None):
        future = asyncio.Future()
        self._loop.add_reader(self._connection, self._handle_data, n, future)
        return future

    def _handle_data(self, n, future):
        future.set_result(self._connection.read(n if n is not None else self._connection.in_waiting))
        self._loop.remove_reader(self._connection)


