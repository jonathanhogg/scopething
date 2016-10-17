
import asyncio
import logging
import os
import serial
import serial.tools.list_ports


Log = logging.getLogger('streams')


class SerialStream:

    @staticmethod
    def available_ports():
        return [port.device for port in serial.tools.list_ports.comports()]

    def __init__(self, port=-1, loop=None, **kwargs):
        self._device = self.available_ports()[port]
        self._connection = serial.Serial(self._device, timeout=0, write_timeout=0, **kwargs)
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._input_buffer = bytes()

    def __repr__(self):
        return '<{}:{}>'.format(self.__class__.__name__, self._device)

    def close(self):
        self._connection.close()
        self._connection = None

    async def write(self, data):
        while data:
            n = await self._write(data)
            data = data[n:]

    def _write(self, data):
        future = asyncio.Future()
        self._loop.add_writer(self._connection, self._feed_data, data, future)
        return future

    def _feed_data(self, data, future):
        n = self._connection.write(data)
        Log.debug('Write {}'.format(repr(data[:n])))
        future.set_result(n)
        self._loop.remove_writer(self._connection)

    async def read(self, n=None):
        while True:
            if self._input_buffer:
                if n is None:
                    data, self._input_buffer = self._input_buffer, bytes()
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
        data = self._connection.read(n if n is not None else self._connection.in_waiting)
        Log.debug('Read {}'.format(repr(data)))
        future.set_result(data)
        self._loop.remove_reader(self._connection)


