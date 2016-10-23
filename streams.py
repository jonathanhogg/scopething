
import asyncio
import logging
import os
import serial


Log = logging.getLogger('streams')


class SerialStream:

    def __init__(self, device, loop=None, **kwargs):
        self._device = device
        self._connection = serial.Serial(self._device, timeout=0, write_timeout=0, **kwargs)
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._input_buffer = bytes()
        self._output_buffer = bytes()
        self._output_wait = None

    def __repr__(self):
        return '<{}:{}>'.format(self.__class__.__name__, self._device)

    def close(self):
        self._connection.close()
        self._connection = None

    def write(self, data):
        self._output_buffer += data
        if self._output_wait is None:
            self._output_wait = asyncio.Future()
            self._loop.add_writer(self._connection, self._feed_data)
        
    async def drain(self):
        if self._output_wait is not None:
            await self._output_wait

    def _feed_data(self):
        n = self._connection.write(self._output_buffer)
        Log.debug('Write {}'.format(repr(self._output_buffer[:n])))
        self._output_buffer = self._output_buffer[n:]
        if not self._output_buffer:
            self._loop.remove_writer(self._connection)
            self._output_wait.set_result(None)
            self._output_wait = None
        
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


