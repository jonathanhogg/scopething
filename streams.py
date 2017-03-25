
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
        self._output_buffer_empty = None

    def __repr__(self):
        return '<{}:{}>'.format(self.__class__.__name__, self._device)

    def close(self):
        self._connection.close()
        self._connection = None

    def write(self, data):
        if not self._output_buffer:
            try:
                n = self._connection.write(data)
            except serial.SerialTimeoutException:
                n = 0
            except Exception as e:
                Log.exception("Error writing to stream")
                raise
            if n:
                Log.debug('Write %r', data[:n])
            self._output_buffer = data[n:]
        else:
            self._output_buffer += data
        if self._output_buffer and self._output_buffer_empty is None:
            self._output_buffer_empty = self._loop.create_future()
            self._loop.add_writer(self._connection, self._feed_data)

    async def send_break(self):
        baudrate = self._connection.baudrate
        await self.drain()
        self._connection.baudrate = 600
        self.write(b'\0')
        await self.drain()
        self._connection.baudrate = baudrate

    async def drain(self):
        if self._output_buffer_empty is not None:
            await self._output_buffer_empty
        n = self._connection.out_waiting
        if n:
            await asyncio.sleep(n * 10 / self._connection.baudrate)

    def _feed_data(self):
        try:
            n = self._connection.write(self._output_buffer)
        except serial.SerialTimeoutException:
            n = 0
        except Exception as e:
            Log.exception("Error writing to stream")
            self._output_buffer_empty.set_exception(e)
            self.remove_writer(self._connection, self._feed_data)
        if n:
            Log.debug('Write %r', self._output_buffer[:n])
            self._output_buffer = self._output_buffer[n:]
        if not self._output_buffer:
            self._loop.remove_writer(self._connection)
            self._output_buffer_empty.set_result(None)
            self._output_buffer_empty = None
        
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

    async def _read(self, n=None):
        future = self._loop.create_future()
        self._loop.add_reader(self._connection, self._handle_data, n, future)
        try:
            data = await future
            Log.debug('Read %r', data)
            return data
        finally:
            self._loop.remove_reader(self._connection)

    def _handle_data(self, n, future):
        if not future.cancelled():
            data = self._connection.read(n if n is not None else self._connection.in_waiting)
            future.set_result(data)


