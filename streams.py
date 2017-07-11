"""
streams
=======

Package for asynchronous serial IO.
"""

import asyncio
import logging
import os
import sys
import serial
from serial.tools.list_ports import comports


Log = logging.getLogger('streams')


if sys.platform == 'linux':

    import fcntl
    import struct

    TIOCGSERIAL = 0x541E
    TIOCSSERIAL = 0x541F
    ASYNC_LOW_LATENCY = 1<<13
    SERIAL_STRUCT_FORMAT = '2iI5iH2ci2HPHIL'
    SERIAL_FLAGS_INDEX = 4

    def set_low_latency(fd):
        data = list(struct.unpack(SERIAL_STRUCT_FORMAT, fcntl.ioctl(fd, TIOCGSERIAL, b'\0'*struct.calcsize(SERIAL_STRUCT_FORMAT))))
        data[SERIAL_FLAGS_INDEX] |= ASYNC_LOW_LATENCY
        fcntl.ioctl(fd, TIOCSSERIAL, struct.pack(SERIAL_STRUCT_FORMAT, *data))

else:

    def set_low_latency(fd):
        pass



class SerialStream:

    @classmethod
    def stream_matching(cls, vid, pid, **kwargs):
        for port in comports():
            if port.vid == vid and port.pid == pid:
                return SerialStream(port.device, **kwargs)
        raise RuntimeError("No matching serial device")

    def __init__(self, device, loop=None, **kwargs):
        self._device = device
        self._connection = serial.Serial(self._device, timeout=0, write_timeout=0, **kwargs)
        set_low_latency(self._connection)
        Log.debug(f"Opened SerialStream on {device}")
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._output_buffer = bytes()
        self._output_buffer_empty = None
        self._pushback_buffer = bytes()

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self._device}>'

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def write(self, data):
        if not self._output_buffer:
            try:
                n = self._connection.write(data)
            except serial.SerialTimeoutException:
                n = 0
            except:
                Log.exception("Error writing to stream")
                raise
            if n:
                Log.debug(f"Write {data[:n]!r}")
            self._output_buffer = data[n:]
        else:
            self._output_buffer += data
        if self._output_buffer and self._output_buffer_empty is None:
            self._output_buffer_empty = self._loop.create_future()
            self._loop.add_writer(self._connection, self._feed_data)

    async def drain(self):
        if self._output_buffer_empty is not None:
            await self._output_buffer_empty

    def _feed_data(self):
        try:
            n = self._connection.write(self._output_buffer)
        except serial.SerialTimeoutException:
            n = 0
        except Exception as e:
            Log.exception("Error writing to stream")
            self._output_buffer_empty.set_exception(e)
            self.remove_writer(self._connection)
        if n:
            Log.debug(f"Write {self._output_buffer[:n]!r}")
            self._output_buffer = self._output_buffer[n:]
        if not self._output_buffer:
            self._loop.remove_writer(self._connection)
            self._output_buffer_empty.set_result(None)
            self._output_buffer_empty = None

    def pushback(self, data):
        self._pushback_buffer += data

    async def read(self, n=None):
        if self._pushback_buffer:
            data = self._pushback_buffer if n is None else self._pushback_buffer[:n]
            self._pushback_buffer = self._pushback_buffer[len(data):]
            return data
        while True:
            w = self._connection.in_waiting
            if w:
                data = self._connection.read(w if n is None else min(n,w))
                Log.debug(f"Read {data!r}")
                return data
            else:
                future = self._loop.create_future()
                self._loop.add_reader(self._connection, future.set_result, None)
                try:
                    await future
                finally:
                    self._loop.remove_reader(self._connection)

    async def readexactly(self, n):
        data = b''
        while len(data) < n:
            data += await self.read(n-len(data))
        return data

