"""
streams
=======

Package for asynchronous serial IO and UDP as asyncio API
Serial IO : added by Original Author
UDP : added by Luc MASSON

"""

# pylama:ignore=W1203,R0916,W0703

import asyncio
import asyncio.transports as transports
import logging
import sys
import threading

import serial
from serial.tools.list_ports import comports


Log = logging.getLogger(__name__)


class SerialStream:

    @classmethod
    def devices_matching(cls, vid=None, pid=None, serial_number=None):
        for port in comports():
            if (vid is None or vid == port.vid) and (pid is None or pid == port.pid) and (serial_number is None or serial_number == port.serial_number):
                yield port.device

    @classmethod
    def stream_matching(cls, vid=None, pid=None, serial_number=None, **kwargs):
        for device in cls.devices_matching(vid, pid, serial_number):
            return SerialStream(device, **kwargs)
        raise RuntimeError("No matching serial device")

    def __init__(self, device, use_threads=None, loop=None, **kwargs):
        self._device = device
        self._use_threads = sys.platform == 'win32' if use_threads is None else use_threads
        self._connection = serial.Serial(self._device, **kwargs) if self._use_threads else \
            serial.Serial(self._device, timeout=0, write_timeout=0, **kwargs)
        Log.debug(f"Opened SerialStream on {device}")
        self._loop = loop if loop is not None else asyncio.get_event_loop()
        self._output_buffer = bytes()
        self._output_buffer_empty = None
        self._output_buffer_lock = threading.Lock() if self._use_threads else None

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self._device}>'

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def write(self, data):
        if self._use_threads:
            with self._output_buffer_lock:
                self._output_buffer += data
                if self._output_buffer_empty is None:
                    self._output_buffer_empty = self._loop.run_in_executor(None, self._write_blocking)
            return
        if not self._output_buffer:
            try:
                nbytes = self._connection.write(data)
            except serial.SerialTimeoutException:
                nbytes = 0
            except Exception:
                Log.exception("Error writing to stream")
                raise
            if nbytes:
                Log.debug(f"Write {data[:nbytes]!r}")
            self._output_buffer = data[nbytes:]
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
            nbytes = self._connection.write(self._output_buffer)
        except serial.SerialTimeoutException:
            nbytes = 0
        except Exception as exc:
            Log.exception("Error writing to stream")
            self._output_buffer_empty.set_exception(exc)
            self._loop.remove_writer(self._connection)
        if nbytes:
            Log.debug(f"Write {self._output_buffer[:nbytes]!r}")
            self._output_buffer = self._output_buffer[nbytes:]
        if not self._output_buffer:
            self._loop.remove_writer(self._connection)
            self._output_buffer_empty.set_result(None)
            self._output_buffer_empty = None

    def _write_blocking(self):
        with self._output_buffer_lock:
            while self._output_buffer:
                data = bytes(self._output_buffer)
                self._output_buffer_lock.release()
                try:
                    nbytes = self._connection.write(data)
                finally:
                    self._output_buffer_lock.acquire()
                Log.debug(f"Write {self._output_buffer[:nbytes]!r}")
                self._output_buffer = self._output_buffer[nbytes:]
            self._output_buffer_empty = None

    async def read(self, nbytes=None):
        if self._use_threads:
            return await self._loop.run_in_executor(None, self._read_blocking, nbytes)
        while True:
            nwaiting = self._connection.in_waiting
            if nwaiting:
                data = self._connection.read(nwaiting if nbytes is None else min(nbytes, nwaiting))
                Log.debug(f"Read {data!r}")
                return data
            future = self._loop.create_future()
            self._loop.add_reader(self._connection, future.set_result, None)
            try:
                await future
            finally:
                self._loop.remove_reader(self._connection)

    def _read_blocking(self, nbytes=None):
        data = self._connection.read(1)
        nwaiting = self._connection.in_waiting
        if nwaiting and (nbytes is None or nbytes > 1):
            data += self._connection.read(nwaiting if nbytes is None else min(nbytes-1, nwaiting))
        Log.debug(f"Read {data!r}")
        return data

    async def readexactly(self, nbytes):
        data = b''
        while len(data) < nbytes:
            data += await self.read(nbytes-len(data))
        return data


class UDPBitscope:

    class UDPProtocol(asyncio.Protocol):

        def __init__(self):
            print("creating UDPTransport...")
            self.transport = None

            # link between callback and asyncio
            self.msg_received = None
            self.datagram_buffer = None        # buffer of received data

        # BaseProtocol
        def connection_made(self, transport: transports.BaseTransport) -> None:
            self.transport = transport

        # Base Protocol
        # def connection_lost(self, exc):
        #     pass

        # Datagram.Protocol
        def error_received(self, exec):
            print("Error received (?!)")

        # Datagram.Protocol
        def datagram_received(self, data, addr):
            """ callback when data is received from UDP connection """
            Log.debug(f"reçu ({addr}): {data}")
            self.datagram_buffer = data
            # update concurrent tasks
            self.msg_received.set()

    @classmethod
    async def create(cls, host, port):
        # initiate connection
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(cls.UDPProtocol, remote_addr=(host, port))
        return UDPBitscope(transport, protocol)

    def __init__(self, transport, protocol):
        # initiate connection
        self.transport, self.protocol = transport, protocol
        Log.debug(f"Opened bitscope connection to UDP:{transport._address[0]}:{transport._address[1]}")

        # link between data_received callback and asyncio
        self.msg_received = asyncio.Event()
        self.protocol.msg_received = self.msg_received
        self.data_buffer = None

        # bitscope-server counter
        self.cmd_counter = 0

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.transport}>'

    def close(self):
        self.transport.close()
        self.transport = None
        self.protocol = None

    def write(self, data: bytes):
        """
        Send commands to bitscope-server
        structure of datagram =
            1 byte = cmd number
            X bytes = cmd_id and eventually data (handled by vm implementation)
        """
        # add nb of sended cmd to the data
        msg = bytes([self.cmd_counter]) + data

        # send data
        self.transport.sendto(msg)       # This method does not block; it buffers the data and arranges for it to be sent out asynchronously.

        # increment cmd_counter (limit 1 byte --> 0 to 255)
        self.cmd_counter += 1
        if self.cmd_counter > 255:
            self.cmd_counter = 0

    async def drain(self):
        # useless for Datagram Transport (I suppose)
        pass

    async def read(self, nbytes=None):
        if self.data_buffer:
            # on garde que les nbytes et on stock le reste (ou on envoie tout)
            if nbytes:
                split_index = nbytes
            else:
                split_index = len(self.data_buffer)
            returned_data = self.data_buffer[:split_index]
            store_data = self.data_buffer[split_index:]
            if store_data:
                self.data_buffer = store_data
            else:
                self.data_buffer = None

        else:
            # We need to wait for the callback
            await self.msg_received.wait()

            # Ok, callback has fired, data has been received
            # retrieve it
            raw_data = self.protocol.datagram_buffer

            # reset event and clear data_buffer
            self.protocol.datagram_buffer = None
            self.msg_received.clear()

            # Now remove the 4 bytes of header from bitscope-server
            data = raw_data[4:]

            # keep only first nbytes (and store the rest) or all
            if nbytes:
                split_index = nbytes
            else:
                split_index = len(data)
            returned_data = data[:split_index]
            store_data = data[split_index:]
            if store_data:
                self.data_buffer = store_data

        return returned_data

    async def readexactly(self, nbytes):
        # call read with amount of bytes requested
        data_buffer = b''
        while len(data_buffer) < nbytes:
            data_buffer += await self.read(nbytes-len(data_buffer))

        return data_buffer
