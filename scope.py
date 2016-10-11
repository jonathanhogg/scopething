
import asyncio
from streams import SerialStream


class Scope(object):

    def __init__(self, stream):
        self._stream = stream

    async def reset(self):
        await self._stream.write(b'!')
        await self._stream.readuntil(b'!')

    async def get_revision(self):
        await self._stream.write(b'?')
        assert await self._stream.readuntil(b'\r') == b'?\r'
        revision = await self._stream.readuntil(b'\r')
        return revision.decode('ascii').strip()



async def main():
    s = Scope(SerialStream())
    await s.reset()
    print(await s.get_revision())

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

