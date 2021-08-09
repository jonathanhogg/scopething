
# ScopeThing

## Quick Notes

- Only tested against a BitScope Micro (BS05)
- Requires Python >3.6 and the **pyserial** package
- Also requires **NumPy** and **SciPy** if you want to do analog range calibration
- Having **Pandas** is useful for wrapping capture results for further processing

## Longer Notes

### Why have I written this?

BitScope helpfully provide applications and libraries for talking to their USB
capture devices, so one can just use those. I wrote this code because I want
to be able to grab the raw data and further process it in various ways. I'm
accustomed to working at a Python prompt with various toolkits like SciPy and
Matplotlib, so I wanted a simple way to just grab a trace as an array.

The BitScope library is pretty simple to use, but also requires you to
understand a fair amount about how the scope works and make a bunch of decisions
about what capture mode and rate to use. I want to just specify a time period,
voltage range and a rough number of samples and have all of that worked out for
me, same way as I'd use an actual oscilloscope: twiddle the knobs and look at
the trace.

Of course, I could have wrapped the BitScope library in something that would do
this, but after reading a bit about how the scope works I was fascinated with
understanding it further and so I decided to go back to first principles and
start with just talking to it with a serial library. This code thus serves as
(sort of) documentation for the VM registers, capture modes and how to use them.
It also has the advantage of being pure Python.

The code prefers the highest capture resolution possible and will do the mapping
from high/low/trigger voltages to the mysterious magic numbers that the device
needs. It can also do logic and mixed-signal capture.

In addition to capturing, the code can also generate waveforms at arbitrary
frequencies – something that is tricky to do as the device operates at specific
frequencies and so one has to massage the width of the waveform buffer to get a
frequency outside of these. It can also control the clock generator.

I've gone for an underlying async design as it makes it easy to integrate the
code into UI programs or network servers – both of which interest me as the end
purpose for this code. However, for shell use there are synchronous wrapper
functions. Of particular note is that the synchronous wrapper understands
keyboard interrupt and will cancel a capture returning the trace around the
cancel point. This is useful if your trigger doesn't fire and you want to
understand why.

### Where's the documentation, mate?

Yeah, yeah. I know.

### Also, I see no unit tests...

It's pretty hard to do unit tests for a physical device. That's my excuse and
I'm sticking to it.

### Long lines and ignoring E221, eh?

"A foolish consistency is the hobgoblin of little minds"

Also, I haven't used an 80 character wide terminal in this century.

### connection scheme

Options for launching the scope to connect to different device location

None
--> looks for local / USB bitscope

socket:///dev/ttyUSB0
--> specify which device to use (for example if you have several bitscopes connected)

bitscope-server://192.168.20.25:15500
--> bitscope is behind a bitscope-server located on ip:port