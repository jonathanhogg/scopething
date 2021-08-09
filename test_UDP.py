
from pylab import figure, plot, show

from analysis import annotate_series
from scope import await_, capture, main

"""
Launch with 
python test_UDP.py bitscope-server://192.168.20.25:15500
"""

await_(main())

# le plus rapide possible
# max nsamples = 12 << 10 = 12288
# ech mini = 50 ns
# period full = 614 400 ns
series = capture(['A'], period=580e-6, nsamples=12000, timeout=50e-3).A

# 5 MHz pour avoir un prétrig
# Tech = 0,2 µs
# max nsamples = 12 << 10 = 12288
# period full = 2457,6 µs
# series = capture(['A'], period=2457.6e-6, nsamples=12288, timeout=50e-3, trigger_position=0.3, hair_trigger=False).A

figure(1)
plot(series.timestamps, series.samples)

show()
