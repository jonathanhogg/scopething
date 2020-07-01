
import numpy as np
from pylab import figure, plot, show

from analysis import annotate_series
from scope import await_, capture, main
from utils import DotDict


await_(main())

# o = 400
# m = 5
# n = o * m
# samples = square_wave(o)
# samples = np.hstack([samples] * m) * 2
# samples = np.hstack([samples[100:], samples[:100]])
# samples += np.random.normal(size=n) * 0.1
# samples += np.linspace(4.5, 5.5, n)
# series = DotDict(samples=samples, sample_rate=1000000)

data = capture(['A'], period=20e-3, nsamples=2000)
series = data.A

figure(1)
plot(series.timestamps, series.samples)

if annotate_series(series):
    waveform = series.waveform
    if 'duty_cycle' in waveform:
        print(f"Found {waveform.frequency:.0f}Hz {waveform.shape} wave, "
              f"with duty cycle {waveform.duty_cycle * 100:.0f}%, "
              f"amplitude ±{waveform.amplitude:.1f}V and offset {waveform.offset:.1f}V")
    else:
        print(f"Found {waveform.frequency:.0f}Hz {waveform.shape} wave, "
              f"with amplitude ±{waveform.amplitude:.2f}V and offset {waveform.offset:.2f}V")

    plot(waveform.timestamps + waveform.capture_start - series.capture_start, waveform.samples * waveform.amplitude + waveform.offset)

show()
