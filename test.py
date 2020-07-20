
from pylab import figure, plot, show

from analysis import annotate_series
from scope import await_, capture, main


await_(main())
series = capture(['A'], period=20e-3, nsamples=2000).A

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

    plot(waveform.timestamps + waveform.capture_start - series.capture_start,
         waveform.samples * waveform.amplitude + waveform.offset)

show()
