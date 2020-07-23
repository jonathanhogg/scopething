"""
analysis
========

Library code for analysing captures returned by `Scope.capture()`.
"""

# pylama:ignore=C0103,R1716

import numpy as np

from utils import DotDict


def interpolate_min_x(f, x):
    return 0.5 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x


def rms(f):
    return np.sqrt((f ** 2).mean())


def sine_wave(n):
    return np.sin(np.linspace(0, 2*np.pi, n, endpoint=False))


def triangle_wave(n):
    x = np.linspace(0, 4, n, endpoint=False)
    x2 = x % 2
    y = np.where(x2 < 1, x2, 2 - x2)
    y = np.where(x // 2 < 1, y, -y)
    return y


def square_wave(n, duty=0.5):
    w = int(n * duty)
    return np.hstack([np.ones(w), -np.ones(n - w)])


def sawtooth_wave(n):
    return 2 * (np.linspace(0.5, 1.5, n, endpoint=False) % 1) - 1


def moving_average(samples, width, mode='wrap'):
    hwidth = width // 2
    samples = np.take(samples, np.arange(-hwidth, len(samples)+width-hwidth), mode=mode)
    cumulative = samples.cumsum()
    return (cumulative[width:] - cumulative[:-width]) / width


def calculate_periodicity(series, window=0.1):
    samples = np.array(series.samples, dtype='double')
    window = int(len(samples) * window)
    errors = np.zeros(len(samples) - window)
    for i in range(1, len(errors) + 1):
        errors[i-1] = rms(samples[i:] - samples[:-i])
    threshold = errors.max() / 2
    minima = []
    for i in range(window, len(errors) - window):
        p = errors[i-window:i+window].argmin()
        if p == window and errors[p + i - window] < threshold:
            minima.append(interpolate_min_x(errors, i))
    if len(minima) <= 1:
        return None
    ks = np.polyfit(np.arange(0, len(minima)), minima, 1)
    return ks[0] / series.sample_rate


def extract_waveform(series, period):
    p = int(round(series.sample_rate * period))
    n = len(series.samples) // p
    if n <= 2:
        return None, None, None, None
    samples = np.array(series.samples)[:p*n]
    cumsum = samples.cumsum()
    underlying = (cumsum[p:] - cumsum[:-p]) / p
    n -= 1
    samples = samples[p//2:p*n + p//2] - underlying
    wave = np.zeros(p)
    for i in range(n):
        o = i * p
        wave += samples[o:o+p]
    wave /= n
    return wave, p//2, n, underlying


def normalize_waveform(samples, smooth=7):
    n = len(samples)
    smoothed = moving_average(samples, smooth)
    scale = (smoothed.max() - smoothed.min()) / 2
    offset = (smoothed.max() + smoothed.min()) / 2
    smoothed -= offset
    last_rising = first_falling = None
    crossings = []
    for i in range(n):
        if smoothed[i-1] < 0 and smoothed[i] > 0:
            last_rising = i
        elif smoothed[i-1] > 0 and smoothed[i] < 0:
            if last_rising is None:
                first_falling = i
            else:
                crossings.append((i - last_rising, last_rising))
    if first_falling is not None:
        crossings.append((n + first_falling - last_rising, last_rising))
    first = min(crossings)[1]
    wave = (np.hstack([samples[first:], samples[:first]]) - offset) / scale
    return wave, offset, scale, first, sorted((i - first % n, w) for (w, i) in crossings)


def characterize_waveform(samples, crossings):
    n = len(samples)
    possibles = []
    if len(crossings) == 1:
        duty_cycle = crossings[0][1] / n
        if 0.45 < duty_cycle < 0.55:
            possibles.append((rms(samples - sine_wave(n)), 'sine', None))
            possibles.append((rms(samples - triangle_wave(n)), 'triangle', None))
            possibles.append((rms(samples - sawtooth_wave(n)), 'sawtooth', None))
        possibles.append((rms(samples - square_wave(n, duty_cycle)), 'square', duty_cycle))
    possibles.sort()
    return possibles


def annotate_series(series):
    period = calculate_periodicity(series)
    if period is not None:
        waveform = DotDict(period=period, frequency=1 / period)
        wave, start, count, underlying = extract_waveform(series, period)
        wave, offset, scale, first, crossings = normalize_waveform(wave)
        waveform.samples = wave
        waveform.beginning = start + first
        waveform.count = count
        waveform.amplitude = scale
        waveform.offset = underlying.mean() + offset
        waveform.timestamps = np.arange(len(wave)) * series.sample_period
        waveform.sample_period = series.sample_period
        waveform.sample_rate = series.sample_rate
        waveform.capture_start = series.capture_start + waveform.beginning * series.sample_period
        possibles = characterize_waveform(wave, crossings)
        if possibles:
            error, shape, duty_cycle = possibles[0]
            waveform.error = error
            waveform.shape = shape
            if duty_cycle is not None:
                waveform.duty_cycle = duty_cycle
        else:
            waveform.shape = 'unknown'
        series.waveform = waveform
        return True
    return False
