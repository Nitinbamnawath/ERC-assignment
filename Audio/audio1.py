import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

fs, raw = wav.read('corrupted.wav')

signal = raw.astype(float)

if len(signal.shape) > 1:
    signal = signal[:, 0]

N = len(signal)
time = np.arange(N) / fs

carrier_freq = 7200

mixed = signal * np.cos(2 * np.pi * carrier_freq * time)

fft_data = np.fft.fft(mixed)
freq_axis = np.fft.fftfreq(N, 1/fs)

cutoff = 4000
lp_mask = np.abs(freq_axis) <= cutoff

fft_data = fft_data * lp_mask

def notch_out(arr, f0, freq_arr, bw=30):
    keep_mask = np.abs(np.abs(freq_arr) - f0) > bw
    result = arr * keep_mask
    return result

fft_data = notch_out(fft_data, 1200, freq_axis)
fft_data = notch_out(fft_data, 2300, freq_axis)
fft_data = notch_out(fft_data, 4000, freq_axis)


reconstructed = np.fft.ifft(fft_data)

reconstructed = reconstructed.real

max_amp = np.max(np.abs(reconstructed))

if max_amp == 0:
    max_amp = 1

norm_audio = reconstructed / max_amp

out_audio = np.int16(norm_audio * 32767)

wav.write('final.wav', fs, out_audio)
print("Saved file")

plt.figure(figsize=(10, 5))
plt.plot(freq_axis, np.abs(fft_data))
plt.title("Spectrum")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 5000)
plt.grid(True)
plt.tight_layout()
plt.savefig('somthing.png')
print("Saved plot")