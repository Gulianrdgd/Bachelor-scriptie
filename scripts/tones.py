import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import wave

# p = pyaudio.PyAudio()

volume = 1     # range [0.0, 1.0]
fs = 96000       # sampling rate, Hz, must be integer
duration = 3.0   # in seconds, may be float
f1 = 38000        # sine frequency, Hz, may be float
f2 = 40000

# generate samples, note conversion to float32 array
each_sample_number = np.arange(duration * fs)
samplesL = (np.sin(2*np.pi*np.arange(fs*duration)*f1/fs)).astype(np.float32)
samplesR = (np.sin(2*np.pi*np.arange(fs*duration)*f2/fs)).astype(np.float32)

N = round(fs * duration)
yf1 = np.fft.rfft(samplesL)
yf2 = np.fft.rfft(samplesR)
xf = np.fft.rfftfreq(N, 1 / fs)

audio = np.array([samplesL, samplesR]).T
audio = (audio * (2 ** 15 - 1)).astype("<h")


plt.plot(xf, np.abs(yf1))
plt.plot(xf, np.abs(yf2))
plt.show()


with wave.open("sound1.wav", "w") as f:
    # 2 Channels.
    f.setnchannels(2)
    # 2 bytes per sample.
    f.setsampwidth(2)
    f.setframerate(fs)
    f.writeframes(audio.tobytes())
# # for paFloat32 sample values must be in range [-1.0, 1.0]
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=2,
#                 rate=fs,
#                 output=True)
#
# # play. May repeat with different volume values (if done interactively)
# stream.write(volume*interleavedSamples)
#
# stream.stop_stream()
# stream.close()

# p.terminate()

# plt.plot(xf, np.abs(yf))
# plt.show()
