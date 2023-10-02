
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# problème 1
# -----------------------
x = np.asarray([1, -1])
h = np.asarray(([1, 1, 2, 2]))
y = np.convolve(x, h)
# print(y)

# problème 3
# -----------------------
x = np.asarray([1, 2, 0])
h = np.asarray([-3, 1, 0])

X = np.fft.fft(x)
H = np.fft.fft(h)

Y = X * H
y = np.fft.ifft(Y)
# print(y)

# dernier prob
# a)
N = 64
fc = 2000
Fe = 16000
n = np.arange(-int(N/2), int(N/2))

K = ((2 * fc * N) / Fe) + 1
print("K : ", K)

h = (1 / N) * (np.sin(np.pi * K * n / N)) / (np.sin(np.pi * n / N) + 1e-20)

h[int(N/2)] = K/N

plt.stem(n, h)
plt.title("h")
plt.show()

# b)
m = (2 * np.pi * n) / N
H = np.fft.fft(h)
Hshift = np.fft.fftshift(H)

plt.subplot(2,2,1)
plt.stem(m, np.abs(Hshift))
plt.title("Module de H (freq normal)")

plt.subplot(2,2,2)
plt.stem(m, np.angle(Hshift))
plt.title("Phase de H (freq normal)")

w, H = signal.freqz(h, worN=int(N/2))

plt.subplot(2,2,3)
plt.stem(w, np.abs(H))
plt.title("amplitude avec frqz de h")

plt.subplot(2,2,4)
plt.stem(w, np.angle(H))
plt.title("phase avec frqz de h")
plt.show()

# c)
hamming_window = np.hamming(N)
signal_windowed = h * hamming_window

H = np.fft.fft(signal_windowed)
Hshift = np.fft.fftshift(H)


plt.subplot(3,1,1)
plt.stem(m, signal_windowed)
plt.title("Fenêtre de Hamming")

plt.subplot(3,1,2)
plt.stem(m, np.abs(Hshift))
plt.title("Module de H (freq normal)")

plt.subplot(3,1,3)
plt.stem(m, np.angle(Hshift))
plt.title("Phase de H (freq normal)")
plt.show()

# d)
n2 = np.arange(0, 128)

x = np.sin(2*np.pi*200*n2/Fe) + 0.25*np.sin(2*np.pi*n2*3000/Fe)
y = np.convolve(x, h)

plt.plot(np.arange(0, len(y)), y)
plt.title("convolution")
plt.show()


