import numpy as np
import matplotlib.pyplot as plt
# Labo 1 - APP 3

# A)
N = 32
n = np.arange(0, N)
m = np.arange(-N/2, N/2)

# init signal
x1 = np.sin(0.1 * np.pi * n + np.pi / 4)
x2 = (-1) ** n
x3 = np.zeros((N,), )
x3[10] = 1.0

# FFT
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

plt.subplot(3, 1, 1)
plt.stem(n, np.abs(X1))
plt.title('module de X1')

plt.subplot(3, 1, 2)
plt.stem(n, np.abs(X2))
plt.title('module de X2')

plt.subplot(3, 1, 3)
plt.stem(n, np.abs(X3))
plt.title('module de X3')

plt.show()

plt.subplot(3, 1, 1)
plt.stem(n, np.angle(X1))
plt.title('phase de X1')

plt.subplot(3, 1, 2)
plt.stem(n, np.angle(X2))
plt.title('phase de X2')

plt.subplot(3, 1, 3)
plt.stem(n, np.angle(X3))
plt.title('phase de X3')

plt.show()

# b) freq normalisée
w = 2*np.pi*m/N
X1 = np.fft.fftshift(x1)
plt.stem(w, np.abs(X1))
plt.title('freq normalisée module de X1')
plt.show()


# application fenêtre C)
# window
w = np.hanning(N)

Xw1 = np.fft.fft(w*x1)

plt.stem(n, np.abs(Xw1))
plt.title('fenêtre de X1')
plt.show()






