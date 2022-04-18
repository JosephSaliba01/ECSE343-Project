import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

x = np.arange(-500, 501, 1)
X, Y = np.meshgrid(x, x)
wavelength = 200

for angle in np.arange(0, 360, 1):
    grating = np.sin(
        2*np.pi*(1*X*np.cos(angle) + 1*Y*np.sin(angle)) / wavelength
    )
    plt.set_cmap("gray")
    plt.subplot(121)
    plt.imshow(grating)
    # Calculate Fourier transform of grating
    ft = np.fft.ifftshift(grating)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    plt.subplot(122)
    plt.imshow(abs(ft))
    plt.xlim([480, 520])
    plt.ylim([520, 480])  # Note, order is reversed for y
    plt.title("Angle: " + str(angle))
    plt.pause(0.005)

plt.show()