# import statements
import timeit
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt   # plotting 
import numpy as np                # all of numpy...
# del sys.modules["numpy"].fft      # ... except FFT helpers

def naive_DFT(inSignal, s: int = -1):
    """
    Naive implementation of the discrete Fourier transform.
    This function is ported from Assignment 4 of the course.

    The time complexity of this function is O(N^2)
    Where N is the length of the discrete input signal.
    """
    y = np.zeros(inSignal.shape, dtype=complex)
    N = inSignal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))

    x = 2 * np.pi * s * k * n / N
    M = np.cos(x) + 1j * np.sin(x)

    y = np.dot(M, inSignal)

    return y


def naive_iDFT(inSignal):
    """
    Naive implementation of the inverse discrete Fourier transform.
    This function is ported from Assignment 4 of the course.

    Leverages the s property of the naive_DFT function.

    Since this function calles the naive_DFT function,
    The time complexity of this function is O(N^2).
    Where N is the length of the discrete input signal.
    """
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    inSignal = inSignal/N
    y = naive_DFT(inSignal,1)
    return y


def fast_DFT(inSignal, s: int = -1):
    """
    Fast implementation of the discrete Fourier transform.

    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.

    This function uses the Cooley-Tukey algorithm.
    """
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    if N <= 32:
        return naive_DFT(inSignal,s)
    else:
        yeven = fast_DFT(inSignal[0:N:2],s) #find FFT at even indices
        yodd = fast_DFT(inSignal[1:N:2],s) #find FFTat odd indices
        w = np.exp(s*2j*np.pi*np.arange(N)/N)
        fe = np.resize(yeven,N) #sampled values of the DFT are N-periodic
        fo = np.resize(yodd,N) 
        y = fe + w*fo 
    return y 


def fast_iDFT(inSignal):
    """
    Fast implementation of the inverse discrete Fourier transform.

    Since this function calls the fast_DFT function,
    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.
    """
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    inSignal = inSignal/N
    y = fast_DFT(inSignal, 1)
    return y


def naive_DFT2D(inSignal2D, s: int = -1):
    x = np.apply_along_axis(naive_DFT, 0, inSignal2D, s)
    y = np.apply_along_axis(naive_DFT, 1, x, s)
    return y


def naive_iDFT2D(inSignal2D: complex):
    y = np.zeros(inSignal2D.shape, dtype = complex)
    N = inSignal2D.shape[0]
    inSignal2D = inSignal2D/(N**2)
    y = naive_DFT2D(inSignal2D,1)
    return y


def fast_DFT2D(inSignal2D, s: int = -1):
    x = np.apply_along_axis(fast_DFT, 0, inSignal2D, s)
    y = np.apply_along_axis(fast_DFT, 1, x, s)
    return y


def fast_iDFT2D(inSignal2D: complex):
    y = np.zeros(inSignal2D.shape, dtype = complex)
    N = inSignal2D.shape[0]
    inSignal2D = inSignal2D/(N**2)
    y = fast_DFT2D(inSignal2D,1)
    return y


def plot_discrete(*signals, title=""):
    """
    Helper function to plot the discrete signals.
    """
    fig, axs = plt.subplots(len(signals))

    for i, signal in enumerate(signals):
        axs[i].stem(signal)

    plt.suptitle(title)
    plt.show()


def main():
    """
    Main method.
    """
    print("[Part 1.a]: Benchmarking FFT performance against a naive DFT for increasing N values.")
    DFTplot = np.array([])
    FFTplot = np.array([])
    Naxis = np.array([])

    for m in range(13): 
        N = 2**m
        Naxis = np.append(Naxis,N)
        x = np.random.randn(N) + np.random.randn(N)*1j

        start_time = timeit.default_timer()
        y = naive_DFT(x)
        DFTtime = timeit.default_timer() - start_time ##time it takes for DFT for a given N
        DFTplot = np.append(DFTplot, DFTtime)
        assert np.allclose(y, np.fft.fft(x)) # assert correctness of naive DFT

        start_time = timeit.default_timer()
        y = fast_DFT(x)
        FFTtime = timeit.default_timer() - start_time ##time it takes for FFT for a given N
        FFTplot = np.append(FFTplot, FFTtime)
        assert np.allclose(y, np.fft.fft(x)) # assert correctness of fast DFT

    plt.plot(Naxis,DFTplot)
    plt.plot(Naxis,FFTplot)
    plt.legend(['naive DFT','fast DFT'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT/FFT as a function of N')
    plt.show()
    
    print("[Part 1.b]: Benchmarking FFT-2D performance against a naive DFT-2D for increasing N values.")
    DFT2plot = np.array([])
    FFT2plot = np.array([])
    Naxis = np.array([])

    for m in range(10): 
        N = 2**m
        Naxis = np.append(Naxis,N)
        x = np.random.randn(N) + np.random.randn(N)*1j
        z = np.zeros((N,N), dtype=complex)
        z[:] = x

        start_time = timeit.default_timer()
        y = naive_DFT2D(z)
        DFT2time = timeit.default_timer() - start_time ##time it takes for DFT for a given N
        DFT2plot = np.append(DFT2plot, DFT2time)

        start_time = timeit.default_timer()
        y = fast_DFT2D(z)
        FFT2time = timeit.default_timer() - start_time ##time it takes for DFT for a given N
        FFT2plot = np.append(FFT2plot, FFT2time)

    plt.plot(Naxis, DFT2plot)
    plt.plot(Naxis,FFT2plot)
    plt.legend(['naive DFT-2D','fast DFT-2D'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT-2D/FFT-2D as a function of N')
    plt.show()

    print("[Part 2]: Benchmarking 2D FFT performance against a naive 2D DFT in image application.")
    print("Example 1: Goose image.")

    goose_clean = np.load('img/goose.npy')
    goose_noise = np.load("img/goose-noise.npy")

    fig, axs = plt.subplots(2,3)

    ft_goose_clean = fast_DFT2D(goose_clean)
    ft_goose_clean = np.fft.fftshift(ft_goose_clean)

    ft_goose_noise = fast_DFT2D(goose_noise)
    ft_goose_noise = np.fft.fftshift(ft_goose_noise)

    ft_goose_filtered = fast_DFT2D(goose_noise)
    ft_goose_filtered = np.fft.fftshift(ft_goose_filtered)

    def blank(M, y, x):
        M[x:x+1, y:y+1] = 0

    blank(ft_goose_filtered, 240, 256)
    blank(ft_goose_filtered, 272, 256)
    blank(ft_goose_filtered, 251, 251)
    blank(ft_goose_filtered, 261, 261)

    goose_filtered = fast_iDFT2D(np.fft.ifftshift(ft_goose_filtered)).real

    axs[0, 0].imshow(goose_clean, plt.get_cmap('gray'))
    axs[0, 1].imshow(goose_noise, plt.get_cmap('gray'))
    axs[0, 2].imshow(goose_filtered, plt.get_cmap('gray'))

    axs[0, 0].set_title("Clean goose image.")
    axs[0, 1].set_title("Noisy goose image.")
    axs[0, 2].set_title("Filtered goose image.")

    axs[1, 0].imshow(np.log(abs(ft_goose_clean) + .01), plt.get_cmap('gray'))
    axs[1, 1].imshow(np.log(abs(ft_goose_noise) + .01), plt.get_cmap('gray'))
    axs[1, 2].imshow(np.log(abs(ft_goose_filtered) + .01), plt.get_cmap('gray'))

    axs[1, 0].set_xlim([231, 281])
    axs[1, 0].set_ylim([281, 231])

    axs[1, 1].set_xlim([231, 281])
    axs[1, 1].set_ylim([281, 231])

    axs[1, 2].set_xlim([231, 281])
    axs[1, 2].set_ylim([281, 231])

    axs[1, 0].set_title("Clean goose image FT spectrum (Zoomed in).")
    axs[1, 1].set_title("Noisy goose image FT spectrum (Zoomed in).")
    axs[1, 2].set_title("Filtered goose image FT spectrum (Zoomed in).")
    plt.show()

    print("Example 2: Lenna.")

    lenna_clean = np.load('img/lenna.npy')
    lenna_noise = np.load("img/lenna-noise.npy")

    fig, axs = plt.subplots(2,3)

    ft_lenna_clean = fast_DFT2D(lenna_clean)
    ft_lenna_clean = np.fft.fftshift(ft_lenna_clean)

    ft_lenna_noise = fast_DFT2D(lenna_noise)
    ft_lenna_noise = np.fft.fftshift(ft_lenna_noise)

    ft_lenna_filtered = fast_DFT2D(lenna_noise)
    ft_lenna_filtered = np.fft.fftshift(ft_lenna_filtered)

    ft_lenna_filtered[256, :] = 0
    ft_lenna_filtered[256, :] = 0

    ft_lenna_filtered[:, 256] = 0
    ft_lenna_filtered[:, 256] = 0

    lenna_filtered = fast_iDFT2D(np.fft.ifftshift(ft_lenna_filtered)).real

    axs[0, 0].imshow(lenna_clean, plt.get_cmap('gray'))
    axs[0, 1].imshow(lenna_noise, plt.get_cmap('gray'))
    axs[0, 2].imshow(lenna_filtered, plt.get_cmap('gray'))

    axs[0, 0].set_title("Clean lenna image.")
    axs[0, 1].set_title("Noisy lenna image.")
    axs[0, 2].set_title("Filtered lenna image.")

    axs[1, 0].imshow(np.log(abs(ft_lenna_clean) + .01), plt.get_cmap('gray'))
    axs[1, 1].imshow(np.log(abs(ft_lenna_noise) + .01), plt.get_cmap('gray'))
    axs[1, 2].imshow(np.log(abs(ft_lenna_filtered) + .01), plt.get_cmap('gray'))

    axs[1, 0].set_title("Clean lenna image FT spectrum (Zoomed in).")
    axs[1, 1].set_title("Noisy lenna image FT spectrum (Zoomed in).")
    axs[1, 2].set_title("Filtered lenna image FT spectrum (Zoomed in).")
    plt.show()


if __name__ == "__main__":
    main()
