# import statements
import sys
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
    if N < 32:
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
    return naive_DFT(naive_DFT(inSignal2D.T, s).T, s)


def naive_iDFT2D(inSignal2D: complex):
    return naive_DFT2D(inSignal2D, s = 1)


def fast_DFT2D(inSignal2D, s: int = -1):
    return fast_DFT(fast_DFT(inSignal2D.T, s).T, s)


def fast_iDFT2D(inSignal2D: complex):
    return fast_DFT2D(inSignal2D, s = 1)


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
    
    print("[Part 1]: Benchmarking FFT performance against a naive DFT for increasing N values.")

    ##benchmark FFTs performance against a naive DFT for increasingly large N
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


    print("[Part 2a]: Benchmarking FFT performance against a naive DFT in audio application.")

    # Benchmark using Falcon voice sample
    unedited_waveform = read("audio/Falcon-sound-unedited.wav")[1].T[0]  # read in the audio file
    falcon_DFT_times, falcon_FFT_times = {}, {}

    for m in range(14):
        N = 2**m
        x = unedited_waveform[:N]

        start_time = timeit.default_timer()
        y_naive_DFT = naive_DFT(x)
        falcon_DFT_times[N] = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        y_fast_DFT = fast_DFT(x)
        falcon_FFT_times[N] = timeit.default_timer() - start_time

        assert np.allclose(y_naive_DFT, np.fft.fft(x))
        assert np.allclose(y_fast_DFT, np.fft.fft(x))

    print("[Part 2b]: Benchmarking 2D FFT performance against a naive 2D DFT in image application.")
    bird_img = np.load("img/bird.npy")

    plt.imshow(bird_img, plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.title = "Bird image"
    plt.show()

    fast_ft_bird = fast_DFT2D(bird_img)
    naive_ft_bird = naive_DFT2D(bird_img)

    assert np.allclose(naive_ft_bird, np.fft.fft2(bird_img))
    assert np.allclose(fast_ft_bird, np.fft.fft2(bird_img))  # Doesn't work for some reason

    plt.imshow(fast_ft_bird.real, plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.show()

if __name__ == "__main__":
    main()
