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

    Since this function calles the fast_DFT function,
    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.
    """
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    inSignal = inSignal/N
    y = fast_DFT(inSignal, 1)
    return y


# Helper functions

def plot_discrete(*signals, title=""):
    """
    Plots the input signals.
    """
    fig, axs = plt.subplots(len(signals))

    for i, signal in enumerate(signals):
        axs[i].stem(signal)

    plt.suptitle(title)
    plt.show()


def discrete_sin(num_period, step):
    """
    Generates a sinusoid signal with the given amount of period and step.
    """
    step = step * np.pi
    return np.sin(np.arange(0, num_period * 2 * np.pi + step, step))


def main():
    """
    Main method.
    """
    # y = discrete_sin(4, 0.125)
    # y_DFT = naive_DFT(y)
    # plot_discrete(y, y_DFT.real, y_DFT.imag, title="Naive plotting test")
    
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

    # Benchmark using Falcon voice sample

    waveform = read("audio/Falcon-sound-2.wav")[1].T[0]  # read in the audio file
    
    start_time = timeit.default_timer()
    y = naive_DFT(waveform)
    DFTtime = timeit.default_timer() - start_time
    plot_discrete(waveform, y.real, y.imag, title="Falcon voice sample using naive DFT")

    start_time = timeit.default_timer()
    y = fast_DFT(waveform)
    FFTtime = timeit.default_timer() - start_time
    plot_discrete(waveform, y.real, y.imag, title="Falcon voice sample using fast DFT")

    start_time = timeit.default_timer()
    y = np.fft.fft(waveform)
    npFFTtime = timeit.default_timer() - start_time
    plot_discrete(waveform, y.real, y.imag, title="Falcon voice sample using numpy.fft.fft")

    print("Naive DFT: ", DFTtime)
    print("Fast DFT: ", FFTtime)
    print("Numpy FFT: ", npFFTtime)

    assert np.allclose(fast_DFT(waveform), np.fft.fft(waveform))

if __name__ == "__main__":
    main()
