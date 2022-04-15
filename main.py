# import statements
import sys
import matplotlib.pyplot as plt   # plotting 
import numpy as np                # all of numpy...
del sys.modules["numpy"].fft      # ... except FFT helpers

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

    return y if s == -1 else y/N


def naive_iDFT(inSignal):
    """
    Naive implementation of the inverse discrete Fourier transform.
    This function is ported from Assignment 4 of the course.

    Leverages the s property of the naive_DFT function.

    Since this function calles the naive_DFT function,
    The time complexity of this function is O(N^2).
    Where N is the length of the discrete input signal.
    """
    return naive_DFT(inSignal, s = 1)


def fast_DFT(inSignal, s: int = -1):
    """
    Fast implementation of the discrete Fourier transform.

    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.

    This function uses the Cooley-Tukey algorithm.
    """
    # TODO: Implement the fast DFT algorithm
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    if N == 1:
        return inSignal
    else:
        yeven = FFT(inSignal[0:N:2])
        yodd = FFT(inSignal[1:N:2])
        w = np.exp(-2j*np.pi*np.arange(N)/N)
        fe = np.tile(yeven,(2))
        fo = np.tile(yodd,(2))
        y = fe + w*fo 
    return y


def fast_iDFT(inSignal):
    """
    Fast implementation of the inverse discrete Fourier transform.

    Since this function calles the fast_DFT function,
    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.
    """
    return fast_DFT(inSignal, s = 1)


if __name__ == "__main__":
    ...
