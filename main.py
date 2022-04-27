# import statements
import sys                        
import timeit                     # for timing
import matplotlib.pyplot as plt   # for plotting 
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


def fast_DFT(inSignal, s: int = -1, base = 32):
    """
    Fast implementation of the discrete Fourier transform.

    The complexity of this function is O(N log N)
    Where N is the length of the discrete input signal.

    This function uses the Cooley-Tukey algorithm.
    """
    y = np.zeros(inSignal.shape, dtype = complex)
    N = inSignal.shape[0]
    if N <= base:
        return naive_DFT(inSignal,s)
    else:
        yeven = fast_DFT(inSignal[0:N:2],s, base) #find FFT at even indices
        yodd = fast_DFT(inSignal[1:N:2],s, base) #find FFTat odd indices
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
    """
    Uses the naive_DFT function to compute the 2D DFT by 
    taking the DFT of each row and then the DFT of each column.
    """
    x = np.apply_along_axis(naive_DFT, 0, inSignal2D, s)
    y = np.apply_along_axis(naive_DFT, 1, x, s)
    return y


def naive_iDFT2D(inSignal2D: complex):
    """
    Uses the naive_DFT2D function to compute the inverse 2D DFT.
    """
    y = np.zeros(inSignal2D.shape, dtype = complex)
    N = inSignal2D.shape[0]
    inSignal2D = inSignal2D/(N**2)
    y = naive_DFT2D(inSignal2D,1)
    return y


def fast_DFT2D(inSignal2D, s: int = -1, base = 32):
    """
    Fast implementation of the 2D discrete Fourier transform.
    Uses the 1D FFT implemented with the Cooley-Tukey algorithm.
    """
    x = np.apply_along_axis(fast_DFT, 0, inSignal2D, s, base)
    y = np.apply_along_axis(fast_DFT, 1, x, s, base)
    return y


def fast_iDFT2D(inSignal2D: complex):
    """
    Fast implementation of the inverse 2D discrete Fourier transform.
    Uses the 1D FFT implemented with the Cooley-Tukey algorithm.
    """
    y = np.zeros(inSignal2D.shape, dtype = complex)
    N = inSignal2D.shape[0]
    inSignal2D = inSignal2D/(N**2)
    y = fast_DFT2D(inSignal2D,1)
    return y


def shift_freq_center(spectrum):
    """
    Helper function to shift the 0 frequency spike to the center.
    """
    N = spectrum.shape[0]
    return np.roll(spectrum, (N//2, N//2), axis=(1, 0))


def i_shift_freq_center(spectrum):
    """
    Helper function to inverse the shifting of the 0 frequency to the center.
    """
    N = spectrum.shape[0]
    return np.roll(spectrum, -(N//2), axis=(1, 0))


def filter_2D(spectrum, k=5, treshold_constant=4):
    """
    Filter the spectrum using band filtering method.
    """
    N = spectrum.shape[0]
    center_value = spectrum[N//2, N//2]

    slides = np.lib.stride_tricks.sliding_window_view(spectrum, (k, k))

    for i in range(len(slides)):
        for j in range(len(slides)):
            mean = np.mean(np.abs(slides[i, j]))
            std = np.std(np.abs(slides[i, j]))

            for m in range(len(slides[i, j])):
                for n in range(len(slides[i, j])):
                    if np.abs(slides[i, j, m, n]) > (mean + treshold_constant * std):
                        spectrum[i + m, j + n] *= mean / np.abs(spectrum[i + m, j + n])

    spectrum[N//2, N//2] = center_value
    return spectrum


def main():
    """
    Main method.
    """
    print("[Part 1.a]: Benchmarking FFT performance against a naive DFT for increasing N values with base case = 32, 2, and 128.")
    #benchmarking 1D-FFT's performance with base case = 32
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

        start_time = timeit.default_timer()
        y = fast_DFT(x)
        FFTtime = timeit.default_timer() - start_time ##time it takes for FFT for a given N
        FFTplot = np.append(FFTplot, FFTtime)

    plt.plot(Naxis,DFTplot)
    plt.plot(Naxis,FFTplot)
    plt.legend(['naive DFT','fast DFT'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT/FFT as a function of N with the base case = 32',fontsize = 10 )
    plt.show()
    
    #benchmarking 1D-FFT's performance with base case = 2
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
        # assert np.allclose(y, np.fft.fft(x)) # assert correctness of naive DFT

        start_time = timeit.default_timer()
        y = fast_DFT(x,base = 2)
        FFTtime = timeit.default_timer() - start_time ##time it takes for FFT for a given N
        FFTplot = np.append(FFTplot, FFTtime)
        # assert np.allclose(y, np.fft.fft(x)) # assert correctness of fast DFT

    plt.plot(Naxis,DFTplot)
    plt.plot(Naxis,FFTplot)
    plt.legend(['naive DFT','fast DFT'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT/FFT as a function of N with the base case = 2', fontsize = 10)
    plt.show()
    
    #benchmarking 1D-FFT's performance with base case = 128
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
        # assert np.allclose(y, np.fft.fft(x)) # assert correctness of naive DFT

        start_time = timeit.default_timer()
        y = fast_DFT(x,base = 128)
        FFTtime = timeit.default_timer() - start_time ##time it takes for FFT for a given N
        FFTplot = np.append(FFTplot, FFTtime)
        # assert np.allclose(y, np.fft.fft(x)) # assert correctness of fast DFT

    plt.plot(Naxis,DFTplot)
    plt.plot(Naxis,FFTplot)
    plt.legend(['naive DFT','fast DFT'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT/FFT as a function of N with the base case = 128', fontsize = 10)
    plt.show()
    
    
    print("[Part 1.b]: Benchmarking FFT-2D performance against a naive DFT-2D for increasing N values with the base case = 32, 2, and 128.")
    #benchmarking 2D-FFT's performance with base case = 32
    DFT2plot = np.array([])
    FFT2plot = np.array([])
    Naxis = np.array([])

    for m in range(11): 
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
    plt.title('Performance of DFT-2D/FFT-2D as a function of N with the base case = 32', fontsize = 10)
    plt.show()
    
    #benchmarking 2D-FFT's performance with base case = 2
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
        y = fast_DFT2D(z, base = 2)
        FFT2time = timeit.default_timer() - start_time ##time it takes for DFT for a given N
        FFT2plot = np.append(FFT2plot, FFT2time)

    plt.plot(Naxis, DFT2plot)
    plt.plot(Naxis,FFT2plot)
    plt.legend(['naive DFT-2D','fast DFT-2D'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT-2D/FFT-2D as a function of N with the base case = 2', fontsize = 10)
    plt.show()
    
    #benchmarking 2D-FFT's performance with base case = 128
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
        y = fast_DFT2D(z, base = 128)
        FFT2time = timeit.default_timer() - start_time ##time it takes for DFT for a given N
        FFT2plot = np.append(FFT2plot, FFT2time)

    plt.plot(Naxis, DFT2plot)
    plt.plot(Naxis,FFT2plot)
    plt.legend(['naive DFT-2D','fast DFT-2D'], title = "Legend")
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.title('Performance of DFT-2D/FFT-2D as a function of N with the base case = 128', fontsize = 10)
    plt.show()

    
    print("[Part 2]: Benchmarking 2D FFT performance against a naive 2D DFT in image application.")
    print("[Example 1]: Goose image.\nResolution: 512 x 512.")

    goose_clean = np.load('img/goose.npy')
    goose_noise = np.load("img/goose-noise.npy")

    fig, axs = plt.subplots(2,4)
    fig.suptitle("Goose image", fontsize = 16)

    ft_goose_clean = fast_DFT2D(goose_clean)
    ft_goose_clean = shift_freq_center(ft_goose_clean)

    ft_goose_noise = fast_DFT2D(goose_noise)
    ft_goose_noise = shift_freq_center(ft_goose_noise)

    ft_goose_filtered = fast_DFT2D(goose_noise)
    ft_goose_filtered = shift_freq_center(ft_goose_filtered)

    def blank(M, y, x):
        M[x:x+1, y:y+1] = 0

    # filter the spectrum statically
    blank(ft_goose_filtered, 240, 256)
    blank(ft_goose_filtered, 272, 256)
    blank(ft_goose_filtered, 251, 251)
    blank(ft_goose_filtered, 261, 261)

    goose_filtered = fast_iDFT2D(i_shift_freq_center(ft_goose_filtered)).real

    ft_goose_advance_filtered = fast_DFT2D(goose_noise)
    ft_goose_advance_filtered = shift_freq_center(ft_goose_advance_filtered)
    ft_goose_advance_filtered = filter_2D(ft_goose_advance_filtered, k=5)

    goose_advance_filtered = fast_iDFT2D(i_shift_freq_center(ft_goose_advance_filtered)).real

    goose_zoom = [231, 281]

    axs[0, 0].imshow(goose_clean, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 1].imshow(goose_noise, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 2].imshow(goose_filtered, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 3].imshow(goose_advance_filtered, plt.get_cmap('gray'), fontsize = 8)

    axs[0, 0].set_title("Clean goose image")
    axs[0, 1].set_title("Noisy goose image")
    axs[0, 2].set_title("Denoised goose | Static method")
    axs[0, 3].set_title("Denoised goose | Advance filter")

    axs[1, 0].imshow(np.log(abs(ft_goose_clean) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 1].imshow(np.log(abs(ft_goose_noise) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 2].imshow(np.log(abs(ft_goose_filtered) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 3].imshow(np.log(abs(ft_goose_advance_filtered) + .01), plt.get_cmap('gray'), fontsize = 8)

    axs[1, 0].set_xlim(goose_zoom)
    axs[1, 0].set_ylim(goose_zoom)

    axs[1, 1].set_xlim(goose_zoom)
    axs[1, 1].set_ylim(goose_zoom)

    axs[1, 2].set_xlim(goose_zoom)
    axs[1, 2].set_ylim(goose_zoom)

    axs[1, 3].set_xlim(goose_zoom)
    axs[1, 3].set_ylim(goose_zoom)

    axs[1, 0].set_title("Goose f spectrum")
    axs[1, 1].set_title("Noisy goose f spectrum")
    axs[1, 2].set_title("f spectrum | Static method")
    axs[1, 3].set_title("f spectrum | Advance filter")

    plt.show()

    print("[Example 2]: Lenna.\nResolution: 1024 x 1024.")

    lenna_clean = np.load('img/lenna-1024.npy')
    lenna_noise = np.load("img/lenna-1024-noise.npy")

    fig, axs = plt.subplots(2,4)

    fig.suptitle("Lenna image", fontsize = 16)

    ft_lenna_clean = fast_DFT2D(lenna_clean)
    ft_lenna_clean = shift_freq_center(ft_lenna_clean)

    ft_lenna_noise = fast_DFT2D(lenna_noise)
    ft_lenna_noise = shift_freq_center(ft_lenna_noise)

    ft_lenna_filtered = fast_DFT2D(lenna_noise)
    ft_lenna_filtered = shift_freq_center(ft_lenna_filtered)

    def blank(M, y, x, c=0):
        M[x-c:x+c+1, y-c:y+c+1] = 0
    
    # filter the spectrum statically
    blank(ft_lenna_filtered, 480, 512, 1)
    blank(ft_lenna_filtered, 544, 512, 1)

    blank(ft_lenna_filtered, 512, 496, 1)
    blank(ft_lenna_filtered, 512, 528, 1)

    lenna_filtered = fast_iDFT2D(i_shift_freq_center(ft_lenna_filtered)).real

    ft_lenna_advance_filtered = fast_DFT2D(lenna_noise)
    ft_lenna_advance_filtered = shift_freq_center(ft_lenna_advance_filtered)
    ft_lenna_advance_filtered = filter_2D(ft_lenna_advance_filtered, k=5)

    lenna_advance_filtered = fast_iDFT2D(i_shift_freq_center(ft_lenna_advance_filtered)).real

    lenna_zoom = [464, 560]

    axs[0, 0].imshow(lenna_clean, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 1].imshow(lenna_noise, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 2].imshow(lenna_filtered, plt.get_cmap('gray'), fontsize = 8)
    axs[0, 3].imshow(lenna_advance_filtered, plt.get_cmap('gray'), fontsize = 8)

    axs[0, 0].set_title("Lenna")
    axs[0, 1].set_title("Noisy Lenna")
    axs[0, 2].set_title("Denoised Lenna | Static method")
    axs[0, 3].set_title("Denoised Lenna | Advance filter")

    axs[1, 0].imshow(np.log(abs(ft_lenna_clean) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 1].imshow(np.log(abs(ft_lenna_noise) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 2].imshow(np.log(abs(ft_lenna_filtered) + .01), plt.get_cmap('gray'), fontsize = 8)
    axs[1, 3].imshow(np.log(abs(ft_lenna_advance_filtered) + .01), plt.get_cmap('gray'), fontsize = 8)

    axs[1, 0].set_title("Lenna f spectrum")
    axs[1, 1].set_title("Noisy Lenna f spectrum")
    axs[1, 2].set_title("f spectrum | Static method")
    axs[1, 3].set_title("f spectrum | Advance filter")

    axs[1, 0].set_xlim(lenna_zoom)
    axs[1, 0].set_ylim(lenna_zoom)

    axs[1, 1].set_xlim(lenna_zoom)
    axs[1, 1].set_ylim(lenna_zoom)

    axs[1, 2].set_xlim(lenna_zoom)
    axs[1, 2].set_ylim(lenna_zoom)

    axs[1, 3].set_xlim(lenna_zoom)
    axs[1, 3].set_ylim(lenna_zoom)

    plt.show()


if __name__ == "__main__":
    main()
