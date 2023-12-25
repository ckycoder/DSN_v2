import numpy as np
from scipy import fftpack

def gen_spectrogram_2(slc, win_size):
    hamming_win = np.hamming(win_size)#(32,)
    hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))#(32,32)

    spectrogram = np.zeros([win_size, win_size, 64-win_size, 64-win_size], dtype=complex)

    slc_fft = fftpack.fftshift(fftpack.fft2(slc))
    #print(slc_fft.shape)#(64,64)
    for i in range(64-win_size):
        for j in range(64-win_size):
            spectrogram[:,:,i,j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i+win_size, j:j+win_size],
                                                 shape=[win_size, win_size])
    return spectrogram


