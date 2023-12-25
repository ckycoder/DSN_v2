import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pylab import subplots_adjust
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
centers = {'water_with_target': [22442, 9235],
           'water': [2860, 12576],
           'residential_1': [7117, 12773],
           'residential_2': [24764, 2464],
           'oil_tank_circle': [10181, 10694],
           'oil_tank_dot': [10185, 11065],
           'forest': [29398, 8752],
           'skyscraper': [24368, 2653],
           'water_forest': [9491,9274],
           'oil_tank_dot_2': [22508, 6051],
           'oil_tank_circle_2': [22131, 5271],
           'forest_1': [27725, 16821],
           'forest_2': [27679, 16612],
           'forest_3': [40733, 18514],
           'residential_3': [21855, 5723],
           'residential_4': [21907, 5900],
           'industrialbuilding_1': [24120, 3735],
           'industrialbuilding_2': [24090, 3863],
           'houston_1_HH': [24197, 2954]}





def gen_spectrogram_1(slc, win_size):
    hamming_win = np.hamming(win_size)
    hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
    spectrogram = np.zeros([win_size, win_size, win_size, win_size], dtype=complex)

    for i in range(win_size):
        for j in range(win_size):
            spectrogram[i, j, :, :] = fftpack.fft2(fftpack.ifftshift(fftpack.ifftn(hamming_win_2d)) * slc[i:i+win_size, j:j+win_size],
                                                   shape=(win_size, win_size))
    return spectrogram

def gen_spectrogram_2(slc, win_size):
    hamming_win = np.hamming(win_size)
    hamming_win_2d = np.sqrt(np.outer(hamming_win, hamming_win))
    spectrogram = np.zeros([win_size, win_size, 64-win_size, 64-win_size], dtype=complex)

    slc_fft = fftpack.fftshift(fftpack.fft2(slc))
    for i in range(64-win_size):
        for j in range(64-win_size):
            spectrogram[:,:,i,j] = fftpack.ifftn(hamming_win_2d * slc_fft[i:i+win_size, j:j+win_size],
                                                 shape=[win_size, win_size])

    return spectrogram

def show_spectrogram(name, slc, spectrogram):
    s0 = np.abs(spectrogram[:, :, 16, 16])
    s1 = np.abs(spectrogram[16, 16, :, :])
    s2 = np.abs(spectrogram[16, :, 16, :])
    s3 = np.abs(spectrogram[:, 16, :, 16])
    s4 = np.abs(spectrogram[:, 16, 16, :])
    s5 = np.abs(spectrogram[16, :, :, 16])


    plt.subplots(ncols=7, nrows=1, figsize=(15,4))
    plt.title(name)
    plt.subplot(1, 7, 1)
    plt.imshow(np.log(1 + np.abs(slc)), cmap=plt.cm.jet)
    plt.title('original slc')
    plt.subplot(1, 7, 2)
    plt.imshow(np.log(1 + s0), cmap=plt.cm.jet)
    plt.title('fr=BWr/2,fa=BWa/2')
    plt.subplot(1, 7, 3)
    plt.imshow(np.log(1 + s1), cmap=plt.cm.jet)
    plt.title('x=N/2,y=N/2')
    plt.subplot(1, 7, 4)
    plt.imshow(np.log(1 + s2), cmap=plt.cm.jet)
    plt.title('x=N/2,fr=BWr/2')
    plt.subplot(1, 7, 5)
    plt.imshow(np.log(1 + s3), cmap=plt.cm.jet)
    plt.title('y=N/2,fa=BWa/2')
    plt.subplot(1, 7, 6)
    plt.imshow(np.log(1 + s4), cmap=plt.cm.jet)
    plt.title('y=N/2,fr=BWr/2')
    plt.subplot(1, 7, 7)
    plt.imshow(np.log(1 + s5), cmap=plt.cm.jet)
    plt.title('x=N/2,fa=BWa/2')

    # plt.savefig(name + '.png')


def show_spectrogram_3D(spectrogram):
    s0 = np.abs(spectrogram[:, :, 16, 16])
    s1 = np.abs(spectrogram[16, 16, :, :])
    s2 = np.abs(spectrogram[16, :, 16, :])
    s3 = np.abs(spectrogram[:, 16, :, 16])
    s4 = np.abs(spectrogram[:, 16, 16, :])
    s5 = np.abs(spectrogram[16, :, :, 16])

    # plt.show()


def show_spectrogram_frfa(name, spectrogram):
    # total = 16
    plt.subplots(ncols=4, nrows=4, figsize=(15,15))
    for count, i in enumerate(range(0,32,2)):
        s = np.abs(spectrogram[:,:,i,i])
        plt.subplot(4,4,count+1)
        plt.imshow(np.log(1+s), cmap=plt.cm.jet)
        plt.title('fr,fa='+str(i))
    plt.savefig(name + '_frfa.png')

def show_spectrogram_xy(name, spectrogram):
    # total = 16
    plt.subplots(ncols=4, nrows=4, figsize=(15,15))
    for count, i in enumerate(range(0,32,2)):
        s = np.abs(spectrogram[i,i,:,:])
        plt.subplot(4,4,count+1)
        plt.imshow(np.log(1+s), cmap=plt.cm.jet)
        plt.title('x,y='+str(i))
    plt.savefig(name + '_xy.png')

def show_spectrogram_xfr(name, spectrogram):
    # total = 16
    plt.subplots(ncols=4, nrows=4, figsize=(15,15))
    for count, i in enumerate(range(0,32,2)):
        s = np.abs(spectrogram[i,:,16,:])
        plt.subplot(4,4,count+1)
        plt.imshow(np.log(1+s), cmap=plt.cm.jet)
        plt.title('x='+str(i))
    plt.savefig(name + '_fr=16_x.png')

def show_spectrogram_yfr(name, spectrogram):
    # total = 16
    plt.subplots(ncols=4, nrows=4, figsize=(15,15))
    for count, i in enumerate(range(0,32,2)):
        s = np.abs(spectrogram[:,i,16,:])
        plt.subplot(4,4,count+1)
        plt.imshow(np.log(1+s), cmap=plt.cm.jet)
        plt.title('y='+str(i))
    plt.savefig(name + '_fr=16_y.png')


# for _, name in enumerate(centers):
#     slc = np.load(name + '.npy')
#     spectrogram = gen_spectrogram_2(slc, win_size)
    # show_spectrogram(name, slc, spectrogram2)
    # show_spectrogram_frfa(name, spectrogram)
    # show_spectrogram_xy(name, spectrogram)
    # show_spectrogram_xfr(name, spectrogram)
    # show_spectrogram_yfr(name, spectrogram)

data_folder = '../data/slc_data/'
name = 'agriculture/agriculture_4312_HH_11703_2055'
# name = 'forest/forest_BBE7_HH_33422_15718'
#name = 'data/slc_data/skyscraper/skyscraper_BBE7_HH_22966_1581'
# name = '../data/slc_data/industrialbuilding/industrialbuilding_BBE7_HH_25053_4587'
# name = '../data/slc_data/industrialbuilding/industrialbuilding_BBE7_HH_25053_4687'
# name = '../data/slc_data/container/container_EBAD_HH_12531_7157'
# name = '../data/slc_data/container/container_EBAD_HH_12863_5707'
# name = '../data/slc_data/skyscraper/skyscraper_BBE7_HH_23016_1631'
# name = '../data/slc_data/storagetank/storagetank_BBE7_HH_10431_9993'
#name = 'container_4312_HH_5575_17797'

for win_size in [16,32,48]:
    spectrogram = gen_spectrogram_2(np.load(data_folder + name + '.npy'), win_size)
    #show_spectrogram(name, np.load('../' + name + '.npy'), spectrogram)
    #plt.imshow(np.log(1 + np.abs(np.load('../' + name + '.npy'))), cmap=plt.cm.gray)

    print(spectrogram.shape)
    fig = plt.figure(figsize=(10,10))
    s1 = np.abs(spectrogram[int(win_size/2),int(win_size/2) , :, :]).reshape(64-win_size,64-win_size)
    x = np.outer(np.linspace(0, 1, 64-win_size), np.ones(64-win_size))
    y = x.copy().T
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, np.log(1+s1), cmap=plt.cm.jet)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tick_params(axis='z',labelsize=15)

    plt.xlabel('range',fontsize=35,labelpad=8)
    plt.ylabel('azimuth',fontsize=35,labelpad=8)
    ax.set_zlabel('amplitude', fontsize=35,labelpad=8)
    zmajorFormatter = FormatStrFormatter('%1.1f')
    ax.zaxis.set_major_formatter(zmajorFormatter)
    plt.savefig('./'+name + '_' + str(win_size)+'_3D.png')
'''
# fig = plt.figure(figsize=(20,4))
plt.subplots(ncols=4, nrows=1, figsize=(12,3))
subplots_adjust(wspace=0.5)
# plt.subplot(1, 5, 1)
# plt.imshow(np.log(1 + np.abs(np.load('../' + name + '.npy'))), cmap=plt.cm.jet)
# plt.xlabel('range (x)')
# plt.ylabel('azimuth (y)')
for i in range(4):
    s = np.abs(spectrogram[16+i*2,16,:,:])
    plt.subplot(1, 4, i+1)
    plt.imshow(np.log(1 + s), cmap=plt.cm.jet)
    plt.xlabel('range (fr)')
    plt.ylabel('azimuth (fa)')
    # plt.title('original slc')
'''
# fig = plt.figure(figsize=(15,12))
# count = 0
# for i in range(3):
#     for j in range(3):
#         count+=1
#         s1 = np.abs(spectrogram[15*i, 15*j, :, :]).reshape(32,32)
#
#         x = np.outer(np.linspace(0, 1, win_size), np.ones(win_size))
#         y = x.copy().T
#         ax = fig.add_subplot(3,3,count, projection='3d')
#         # plt.axes(projection='3d')
#         ax.plot_surface(x, y, np.log(1+s1), cmap=plt.cm.jet)
#         plt.xlabel('range')
#         plt.ylabel('azimuth')
# plt.title(name)
# plt.savefig(name + '_3D.png')
#plt.show()
#
# slc_fft = fftpack.fft2(slc)
#
# print(2)
