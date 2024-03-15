import os
import time
import numpy as np
import pandas as pd
from scipy import signal


#读文本文件中的数据
def txt2array(txt_path):
    """
    :param txt_path:    specific path of a single txt file
    :return:            1-dimension preprocessed vector <class 'np.ndarray'>
                        of the input txt file
    """
    #使用pandas读文件数据
    table_file = pd.read_table(txt_path, header=None)
    #有的文件8个后面多一个tab或空格，pandas会多一个出来，下面语句删除多余的tab或空格
    table_file = table_file.dropna(axis=1)
    txt_file = table_file.iloc[:, :]
    #只取数据返回
    txt_array = txt_file.values

    return txt_array


def preprocessing(data):
    """
    :param data:    8*400 emg data <class 'np.ndarray'>    400*8
    :return:        data instance after rectifying and filter  8*400
    """
    # scalar
    #归一化处理
    data = 2 * (data + 128) / 256 - 1

    # rectify
    #整形，取绝对值
    data_processed = np.abs(data)

    # transpose (400, 8) -> (8, 400)
    #整形，矩阵转置
    data_processed = np.transpose(data_processed)

    # filter
    #四阶巴特沃斯低通滤波器
    wn = 0.05
    order = 4
    b, a = signal.butter(order, wn, btype='low')
    data_processed = signal.filtfilt(b, a, data_processed)      # data_processed <class 'np.ndarray': 8*400>

    return data_processed       # <class 'np.ndarray'> 4*800


def detect_muscle_activity(emg_data):
    """
    :param      emg_date: 8 channels of emg data -> 8*400
    :return:
                index_start: star index of muscle activation region
                index_end:   end index of muscle activation region
    """

    # plot emg_data
    # plt.plot(emg_data.transpose())
    # plt.show()

    #判断肌肉激活区域，参见相关论文，主要采用傅里叶变换后的能量谱来判断激活区域
    fs = 200        # sampling frequency
    min_activation_length = 50
    num_frequency_of_spec = 50
    hamming_window_length = 28 #28 25
    overlap_samples = 20        #20 10
    threshold_along_frequency = 18

    sumEMG = emg_data.sum(axis=0)   # sum 8 channel data into one vector
    # plt.plot(sumEMG)
    # plt.show()

    f, time, Sxx = signal.spectrogram(sumEMG, fs=fs,
                                   window='hamming',
                                   nperseg=hamming_window_length,
                                   noverlap=overlap_samples,
                                   nfft=num_frequency_of_spec,
                                   detrend=False,
                                   mode='complex')

    # 43.6893
    # test plot
    Sxx = Sxx * 43.6893

    # spec_values = abs(Sxx)
    # plt.pcolormesh(time, f, spec_values, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    #
    # plt.plot(sumEMG)
    # plt.show()

    spec_values = abs(Sxx)
    spec_vector = spec_values.sum(axis=0)

    # plt.plot(spec_vector)
    # plt.show()

    # 使用np.diff 求差分
    # indicated_vector 标记序列中哪些位置的强度值高于阈值
    indicated_vector = np.zeros(shape=(spec_vector.shape[0] + 2),)

    for index, element in enumerate(spec_vector):
        if element > threshold_along_frequency:
            indicated_vector[index+1] = 1

    # print('indicated_vector: %s' % str(indicated_vector))
    # print('indicated_vector.shape: %s' % str(indicated_vector.shape))

    index_greater_than_threshold = np.abs(np.diff(indicated_vector))

    if index_greater_than_threshold[-1] == 1:
        index_greater_than_threshold[-2] = 1

    # 删去最后一个元素
    index_greater_than_threshold = index_greater_than_threshold[:- 1]

    # 找出非零元素的序号
    index_non_zero = np.where(index_greater_than_threshold == 1)[0]

    index_of_samples = np.floor(fs * time - 1)
    num_of_index_non_zero = index_non_zero.shape[0]

    length_of_emg = sumEMG.shape[0]
    # print('length of emg : %f points' % length_of_emg)

    # find the start and end indexes
    if num_of_index_non_zero == 0:
        index_start = 1
        index_end = length_of_emg
    elif num_of_index_non_zero == 1:
        index_start = index_of_samples[index_non_zero]
        index_end = length_of_emg
    else:
        index_start = index_of_samples[index_non_zero[0]]
        index_end = index_of_samples[index_non_zero[-1] - 1]

    num_extra_samples = 25
    index_start = max(1, index_start - num_extra_samples)
    index_end = min(length_of_emg, index_end + num_extra_samples)

    if (index_end - index_start) < min_activation_length:
        index_start = 0
        index_end = length_of_emg - 1

    # print(index_start)
    # print(index_end)

    # return spec_vector, time, spec_values
    return index_start, index_end


def label_indicator(path):
    #根据手势文件名生成手势标签，要求手势文件名符合判断的标准
    label = None
    if 'relax_' in path:
        label = 0
    elif 'A01' in path:
        label = 1
    elif 'A02' in path:
        label = 2
    elif 'A03' in path:
        label = 3
    elif 'A04' in path:
        label = 4
    elif 'A05' in path:
        label = 5
    elif 'A06' in path:
        label = 6
    elif 'A07' in path:
        label = 7
    elif 'A08' in path:
        label = 8
    elif 'A09' in path:
        label = 9
    elif 'A10' in path:
        label = 10
    elif 'A11' in path:
        label = 11
    elif 'A12' in path:
        label = 12
    elif 'A13' in path:
        label = 13
    elif 'A14' in path:
        label = 14
    elif 'A15' in path:
        label = 15
    elif 'A16' in path:
        label = 16
    elif 'A17' in path:
        label = 17

    return label



def mav(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 400)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    #求一段肌电信号的mav值，详见公式
    mav_result = np.mean(abs(emg_data), axis=1)
    return mav_result

def ssc(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 400)
    :return:            Slope Sign Changes (SSC) of the input emg matrix     (8, )
    """   
    #求一段肌电信号的ssc值，详见公式
    ssc_result = np.count_nonzero(np.diff(np.sign(np.diff(emg_data))),axis=1)
    return ssc_result

def wl(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 400)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    #求一段肌电信号的wavelength值，详见公式
    wl_result = np.sum(abs(np.diff(emg_data)),axis=1)
    return wl_result

def hjorth(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 400)
    :return:            hjorth feature vector of the input emg matrix     (24, )
    """
    #求一段肌电信号的hj参数值，详见公式
    act = np.var(emg_data,axis=1)
    mob = np.sqrt(np.var(np.diff(emg_data,axis=1),axis=1)/np.var(emg_data,axis=1))
    com = np.sqrt(np.var(np.diff(np.diff(emg_data,axis=1),axis=1),axis=1)/np.var(np.diff(emg_data,axis=1),axis=1))/np.sqrt(np.var(np.diff(emg_data,axis=1),axis=1)/np.var(emg_data,axis=1))
    hjorth_result = np.concatenate((act,mob,com))
    return hjorth_result

def rms(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 400)
    :return:            rms feature vector of the input emg matrix     (8, )
    """
    #求一段肌电信号的rms均方根值，详见公式
    rms_result = np.sqrt((emg_data ** 2).mean(axis =1))
    return rms_result