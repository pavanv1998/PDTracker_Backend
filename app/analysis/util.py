import scipy.signal as signal


def filter_signal(raw_signal, fs=25, cut_off_frequency=5):
    b, a = signal.butter(2, cut_off_frequency, fs=fs, btype='low', analog=False)
    return signal.filtfilt(b, a, raw_signal)
