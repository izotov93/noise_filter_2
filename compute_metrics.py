import numpy as np
from typing import Tuple
from pesq import pesq
import os
from scipy.io import wavfile
from sklearn.metrics import r2_score, mean_absolute_error
from pystoi import stoi


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) for two signals.

        :param original: The original signal.
        :param processed: The processed signal.

        :return: float - SNR in dB.
    """

    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')
    
    return 10 * np.log10(signal_power / noise_power)


def synchronize_signals(signal_one: np.ndarray, signal_two: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronizes two signals by trimming them to the same length, centered around their middle.

        :param signal_one: The first signal.
        :param signal_two: The second signal.

        :return: Tuple[np.ndarray, np.ndarray] - The trimmed signals aligned in length.
    """

    len_clean = len(signal_one)
    len_noisy = len(signal_two)

    #  Check which signal is longer and calculate the number of points to discard
    if len_clean > len_noisy:
        diff = len_clean - len_noisy
        trim = diff // 2

        signal_one_trimmed = signal_one[trim:len_clean - trim]
        return signal_one_trimmed, signal_two

    elif len_noisy > len_clean:
        diff = len_noisy - len_clean
        trim = diff // 2
        signal_two_trimmed = signal_two[trim:len_noisy - trim]
        return signal_one, signal_two_trimmed

    else:
        # Lengths are equal, return the original signals
        return signal_one, signal_two


def computing_metrics(mode: str = 'original', additional_LogNNet_param = None, name_filter = None):
    """
        Computes performance metrics for different noise conditions and signal processing modes.

        This function calculates various metrics (R2, MAE, PESQ, SNR, STOI) for audio signals
        across different noise levels and processing modes.

        :param mode: Processing mode for signal analysis.
                     Options include 'original', 'LogNNet_filtered_sounds', 'LogNNet_and_standard_filter'.
                     Defaults to 'original'.
        :param additional_LogNNet_param: Optional parameter for prediction-specific information.
        :param name_filter: Optional name of the filter used in signal processing.

        :return: Prints and stores performance metrics for each noise condition.
                 Metrics include mean values of:
                 - R2 (coefficient of determination)
                 - MAE (mean absolute error)
                 - PESQ (perceptual evaluation of speech quality)
                 - SNR (signal-to-noise ratio)
                 - STOI (short-time objective intelligibility)
        """
    # Configuration parameters - can be modified as needed
    dir_contained_original_sounds = 'NOIZEUS'
    snr_level = 5 # Can be changed to 10, 15, or 20
    dir_contained_filtered_LogNNet_sounds = 'LogNNet_filtered_sounds'
    dir_contained_filtered_sounds_after_LogNNet = 'LogNNet_and_standard_filters'

    noise_name = [d for d in os.listdir(dir_contained_original_sounds)
        if os.path.isdir(os.path.join(dir_contained_original_sounds, d)) and d != 'clean']

    clean_dir = os.path.join(dir_contained_original_sounds, 'clean')
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.startswith('sp') and f.endswith('.wav')])
    available_ids = sorted([f[2:4] for f in clean_files])

    addition_info = f'\nMode - {mode}'
    addition_info += f' - {additional_LogNNet_param}' if additional_LogNNet_param is not None else ''
    addition_info += f' {name_filter} filter' if name_filter is not None else ''

    print(addition_info)
    accumulated_results = {}
    print('noise\t"R2"\t"MAE"\t"PESQ"\t"SNR"\t"STOI"')
    for noise in noise_name:
        res_metric_sound = []
        for sound in available_ids:
            clean_filename = os.path.join(clean_dir, f'sp{sound}.wav')

            if mode == 'original':
                noisy_filename = os.path.join(dir_contained_original_sounds, noise,
                                              f'{snr_level}dB', f'sp{sound}_{noise}_sn{snr_level}.wav')

            elif mode == 'LogNNet_filtered_sounds':
                noisy_filename = os.path.join(dir_contained_filtered_LogNNet_sounds, noise,
                                              f"{snr_level}dB", additional_LogNNet_param,
                                              f'pred_sp{sound}_{noise}_sn{snr_level}_{additional_LogNNet_param}.wav')

            elif mode == 'LogNNet_and_standard_filter':
                noisy_filename = os.path.join(dir_contained_filtered_sounds_after_LogNNet,
                                              name_filter, f"{snr_level}dB", noise,
                                              f'LogNNet_and_standard_filters_sp{sound}_{noise}_sn{snr_level}.wav')
            else:
                print('Error. Wrong mode!')
                exit(0)

            fs, clean_signal = wavfile.read(clean_filename)
            fs_noisy, noisy_signal = wavfile.read(noisy_filename)
            clean_signal, noisy_signal = synchronize_signals(clean_signal, noisy_signal)

            pesq_score = pesq(fs, clean_signal, noisy_signal, 'wb' if fs != 8000 else 'nb')
            stoi_score = stoi(clean_signal, noisy_signal, fs)

            # Loop over k from 0.5 to 1.5 in 0.01 increments,
            # multiplying noisy_signal by k and calculating PESQ, SNR, R2 and MAE for each k.
            ks = np.arange(0.02, 1.5 + 0.001, 0.01)
            inter_res = []

            for k in ks:
                scaled_noisy = noisy_signal * k
                snr_val = compute_snr(clean_signal, scaled_noisy)
                r2_val = r2_score(clean_signal, scaled_noisy)
                mae_val = mean_absolute_error(clean_signal, scaled_noisy)

                inter_res.append([r2_val, mae_val, pesq_score, snr_val, stoi_score])

            # [max(R2), min(MAE), max(PESQ), max(SNR), max(STOI)]
            inter_res = np.array(inter_res)
            res_metric_sound.append([float(np.max(inter_res[:, 0])),
                                     float(np.min(inter_res[:, 1])),
                                     float(np.max(inter_res[:, 2])),
                                     float(np.max(inter_res[:, 3])),
                                     float(np.max(inter_res[:, 4]))])

        #  mean ["R2", "MAE", "PESQ", "SNR", "STOI"]
        accumulated_results[noise] = np.mean(np.array(res_metric_sound), axis=0)
        row = ' '.join(str(round(val, 6)) for val in accumulated_results[noise])
        print(f'{noise} {row}')


if __name__ == '__main__':
    # Options: mode - 'original', 'LogNNet_filtered_sounds', 'LogNNet_and_standard_filter'
    # Options: use_predict_info - 'HVG_HW_1', 'NO_HVG_HW_1', 'HVG_HW_3', 'NO_HVG_HW_3', 'HVG_HW_10', 'NO_HVG_HW_10'
    # Options: filter_name - 'Butter', 'Kalman', 'Kalman', 'Savgol', 'Wavelet', 'Wiener'

    # Calculation of metrics originals sounds
    computing_metrics(mode='original')

    # Calculation of metrics filtered LogNNet sounds
    for use_predict_info in ['HVG_HW_1', 'NO_HVG_HW_1', 'HVG_HW_3', 'NO_HVG_HW_3', 'HVG_HW_10', 'NO_HVG_HW_10']:
        computing_metrics(mode='LogNNet_filtered_sounds', additional_LogNNet_param=use_predict_info)

    # calculation of metrics after applying standard filters on LogNNet filtered sound
    for name_filter in ['Butter', 'Kalman', 'Savgol', 'Wavelet', 'Wiener']:
        for info in ['HVG_HW_1', 'NO_HVG_HW_1', 'HVG_HW_3', 'NO_HVG_HW_3', 'HVG_HW_10', 'NO_HVG_HW_10']:
            computing_metrics(mode='LogNNet_and_standard_filter', additional_LogNNet_param=info, name_filter=name_filter)
