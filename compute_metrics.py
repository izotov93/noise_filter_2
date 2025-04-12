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
        # Lengths are equal, return the originals signals
        return signal_one, signal_two


def main():
    dir_contained_original_sounds = 'NOIZEUS'
    snr_level = 5
    dir_contained_predict_sounds = 'LogNNet_filtered_sounds'
    use_predict_info = 'NO_HVG_NW_1'

    # Operation mode flag
    # Options: 'original', 'predict'
    mode = 'predict'

    clean_dir = os.path.join(dir_contained_original_sounds, 'clean')
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.startswith('sp') and f.endswith('.wav')])
    available_ids = sorted([f[2:4] for f in clean_files])

    # Set the range for the training set (e.g., from 01 to 04)
    training_start = 1
    training_end = 4

    # Form lists of IDs for the training and test sets
    training_ids = [f'{i:02d}' for i in range(training_start, training_end + 1)]
    available_ids = [id_ for id_ in available_ids if id_ not in training_ids]

    res_metric_noise = {}

    noise_name = ['white_noise_0.01', 'white_noise_0.05', 'white_noise_0.1',
                  'white_noise_0.2', 'white_noise_0.3', 'white_noise_0.4',
                  'airport', 'babble', 'car', 'exhibition', 'restaurant',
                  'station', 'street', 'train']

    print(f'Computing metrics from {mode} sounds\n')
    print('noise\t"R2"\t"MAE"\t"PESQ"\t"SNR"\t"STOI"')
    for noise in noise_name:
        res_metric_sound = []
        for sound in available_ids:
            clean_filename = os.path.join(dir_contained_original_sounds, 'clean', f'sp{sound}.wav')

            if mode == 'original':
                noisy_filename = os.path.join(dir_contained_original_sounds, noise,
                                              f'{snr_level}dB', f'sp{sound}_{noise}_sn{snr_level}.wav')
            elif mode == 'predict':
                noisy_filename = os.path.join(dir_contained_predict_sounds, noise, f"{snr_level}dB", use_predict_info,
                                              f'pred_sp{sound}_{noise}_sn{snr_level}_{use_predict_info}.wav')
            else:
                print('Error. Wrong mode!')
                exit(0)

            fs, clean_signal = wavfile.read(clean_filename)
            fs_noisy, noisy_signal = wavfile.read(noisy_filename)
            clean_signal, noisy_signal = synchronize_signals(clean_signal, noisy_signal)

            # compute PESQ
            pesq_score = pesq(fs, clean_signal, noisy_signal, 'wb' if fs != 8000 else 'nb')
            # compute STOI
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
        res_metric_noise[noise] = np.mean(np.array(res_metric_sound), axis=0)
        row = '\t'.join(str(round(val, 6)) for val in res_metric_noise[noise])
        print(f'{noise} {row}')


if __name__ == '__main__':
    main()