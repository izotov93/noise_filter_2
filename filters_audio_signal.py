import os
import numpy as np
import scipy.signal as signal
import pywt
from scipy.optimize import differential_evolution
from pesq import pesq
from scipy.io import wavfile
import json
from compute_metrics import synchronize_signals


def compute_pesq(clean_signal: np.ndarray, processed_signal: np.ndarray, fs: int) -> float:
    """
    Computes the PESQ (Perceptual Evaluation of Speech Quality) score between a clean signal and a processed signal.

        :param clean_signal: The original clean audio signal.
        :param processed_signal: The processed audio signal to compare against the clean signal.
        :param fs: The sampling frequency of the audio signals.

        :return: The PESQ score, where a higher score indicates better audio quality.
    """

    if np.isnan(processed_signal).any():
        print("Error: NaN found in the processed signal")
        return 0

    try:
        # Determine mode based on sampling frequency
        mode = 'nb' if fs == 8000 else 'wb'
        score = pesq(fs, clean_signal, processed_signal, mode)

    except Exception as e:
        score = 0
        print("Error while computing PESQ:", e)

    return score


def apply_wiener(noisy_signal: np.ndarray, mysize: int = 5) -> np.ndarray:
    """
    Applies Wiener filter to a noisy signal to suppress noise.

        :param noisy_signal: The input noisy audio signal that needs denoising.
        :param mysize: The size of the filter. Higher values may lead to better
            denoising but may also cause loss of detail.

        :return: The denoised audio signal after applying the Wiener filter.
    """

    mysize = int(round(mysize))  # Ensure `mysize` is an integer

    return signal.wiener(noisy_signal, mysize=mysize)


def apply_kalman(noisy_signal: np.ndarray, Q: float = 1e-7, R: float = 0.1) -> np.ndarray:
    """
    Applies a Kalman filter to a noisy signal for noise reduction.

        :param noisy_signal: The input noisy audio signal that needs denoising.
        :param Q: The process noise covariance. Smaller values result in less noise suppression.
        :param R: The measurement noise covariance. Smaller values give more weight to the measurements.

        :return: The filtered signal after applying the Kalman filter.
    """

    n = len(noisy_signal)
    # Estimated signal and estimated error covariance
    xhat = np.zeros(n)
    P = np.zeros(n)

    # Initialize the first estimated value
    xhat[0] = noisy_signal[0]
    P[0] = 1.0

    for k in range(1, n):
        xhat_minus = xhat[k - 1] # Predicted state
        P_minus = P[k - 1] + Q # Predicted error covariance
        K = 0 if (P_minus + R) == 0 else P_minus / (P_minus + R) # Kalman gain
        xhat[k] = xhat_minus + K * (noisy_signal[k] - xhat_minus) # Updated state
        P[k] = (1 - K) * P_minus # Updated error covariance
        
    return xhat


def apply_wavelet_denoising(noisy_signal: np.ndarray, wavelet: str = 'db8', level: int = 1,
                            thresh_coeff: float = 1.0, thresh_mode: int = 0) -> np.ndarray:
    """
    Applies wavelet denoising to a noisy signal using discrete wavelet transforms.

        :param noisy_signal: The input noisy audio signal that needs denoising.
        :param wavelet: The wavelet type to be used for the decomposition.
                        Default is 'db8'. Possible values include 'db1', 'db2', 'db4', 'db8', 'sym2', 'sym4'.
        :param level: The level of wavelet decomposition. The higher the level, the more aggressive the denoising.
        :param thresh_coeff: A coefficient to scale the threshold for denoising.
        :param thresh_mode: The mode of thresholding; 0 for 'soft' and 1 for 'hard'.

        :return: The denoised signal obtained after applying wavelet thresholding.
    """

    wavelet_options = ['db1', 'db2', 'db4', 'db8', 'sym2', 'sym4']
    if isinstance(wavelet, int):
        wavelet = wavelet_options[min(wavelet, len(wavelet_options)-1)]

    level = int(round(level))
    thresh_coeff = float(thresh_coeff)
    thresh_mode = int(round(thresh_mode))
    mode_options = ['soft', 'hard']

    # Perform wavelet decomposition
    thresh_mode_str = mode_options[min(thresh_mode, len(mode_options)-1)]
    coeff = pywt.wavedec(noisy_signal, wavelet, mode="per")

    #  Estimate the noise standard deviation
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level]))
    uthresh = thresh_coeff * sigma * np.sqrt(2 * np.log(len(noisy_signal)))

    # Apply thresholding to the wavelet coefficients
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode=thresh_mode_str) for c in coeff[1:]]

    # Reconstruct the denoised signal
    return pywt.waverec(coeff, wavelet, mode="per")


def apply_savgol(noisy_signal: np.ndarray, window_length: int = 15, polyorder: int = 2) -> np.ndarray:
    """
    Applies a Savitzky-Golay filter to a noisy signal for noise reduction.

        :param noisy_signal: The input noisy audio signal that needs smoothing.
        :param window_length: The length of the filter window (must be a positive odd integer).
        :param polyorder: The order of the polynomial used to fit the samples. Must be less than `window_length`.

        :return: The smoothed signal after applying the Savitzky-Golay filter.
    """

    window_length = int(round(window_length))
    polyorder = int(round(polyorder))
    return signal.savgol_filter(noisy_signal, window_length=window_length, polyorder=polyorder)


def apply_butter_lowpass(noisy_signal: np.ndarray, fs: float, cutoff: int = 1282, order: int = 1) -> np.ndarray:
    """
    Applies a Butterworth low-pass filter to a noisy signal.

        :param noisy_signal: The input noisy audio signal that needs filtering.
        :param fs: The sampling frequency of the signal.
        :param cutoff: The cutoff frequency of the low-pass filter. It should be less than `fs / 2`.
        :param order: The order of the Butterworth filter. Higher values result in steeper roll-off.

        :return: The filtered signal after applying the Butterworth low-pass filter.
    """

    cutoff = int(round(cutoff))
    order = int(round(order))

    # Calculate the Nyquist frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter using zero-phase filtering
    return signal.filtfilt(b, a, noisy_signal)


def filter_and_check(filter_func, noisy_signal: np.ndarray, params: list) -> np.ndarray:
    """
    Applies a given filter function to a noisy signal and checks for NaN values in the output.

        :param filter_func: A function that takes a noisy signal and additional parameters to apply a filter.
        :param noisy_signal: The input noisy audio signal that needs filtering.
        :param params: A list of parameters to be passed to the filter function.

        :return: The filtered signal, with NaN values replaced by zeros if applicable.
    """

    filtered_signal = filter_func(noisy_signal, *params)
    if np.isnan(filtered_signal).any():
        print(f"Filter  {filter_func.__name__} returned NaN; replacing with zeros.")
        return np.nan_to_num(filtered_signal)
    return filtered_signal


def optimize_filter(filter_func, param_bounds: list, clean_signal: np.ndarray,
                    noisy_signal: np.ndarray, fs: float, int_indices=None) -> list:
    """
    Optimizes filter parameters to maximize the PESQ score between the clean and noisy signals.

        :param filter_func: The filtering function to be optimized.
        :param param_bounds: A list of tuples specifying the bounds for each parameter to be optimized.
        :param clean_signal: The original clean audio signal for comparison.
        :param noisy_signal: The input noisy audio signal that needs filtering.
        :param fs: The sampling frequency of the signals.
        :param int_indices: A list of indices of parameters that should be converted to integers.

        :return: A list of optimized parameters for the given filter function.
    """

    if int_indices is None:
        int_indices = []

    def objective(params):
        # Convert parameters according to int_indices for PESQ computation
        new_params = [int(round(p)) if i in int_indices else p for i, p in enumerate(params)]

        # Additional check for Savitzky–Golay filter
        if filter_func.__name__ == 'apply_savgol':
            window_length, polyorder = new_params
            if polyorder >= window_length:
                # If condition is violated, apply a penalty
                return 1e6

        filtered_signal = filter_func(noisy_signal, *new_params)
        # Maximize PESQ
        return -compute_pesq(clean_signal, filtered_signal, fs)

    result = differential_evolution(objective, param_bounds, strategy='best1bin', tol=1e-2,  maxiter=10)
    # Convert final parameters according to int_indices
    converted_result = [int(round(p)) if i in int_indices else p for i, p in enumerate(result.x)]
    return converted_result


def train_optimal_params_filters(clean_dir: str, noisy_dir: str, training_ids: list,
                                  noise: str, snr_level: int, pred_prefix: str = '',
                                  pred_suffix: str = '') -> dict:
    """
    Trains optimal filter parameters by combining clean and noisy signals.

        :param clean_dir: Directory containing clean audio files.
        :param noisy_dir: Directory containing noisy audio files.
        :param training_ids: List of identifiers for the training audio files.
        :param noise: Type of noise applied to the signals.
        :param snr_level: Signal-to-noise ratio level used in the noisy signals.
        :param pred_prefix: Prefix for the noisy file names (optional).
        :param pred_suffix: Suffix for the noisy file names (optional).

        :return: A dictionary with optimal parameters for each filter type.
    """

    combined_clean_train = np.array([], dtype=np.int16)
    combined_noisy_train = np.array([], dtype=np.int16)

    for id_ in training_ids:
        clean_path = os.path.join(clean_dir, f"sp{id_}.wav")
        noisy_path = os.path.join(noisy_dir, f"{pred_prefix}sp{id_}_{noise}_sn{snr_level}{pred_suffix}.wav")

        if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
            print(f"File missing for ID {id_}")
            continue

        fs, clean_sig = wavfile.read(clean_path)
        fs_noisy, noisy_sig = wavfile.read(noisy_path)
        clean_sig, noisy_sig = synchronize_signals(clean_sig, noisy_sig)
        
        if fs != fs_noisy:
            raise ValueError(f"Sampling frequencies do not match for ID {id_}!")

        combined_clean_train = np.concatenate((combined_clean_train, clean_sig))
        combined_noisy_train = np.concatenate((combined_noisy_train, noisy_sig))

    # Output information about the combined training signals
    print(f"Combined training signal: fs = {fs}, total length = {len(combined_clean_train)}")
    print(f"Combined training signal noisy: fs = {fs}, total length = {len(combined_noisy_train)}")
    print(f"Filter - {noise}")

    # Optimize filter parameters on the combined training signal
    best_wiener_params = optimize_filter(apply_wiener,
                                         [(3, 15)],
                                         combined_clean_train,
                                         combined_noisy_train,
                                         fs,
                                         int_indices=[0])
    print("Optimal Wiener params:", best_wiener_params)

    best_kalman_params = optimize_filter(apply_kalman,
                                         [(1e-9, 1e-6), (0.05, 0.5)],
                                         combined_clean_train,
                                         combined_noisy_train,
                                         fs)
    best_kalman_params = [float(item) for item in best_kalman_params]
    print("Optimal Kalman params:", best_kalman_params)

    best_wavelet_params = optimize_filter(apply_wavelet_denoising,
                                          [(0, 5), (1, 8), (0.5, 2.0), (0, 1)],
                                          combined_clean_train,
                                          combined_noisy_train,
                                          fs,
                                          int_indices=[0, 1, 3])
    best_wavelet_params[2] = float(best_wavelet_params[2])
    print("Optimal Wavelet params:", best_wavelet_params)

    best_savgol_params = optimize_filter(apply_savgol,
                                         [(5, 21), (1, 5)],
                                         combined_clean_train,
                                         combined_noisy_train,
                                         fs,
                                         int_indices=[0, 1])
    print("Optimal Savgol params:", best_savgol_params)

    best_butter_params = optimize_filter(lambda sig, cutoff, order:
                                         apply_butter_lowpass(sig, fs, cutoff, order),
                                         [(300, 3000), (1, 5)],
                                         combined_clean_train,
                                         combined_noisy_train,
                                         fs,
                                         int_indices=[0, 1])
    print("Optimal Butter Lowpass params:", best_butter_params)

    return {
        'Wiener': best_wiener_params,
        'Kalman': best_kalman_params,
        'Wavelet': best_wavelet_params,
        'Savgol': best_savgol_params,
        'Butter': best_butter_params
    }


def apply_filter_by_optimal_params(dir_contained_sounds: str, dir_pred_sound: str,
                                   snr_level: int, test_ids: list,
                                   optimal_params: dict, prefix_out_name: str = None,
                                   use_predict_info: str = None) -> dict:
    """
        Applies optimized filters to test audio signals and computes PESQ scores.

        :param dir_pred_sound: Directory containing prediction audio files.
        :param dir_contained_sounds: Directory containing clean and noisy audio files.
        :param snr_level: Signal-to-noise ratio level for the noisy signals.
        :param test_ids: List of identifiers for the test audio files.
        :param optimal_params: Dictionary containing optimal parameters for each noise type.
        :param prefix_out_name: Prefix for output filenames (optional).
        :param use_predict_info: Additional information for prediction directory structure (optional).

        :return: A dictionary with average PESQ results for each noise type.
    """

    prefix_out_name = 'filter_before' if prefix_out_name is None else prefix_out_name
    result_test_all = {}
    clean_dir = os.path.join(dir_contained_sounds, "clean")

    for noise in list(optimal_params.keys()):
        result_testing = {'Signal/noise': []}
        for sub_key in optimal_params[noise].keys():
            result_testing[sub_key] = []

        if use_predict_info is None:
            noisy_dir = os.path.join(dir_contained_sounds, noise, f"{snr_level}dB")
        else:
            noisy_dir = os.path.join(dir_pred_sound, noise, f"{snr_level}dB", use_predict_info)

        for id_ in test_ids:
            clean_path = os.path.join(clean_dir, f"sp{id_}.wav")
            if use_predict_info is None:
                noisy_path = os.path.join(noisy_dir, f"sp{id_}_{noise}_sn{snr_level}.wav")
            else:
                noisy_path = os.path.join(noisy_dir, f"pred_sp{id_}_{noise}_sn{snr_level}_{use_predict_info}.wav")

            if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
                print(f"Missing test file for ID {id_}")
                continue
            fs_test, clean_sig = wavfile.read(clean_path)
            fs_test2, noisy_sig = wavfile.read(noisy_path)

            clean_sig, noisy_sig = synchronize_signals(clean_sig, noisy_sig)

            if fs_test != fs_test2:
                raise ValueError(f"Sampling frequencies do not match for test file {id_}!")

            # Compute PESQ for the original/noisy) signals
            result_testing['Signal/noise'].append(compute_pesq(clean_sig, noisy_sig, fs_test))

            # Apply optimized filters to the test signal
            filtered_signal = {'Wiener': filter_and_check(apply_wiener, noisy_sig,
                                                          optimal_params[noise]['Wiener']),
                               'Kalman': filter_and_check(apply_kalman, noisy_sig,
                                                          optimal_params[noise]['Kalman']),
                               'Wavelet': filter_and_check(apply_wavelet_denoising, noisy_sig,
                                                           optimal_params[noise]['Wavelet']),
                               'Savgol': filter_and_check(apply_savgol, noisy_sig,
                                                          optimal_params[noise]['Savgol']),
                               'Butter': filter_and_check(lambda sig, cutoff, order:
                                                          apply_butter_lowpass(sig, fs_test, cutoff, order),
                                                          noisy_sig, optimal_params[noise]['Butter'])}

            for filter_name in filtered_signal.keys():
                result_testing[filter_name].append(compute_pesq(clean_sig,
                                                                filtered_signal[filter_name],
                                                                fs_test))

                filter_path = os.path.join(dir_contained_sounds, prefix_out_name, filter_name, f'{snr_level}dB', noise)
                os.makedirs(filter_path, exist_ok=True)
                wav_file = os.path.join(filter_path, f'{prefix_out_name}_sp{id_}_{noise}_sn{snr_level}.wav')
                wavfile.write(wav_file, fs_test, filtered_signal[filter_name])

        # Calculate average PESQ values for the test set
        print(f"\nRESULT - {noise}")
        for key in result_testing.keys():
            result_testing[key] = round(np.mean(result_testing[key]), 6) if result_testing[key] else 0
            print(f"{key} {result_testing[key]}")
        result_test_all[noise] = result_testing

    return result_test_all


def training_optimal_params_by_original_sounds(dir_contained_sounds: str, training_ids: list,
                                               snr_level: int, out_file_name: str,
                                               noise_name: list = None, ) -> str:
    """
    Trains to obtain optimal filter parameters for original sounds across different types of noise.


        :param dir_contained_sounds: Directory containing clean and noisy audio files.
        :param training_ids: List of identifiers for the training audio files.
        :param snr_level: Signal-to-noise ratio level for the noisy signals.
        :param noise_name: List of noise types to train the filters on (optional).
                           If None, defaults to a predefined list of noise types.
        :param out_file_name: The name of the output JSON file containing the optimal parameters.

        :return: Dict containing the optimal parameters.
    """

    if noise_name is None:
        noise_name = ['airport', 'babble', 'car', 'exhibition', 'restaurant', 'station', 'street', 'train']

    clean_dir = os.path.join(dir_contained_sounds, "clean")
    for noise in noise_name:
        noisy_dir = os.path.join(dir_contained_sounds, noise, f"{snr_level}dB")

        optimal_params[noise] = train_optimal_params_filters(clean_dir=clean_dir,
                                                             noisy_dir=noisy_dir,
                                                             training_ids=training_ids,
                                                             noise=noise,
                                                             snr_level=snr_level)

    with open(out_file_name, 'w') as fp:
        json.dump(optimal_params, fp, indent=4)

    return optimal_params


def training_optimal_params_by_predict_sound(dir_clear_sound: str, dir_pred_sound: str,
                                             training_ids: list, snr_level: int, info: str,
                                             pred_suffix: str, out_file_name: str,
                                             noise_name: list = None, pred_prefix: str = 'pred_') -> str:
    """
    Trains to obtain optimal filter parameters from predicted sound data across different types of noise.

        :param dir_clear_sound: Directory containing clean audio files
        :param dir_pred_sound: Directory containing predictions audio files.
        :param training_ids: List of identifiers for the training audio files.
        :param snr_level: Signal-to-noise ratio level for the noisy signals.
        :param info: Additional information that may be used in file naming or structure.
        :param pred_suffix: Suffix for the predicted audio file names.
        :param out_file_name: The name of the output JSON file containing the optimal parameters.
        :param noise_name: List of noise types to train the filters on (optional).
                           If None, defaults to a predefined list of noise types.
        :param pred_prefix: Prefix for the predicted audio file names (optional).

        :return: Dict containing the optimal parameters.
    """

    if noise_name is None:
        noise_name = ['airport', 'babble', 'car', 'exhibition', 'restaurant', 'station', 'street', 'train']
    clean_dir = os.path.join(dir_clear_sound, "clean")

    # Initialize the dictionary to hold optimal parameters
    optimal_params = {}
    for noise in noise_name:
        # Directory containing predicted noisy signals
        noisy_dir = os.path.join(dir_pred_sound, noise, f"{snr_level}dB", info)

        # Train optimal parameters for the current type of noise using predicted signals
        optimal_params[noise] = train_optimal_params_filters(clean_dir=clean_dir,
                                                             noisy_dir=noisy_dir,
                                                             training_ids=training_ids,
                                                             noise=noise,
                                                             snr_level=snr_level,
                                                             pred_prefix=pred_prefix,
                                                             pred_suffix=pred_suffix)

    with open(out_file_name, 'w') as fp:
        json.dump(optimal_params, fp, indent=4)

    return optimal_params



if __name__ == "__main__":

    # Operation mode flag
    # Options: 'train_original', 'train_predicted', 'test'
    mode = 'train_predicted'

    # Define the sounds directory and signal-to-noise ratio level
    dir_contained_original_sounds = 'NOIZEUS'
    snr_level = 5

    dir_contained_predict_sounds = 'LogNNet_filtered_sounds'
    dir_saving_optimal_params = 'optimal_params'
    os.makedirs(dir_saving_optimal_params, exist_ok=True)

    # Set the range for the training set (e.g., from 01 to 04)
    training_start = 1
    training_end = 4

    # Determine directories for clean and noisy signals
    clean_dir = os.path.join(dir_contained_original_sounds, "clean")
    # Get the list of files from the folder with clean signals (files like 'spXX.wav')
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.startswith('sp') and f.endswith('.wav')])
    # Extract file numbers (assuming the number is always two digits)
    available_ids = sorted([f[2:4] for f in clean_files])
    # Form lists of IDs for the training and test sets
    training_ids = [f'{i:02d}' for i in range(training_start, training_end + 1)]
    test_ids = [id_ for id_ in available_ids if id_ not in training_ids]

    print('Training IDs:', training_ids)
    print('Test IDs:', test_ids)
    print(f'\nSEARCH OPTIMAL PARAMETERS - mode - {mode}\n')

    optimal_params = {}
    LogNNet_info_params = ['HVG_NW_1', 'NO_HVG_NW_1', 'HVG_NW_3', 'NO_HVG_NW_3', 'HVG_NW_10', 'NO_HVG_NW_10']

    noise_name = ['white_noise_0.01', 'white_noise_0.05', 'white_noise_0.1',
                  'white_noise_0.2', 'white_noise_0.3', 'white_noise_0.4']
    noise_name = noise_name +  ['airport', 'babble', 'car', 'exhibition', 'restaurant', 'station', 'street', 'train', 'noise0.1']

    use_predict_info = None
    prefix_out_name = None
    # Train to obtain optimal parameters for each filter on original sounds
    if mode == 'train_original':
        optimal_params = training_optimal_params_by_original_sounds(
            dir_contained_sounds=dir_contained_original_sounds,
            training_ids=training_ids,
            snr_level=snr_level,
            noise_name=noise_name,
            out_file_name=os.path.join(dir_saving_optimal_params, 'optimal_params_filters_from_original.json')
        )

    # Train to obtain optimal parameters for each filter on predicted sounds
    elif mode == 'train_predicted':
        for info in LogNNet_info_params:
            optimal_params = training_optimal_params_by_predict_sound(
                dir_clear_sound=dir_contained_original_sounds,
                dir_pred_sound=dir_contained_predict_sounds,
                info=info,
                noise_name=noise_name,
                training_ids=training_ids,
                snr_level=snr_level,
                pred_prefix='pred_',
                pred_suffix=f'_{info}',
                out_file_name = os.path.join(dir_saving_optimal_params, f'optimal_params_{info}.json')
            )

            use_predict_info = info
            prefix_out_name = 'filter_after'


    # Load optimal params from original sounds
    # OPTIONS
    #with open(os.path.join(dir_saving_optimal_params, 'optimal_params_filters_from_original.json'), 'r') as json_file:
    #    optimal_params = json.load(json_file)

    # Load optimal params from LogNNet predicted sounds
    #with open(os.path.join(dir_saving_optimal_params, f'optimal_params_{use_predict_info}.json'), 'r') as json_file:
    #    optimal_params = json.load(json_file)

    str_out = 'BY ORIGINAL SOUND' if use_predict_info is None else f'BY LogNNet ({use_predict_info}) PREDICTION SOUND'
    print(f'\nTEST FILTERS {str_out}')

    # Testing: apply optimal parameters to each file from the test set
    result_test = apply_filter_by_optimal_params(dir_contained_sounds=dir_contained_original_sounds,
                                                 dir_pred_sound=dir_contained_predict_sounds,
                                                 snr_level=snr_level,
                                                 test_ids=test_ids,
                                                 optimal_params=optimal_params,
                                                 use_predict_info=use_predict_info,
                                                 prefix_out_name=prefix_out_name)

