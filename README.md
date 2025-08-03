# Advanced Noise Filtering Techniques with LogNNet: Evaluation Across White and Real-World Noise Types

This repository contains tools for evaluating various noise filtering methods in audio signals, including classical filters and modern LogNNet-based approaches. The system supports both white noise and real-world noise types.

### Citing the Work

... Article...

## Table of Contents

- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
  - [Generating Audio Files with Noise](#generating-audio-files-with-noise)
  - [Training and Testing Filters](#training-and-testing-filters)
  - [Computing Metrics](#computing-metrics)
- [Datasets](#datasets)
- [Complete Usage Cycle](#complete-usage-cycle)
- [License](#license)
- [Contact](#contact)

## Installation and Requirements

### System Requirements
- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- librosa
- soundfile
- matplotlib
- pandas
- pesq
- pystoi

### Installing Dependencies
```bash
pip install numpy scipy scikit-learn librosa soundfile matplotlib pandas pesq pystoi
```

## Usage

### Generating Audio Files with Noise

The `generate_audio_file.py` script creates audio files with various levels of white noise based on clean files from the NOIZEUS dataset.

#### Execution:
```bash
python generate_audio_file.py
```

#### Output File Structure:
The script creates the following folder structure:
```
NOIZEUS/
├── white_noise_0.1/
│   ├── 5dB/
│   │   ├── sp01_white_noise_0.1_sn5.wav
│   │   ├── sp02_white_noise_0.1_sn5.wav
│   │   └── ...
│   ├── 10dB/
│   ├── 15dB/
│   └── 20dB/
├── white_noise_0.2/
├── white_noise_0.3/
└── ...
```

#### Noise Levels:
Files are automatically generated with various noise levels and SNR (signal-to-noise ratio):
- Noise levels: 0.1, 0.2, 0.3, 0.4
- SNR levels: 5dB, 10dB, 15dB, 20dB

### Training and Testing Filters

The main script `filters_audio_signal.py` implements the search for optimal parameters for various classical filters and their testing. **All available filters are always processed automatically**.

#### Execution:
```bash
python filters_audio_signal.py
```

#### Main Configuration Parameters:
Open the `filters_audio_signal.py` file and find the `if __name__ == "__main__":` block. Main parameters to modify:


```python
if __name__ == "__main__":
    # Operation mode (MAIN PARAMETER)
    mode = 'train_predicted'  # Options: 'train_original', 'train_predicted', 'test'
    
    # Directories
    dir_contained_original_sounds = 'NOIZEUS'          # Folder with original sounds
    dir_contained_predict_sounds = 'LogNNet_filtered_sounds'  # Folder with LogNNet predictions
    
    # Dataset parameters
    snr_level = 5              # SNR level in dB (5, 10, 15, 20)
    training_start = 1         # Starting file number for training
    training_end = 4           # Ending file number for training
```

#### Operation Modes:

**1. `mode = 'train_original'`** - Training on original noisy sounds
- Finds optimal filter parameters for cleaning original noisy files
- Saves parameters to `optimal_params/optimal_params_filters_from_original.json`

**2. `mode = 'train_predicted'`** - Training on LogNNet preprocessed sounds
- Finds optimal filter parameters for additional cleaning after LogNNet
- Processes all LogNNet variants: `HVG_NW_1`, `NO_HVG_NW_1`, `HVG_NW_3`, `NO_HVG_NW_3`, `HVG_NW_10`, `NO_HVG_NW_10`
- Saves parameters for each variant separately

**3. `mode = 'test'`** - Testing with found parameters
- Applies found optimal parameters to test files
- Requires prior execution of training mode

#### Supported Noise Types:
The script automatically processes all noise types:
- **White noise**: `white_noise_0.01`, `white_noise_0.05`, `white_noise_0.1`, `white_noise_0.2`, `white_noise_0.3`, `white_noise_0.4`
- **Real-world noise**: `airport`, `babble`, `car`, `exhibition`, `restaurant`, `station`, `street`, `train`

#### Processed Filters:
The script always applies **all available classical filters**:
- Wiener Filter
- Kalman Filter
- Wavelet Filter
- Savitzky-Golay Filter
- Butterworth Filter

#### Output Data:
- **Optimal parameters**: saved in `optimal_params/` folder
- **Filtered files**: created during testing

#### Output File Structure for Filtered Audio:
When running in `test` mode, the script creates filtered audio files in the following structure:
```
NOIZEUS/
└── filter_after/                    # Created when prefix_out_name = 'filter_after'
    ├── wiener/                      # Wiener filter results
    │   ├── 5dB/
    │   │   ├── white_noise_0.1/
    │   │   │   ├── filter_after_sp05_white_noise_0.1_sn5.wav
    │   │   │   ├── filter_after_sp06_white_noise_0.1_sn5.wav
    │   │   │   └── ...
    │   │   ├── airport/
    │   │   ├── babble/
    │   │   └── ...
    │   ├── 10dB/
    │   └── ...
    ├── kalman/                      # Kalman filter results
    ├── wavelet/                     # Wavelet filter results
    ├── savgol/                      # Savitzky-Golay filter results
    └── butterworth/                 # Butterworth filter results
```

#### Filtered File Naming Convention:
- **Format**: `{prefix_out_name}_sp{file_id}_{noise_type}_sn{snr_level}.wav`
- **Example**: `filter_after_sp05_white_noise_0.1_sn5.wav`
  - `filter_after`: prefix indicating post-LogNNet filtering
  - `sp05`: original file identifier (sp05.wav)
  - `white_noise_0.1`: noise type
  - `sn5`: SNR level (5dB)

### Computing Metrics

The `compute_metrics.py` script calculates quality metrics to assess filtering effectiveness and outputs formatted results to the console in tabular format.

#### Execution:
```bash
python compute_metrics.py
```

#### Main Configuration Parameters:
Open the `compute_metrics.py` file and find the `main()` function. Main parameters:

```python
def main():
    # Directories
    dir_contained_original_sounds = 'NOIZEUS'
    dir_contained_predict_sounds = 'LogNNet_filtered_sounds'
    
    # Parameters
    snr_level = 5                    # SNR level (5, 10, 15, 20)
    use_predict_info = 'NO_HVG_NW_1' # LogNNet variant for 'predict' mode
    
    # Operation mode (MAIN PARAMETER)
    mode = 'predict'                 # Options: 'original', 'predict'
    
    # Data split settings
    training_start = 1               # Starting number for training set
    training_end = 4                 # Ending number for training set
```

#### Operation Modes:

**1. `mode = 'original'`** - Calculate metrics for original noisy files
- Compares clean files with noisy files (before processing)
- Shows baseline quality before applying filters

**2. `mode = 'predict'`** - Calculate metrics for LogNNet processed files
- Compares clean files with files after LogNNet processing
- Shows quality after applying neural network filtering
- Requires specifying LogNNet variant in `use_predict_info`

#### Available LogNNet Variants:
- `HVG_NW_1`, `NO_HVG_NW_1`
- `HVG_NW_3`, `NO_HVG_NW_3`  
- `HVG_NW_10`, `NO_HVG_NW_10`

#### Calculated Metrics:
The script automatically calculates for all noise types:
- **R²** - coefficient of determination (maximum value)
- **MAE** - mean absolute error (minimum value)
- **PESQ** - perceptual evaluation of speech quality (maximum value)
- **SNR** - signal-to-noise ratio (maximum value)
- **STOI** - short-time objective intelligibility (maximum value)

#### Calculation Features:
- Metrics are calculated for the test set (files not included in the training set)
- For R², MAE, and SNR, optimization is applied over scaling coefficient k ∈ [0.02, 1.5]
- Average metric values across all test files are output for each noise type

#### Output Format:
```
Computing metrics from predict sounds

noise           "R2"      "MAE"     "PESQ"    "SNR"     "STOI"
white_noise_0.01 0.895123 0.045234 2.856789 18.234567 0.923456
white_noise_0.05 0.823456 0.067890 2.634512 15.678901 0.876543
...
```

## Datasets

### NOIZEUS Dataset
A standard dataset for evaluating speech enhancement algorithms, containing:
- Clean speech recordings
- Recordings with various types of real noise
- Reference data for comparison

**Link**: [NOIZEUS Database](http://ecs.utdallas.edu/loizou/speech/noizeus/)

**Citation**: 
> Hu, Y. and Loizou, P. (2007). "Subjective evaluation and comparison of speech enhancement algorithms," Speech Communication, 49, 588-601.
> 
## Complete Usage Cycle

### Step 1: Data Preparation
```bash
# Ensure the NOIZEUS dataset is in the correct directory
ls NOIZEUS/clean/

# Generate files with noise (creates NOIZEUS/white_noise_X.X/YdB/ structure)
python generate_audio_file.py
```

### Step 2: Finding Optimal Filter Parameters
```bash
# 1. Open filters_audio_signal.py
# 2. Find the if __name__ == "__main__": block
# 3. Set the desired mode:
#    mode = 'train_original'   # for training on original noisy files
#    mode = 'train_predicted'  # for training on LogNNet files (recommended)
#    mode = 'test'            # for testing with found parameters
# 4. Configure SNR level (snr_level = 5, 10, 15, or 20)
# 5. Run the script
python filters_audio_signal.py
```

**Note**: The test mode creates filtered audio files in `NOIZEUS/filter_after/` with organized subdirectories for each filter type, SNR level, and noise condition.

### Step 3: Results Assessment
```bash
# 1. Open compute_metrics.py
# 2. Set the desired mode ('original' or 'predict')
# 3. Configure LogNNet variant if using 'predict' mode
# 4. Run metrics calculation
python compute_metrics.py
```

## License

This project is provided for research and educational purposes.

## Contact

For questions and suggestions, please create an issue in the repository.