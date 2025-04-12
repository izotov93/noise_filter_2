# Advanced Noise Filtering Techniques with LogNNet: Evaluation Across White and Real-World Noise Types

The repository contains the following directories and files:

Source Code:
* Classical Filters: Implementation of various classical noise reduction techniques applied to intermediate signals, including Wiener, Kalman, Wavelet, Savitzky–Golay, and Butterworth filters,
* White Noise Addition: Code for adding white Gaussian noise at six different levels (0.01, 0.05, 0.1, 0.2, 0.3, and 0.4), representing fractions of the maximum amplitude of the original signal,
* Compute Evaluation Metrics: Functions for calculating evaluation metrics such as R², MAE, PESQ, SNR, and STOI.

Datasets:
* NOIZEUS Dataset [1]: The original Noizeus dataset is provided, which serves as a benchmark for evaluating noise reduction algorithms.
* Generated Audio Samples: This includes audio samples with added noise and the corresponding filtered outputs produced by the LogNNet model.

This repository aims to facilitate further research and development in the field of audio signal processing, particularly in noise reduction techniques. We encourage users to explore the provided materials and contribute to advancements in this area

## Citing the Work

...

[Link to article]()


## References
[1] Hu, Y. and Loizou, P. (2007). “Subjective evaluation and comparison of speech enhancement algorithms,” 
Speech Communication, 49, 588-601.


