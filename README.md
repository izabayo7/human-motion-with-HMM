# Human Activity Recognition Using Hidden Markov Models

## Overview

This project implements a Hidden Markov Model (HMM) to classify four human activities — **Standing**, **Walking**, **Jumping**, and **Still** — from smartphone accelerometer and gyroscope data.

Built for the Machine Learning course at African Leadership University (Formative 2).

## Team

| Member | Role |
|--------|------|
| Cedric Izabayo | Data collection (iPhone 17 Pro), HMM implementation, Viterbi decoding |
| Pauline Ishimwe | Data collection (Sony SO-52C), feature extraction, model evaluation |

## Data Collection

- **53 total recordings** across 4 activities (Standing, Walking, Jumping, Still)
- **Cedric:** 28 recordings using iPhone 17 Pro (iOS), 100 Hz sampling rate
- **Pauline:** 25 recordings using Sony SO-52C (Android), 100 Hz sampling rate
- Both phones use the same 100 Hz sampling rate — no resampling needed
- Recorded using the Sensor Logger app

## Project Structure

```
├── data/
│   ├── cedric/          # 28 recordings (7 per activity)
│   └── pauline/         # 25 recordings (6-7 per activity)
├── hmm_activity_recognition.ipynb   # Main notebook
├── report.pdf                       # 4-5 page project report
├── raw_signals.png                  # Raw sensor signal plots
├── convergence.png                  # Baum-Welch convergence plot
├── transition_matrix.png            # Learned transition matrix heatmap
├── confusion_matrix.png             # Test set confusion matrix
├── emission_distributions.png       # Gaussian emission distributions
├── decoded_sequences.png            # Viterbi decoded activity sequences
├── emission_means.png               # Emission means bar chart
└── README.md
```

## Features Extracted

5 features per 1-second sliding window (100 samples, 50% overlap):

| Feature | Domain | Description |
|---------|--------|-------------|
| ACC_RMS | Time | Root mean square of accelerometer magnitude |
| ACC_STD | Time | Standard deviation of accelerometer magnitude |
| GYRO_SMA | Time | Signal magnitude area of gyroscope |
| Dominant Frequency | Frequency (FFT) | Frequency with highest spectral power |
| Spectral Energy | Frequency (FFT) | Total energy in the frequency domain |

All features are Z-score normalized.

## Results

The GaussianHMM achieves **99.1% accuracy** on unseen test data:

| Activity | Sensitivity | Specificity |
|----------|-------------|-------------|
| Standing | 1.0000 | 0.9886 |
| Walking  | 1.0000 | 1.0000 |
| Jumping  | 1.0000 | 1.0000 |
| Still    | 0.9643 | 1.0000 |

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scipy hmmlearn scikit-learn
jupyter notebook hmm_activity_recognition.ipynb
```

Run all cells top-to-bottom. The notebook generates all visualizations and the evaluation metrics.

## Dependencies

- Python 3.10+
- numpy, pandas, matplotlib, seaborn
- scipy (FFT)
- hmmlearn (GaussianHMM)
- scikit-learn (metrics)
