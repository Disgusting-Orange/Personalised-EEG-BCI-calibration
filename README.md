# Personalised-EEG-BCI-calibration
Personalized EEG-BCI Calibration using Resting-State EEG and Motor Imagery EEG with Graph Neural Networks.

# Personalized EEG-BCI Calibration using Resting-State EEG and Motor Imagery EEG

## Overview

This project investigates whether Resting-State EEG (RS-EEG) can be used to improve and personalize Motor Imagery Brain-Computer Interface (MI-BCI) calibration.

The objective is to predict or enhance Motor Imagery performance by leveraging neural patterns extracted from resting-state recordings and integrating them with Motor Imagery EEG data.

The project follows a complete EEG processing and machine learning pipeline:

Raw EEG → Preprocessing → Feature Extraction → Baseline Models → Graph Neural Networks

---

## Research Motivation

Traditional Motor Imagery BCIs often require extensive calibration sessions for each user.

This research explores whether information present in Resting-State EEG can:

- Reduce calibration effort
- Improve MI classification performance
- Enable subject-specific adaptation
- Address BCI illiteracy issues

---

## Dataset

Dataset:
PhysioNet EEG Motor Movement/Imagery Dataset

Source:
https://physionet.org/content/eegmmidb/1.0.0/

Characteristics:

- 109 subjects
- 64 EEG channels
- Resting-State EEG
- Motor Execution EEG
- Motor Imagery EEG
- Sampling Frequency: 160 Hz

---

## Preprocessing Pipeline

The preprocessing workflow includes:

### Signal Preparation

- Channel normalization
- Standard EEG montage assignment
- Bad channel detection
- Bad channel interpolation/drop fallback

### Noise Removal

- Bandpass filtering (0.5–45 Hz)
- Notch filtering (60 Hz)
- ICA-based artifact removal
- EOG artifact detection

### Signal Standardization

- Average re-referencing
- Epoch generation (2-second epochs)
- Automatic artifact rejection

Final validated preprocessing parameters:

```python
EPOCH_REJECT = {"eeg": 550e-6}
