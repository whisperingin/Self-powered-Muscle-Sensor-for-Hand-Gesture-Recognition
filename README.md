# Self-powered-Muscle-Sensor-for-Hand-Gesture-Recognition
Developed a wearable hybrid-powered sEMG hand-gesture recognition system integrating a low-noise four-channel analogue front end, solar energy harvesting with battery backup, STM32-based embedded acquisition, and a 1D-CNN classifier for accurate recognition of seven hand gestures across ten recording sets.
# Self-Powered sEMG Hand Gesture Recognition System

This repository contains the hardware, embedded firmware, data processing scripts, training code, and visualisation tools for a self-powered surface electromyography (sEMG) hand gesture recognition system. The project integrates multi-channel sEMG acquisition, digital signal processing, convolutional neural network (CNN) based gesture classification, interactive hand-model visualisation, and solar energy harvesting hardware.

## Project Overview

The system is designed to acquire multi-channel sEMG signals, preprocess and filter the recorded data, train a CNN model for gesture recognition, and visualise the predicted hand gestures through an interactive hand model. The repository also includes PCB design files for different analogue front-end configurations and a solar energy harvesting board.

The supported gesture set includes:

- Static Rest
- Fist
- Spread
- Flexion
- Extension
- Pronation
- Supination

---

## Repository Structure

### 1. `1 channel.zip`
PCB design files for the **1-channel sEMG analogue front-end board**.

This version is intended for single-channel signal acquisition and hardware verification. It can be used for basic analogue front-end testing, filter validation, and initial signal quality evaluation.

---

### 2. `2 channel.zip`
PCB design files for the **2-channel sEMG analogue front-end board**.

This board extends the design to two channels and is suitable for multi-muscle acquisition with a more compact hardware implementation than the full 4-channel version.

---

### 3. `4channel.zip`
PCB design files for the **4-channel sEMG analogue front-end board**.

This version is the full multi-channel acquisition hardware for the gesture recognition system. It supports four recording channels for capturing richer muscle activity across different gesture classes.

---

### 4. `CNN_model.py`
Python script for **CNN training**.

This file contains the model definition and training pipeline for classifying sEMG signals into the seven gesture classes. It is used after preprocessing the acquired dataset into a suitable format for supervised learning.

Main functions include:

- loading preprocessed data
- defining the CNN architecture
- training and validation
- model saving
- evaluation of classification performance

---

### 5. `Hand Model.html`
Interactive **3D hand model** for gesture visualisation.

This file provides a visual representation of the hand gestures. It is used to display gesture states and can be linked with prediction results for interactive demonstration.

---

### 6. `Pre-process.py`
Python script for **training data preprocessing**.

This script processes the raw data collected through `sEMG_acquisition.html` and prepares it for CNN training. Typical steps may include:

- data cleaning
- segmentation
- channel selection
- filtering or normalisation
- window generation
- label formatting

This file is a key step between raw acquisition and model training.

---

### 7. `README.md`
This documentation file.

It summarises the contents of the repository, explains the role of each file, and provides a high-level guide to the complete workflow of the project.

---

### 8. `predict_interactive.py`
Python script for **interactive prediction and hand-model connection**.

This file loads the trained model and uses the processed sEMG input to generate gesture predictions. The prediction output can then be linked to the hand model for real-time or offline visualisation of recognised gestures.

---

### 9. `sEMG_acquisition.html`
HTML-based tool for **4-channel sEMG acquisition and visualisation**.

This file is used to collect and display sEMG signals. It provides an interface for visualising acquired data and generating raw datasets for later preprocessing and training.

It is intended as the front-end acquisition and signal display component of the project.

---

### 10. `sEMG_project.zip`
Embedded project files for the **STM32F401RE** platform.

This archive contains the embedded firmware related to:

- ADC-based multi-channel sEMG sampling
- digital filtering
- embedded signal acquisition pipeline

The firmware forms the low-level real-time acquisition layer of the system.

---

### 11. `solar PCB.zip`
PCB design files for the **solar energy harvesting board**.

This board is designed to support the self-powered aspect of the system. It provides the hardware platform for photovoltaic energy harvesting and power management for wearable or portable operation.

---

## Typical Workflow

A typical project workflow is as follows:
1. Use the hardware boards in `1 channel.zip`, `2 channel.zip`, `4channel.zip`, and `solar PCB.zip` in the Kicadfor physical implementation
2. Deploy acquisition and filtering on the STM32F401RE firmware in `sEMG_project.zip`
3. Acquire raw sEMG data using `sEMG_acquisition.html`
4. Preprocess the raw data using `Pre-process.py`
5. Train the gesture classifier using `CNN_model.py`
6. Run inference using `predict_interactive.py`
7. Visualise the recognised gesture using `Hand Model.html`


---

## Hardware Contents

This repository includes several hardware design files:

- 1-channel sEMG board
- 2-channel sEMG board
- 4-channel sEMG board
- solar energy harvesting board

These files support the analogue front-end and self-powered implementation of the complete system.

---

## Software Contents

This repository includes software tools for:

- sEMG acquisition visualisation
- data preprocessing
- CNN-based training
- interactive prediction
- gesture visualisation

---

## Embedded System

The embedded implementation is based on the **STM32F401RE** microcontroller and includes:

- ADC sampling
- digital filtering
- sEMG signal acquisition

This allows the system to move towards real-time embedded gesture recognition.

---

## Notes

- The zipped hardware and embedded project files should be extracted before use.
- Python dependencies may need to be installed before running the training and preprocessing scripts.
- The acquisition, preprocessing, training, and prediction stages are intended to work as one complete pipeline.
  welcome to connect me with this email: wangxigan2004@163.com
---

## Author

Project repository for sEMG-based hand gesture recognition, embedded acquisition, and self-powered wearable system development.
