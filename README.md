# InterSync: Interpersonal Synchrony Analysis Tool

## Overview

Evaluation of Interpersonal Synchronization between Individuals based on Video Recording
Capstone Project Phase B
P. 24-1-R-17

The **InterSync** project is a robust tool designed for detecting and analyzing **Interpersonal Synchrony (IS)** using advanced computer vision techniques. By leveraging Google's **MediaPipe** for body landmark detection and implementing various algorithms such as **Dynamic Time Warping (DTW)**, **Time-Lagged Cross-Correlation (TLCC)**, and an adapted version of the **Smith-Waterman** algorithm, InterSync offers a granular analysis of movement synchrony between individuals.


---

## Features

- **Detailed Body Part Synchrony Analysis**: Tracks and analyzes individual body part movements (hands, legs, etc.) instead of general motion analysis.
- **Multiple Synchrony Algorithms**:
  - **Dynamic Time Warping (DTW)** for global sequence alignment.
  - **Time-Lagged Cross-Correlation (TLCC)** for time-shifted synchronization analysis.
  - **Smith-Waterman Algorithm** for detecting localized movement synchrony.
- **Automated Report Generation**: Generates reports in **PDF**, **Excel**, and **CSV** formats containing visual graphs and detailed numerical data.
- **Customizable Synchrony Metrics**: Allows users to define specific body parts for analysis and select the best-fit algorithm for their use case.
- **Graphical User Interface (GUI)**: Built using **PyQt6** for ease of use, even for non-technical users.

---

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.12 or higher

### Required Python Libraries:

fastdtw==0.3.4

joblib==1.4.2

matplotlib==3.9.2

mediapipe==0.10.14

numpy==2.1.1

opencv_contrib_python==4.10.0.84

openpyxl==3.1.5

pandas==2.2.2

PeakUtils==1.3.5

PyQt6==6.7.1

PyQt6_sip==13.8.0

reportlab==4.2.2

scipy==1.14.1

seaborn==0.13.2

---

## Installation

1. **Install Python**: Ensure Python 3.12 is installed on your system. This is important - some of the libraries used will not work otherwise
2. **Clone the Repository**:
    - git clone https://github.com/Dana-Bet/InterSync.git
    - download the repo as zip
4. **Run the Setup / Startup Script**:
   
  start.bat

---

## Usage
1. **Open the Application**: Run the application using the provided GUI.
2. **Load Video Files**: Upload a video of the subjects whose synchrony is to be analyzed.
3. **Select Analysis Method**: Choose the body parts to be analyzed (e.g., arms, legs) and select the algorithm: DTW, TLCC, Smith-Waterman, or simple similarity.
4. **Generate Reports**: The tool generates a report summarizing the synchrony metrics and provides both raw data (CSV/Excel) and visual graphs (PDF).

---

## Outputs
**Summary Reports**: PDF, Excel, or CSV files summarizing synchronization analysis results.
**Visual Graphs**: Displays key metrics for body part movement synchrony with customizable thresholds.
**Data Export**: Raw data for custom analysis or further processing.



