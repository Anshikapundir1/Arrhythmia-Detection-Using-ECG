# Arrhythmia-Detection-Using-ECG

This project is a deep learning-based Arrhythmia Detection System developed using the MIT-BIH Arrhythmia Dataset. The system analyzes ECG signals and predicts different heartbeat abnormalities using a Convolutional Neural Network (CNN).

The project also includes a Flask-based web application where users can upload ECG record files (.dat and .hea) and get real-time arrhythmia prediction results.


<img width="1157" height="722" alt="image" src="https://github.com/user-attachments/assets/501759f5-dcfd-4213-9f06-b401917a44d9" />


## Features
- ECG signal analysis using CNN
- MIT-BIH dataset support
- Upload .dat and .hea ECG files
- Real-time heartbeat classification
- Arrhythmia detection status
- Flask web interface
- Deep Learning based prediction

## Dataset Used
The MIT-BIH Arrhythmia Database is one of the earliest publicly available benchmark datasets used for arrhythmia detection research. It contains ECG recordings collected from long-duration Holter monitor recordings. Each ECG record is represented using three files: a header file (.hea), a signal data file (.dat), and an annotation file (.atr). The dataset is widely used for training and evaluating machine learning and deep learning models for heartbeat classification and arrhythmia prediction.

## Model Information
The project uses a Convolutional Neural Network (CNN) model for ECG heartbeat classification and arrhythmia detection. The CNN model is trained on ECG signal segments extracted from the MIT-BIH Arrhythmia Dataset.
The model processes ECG signal data with an input shape of (360,1) and learns important heartbeat patterns using convolution and pooling layers. After feature extraction, fully connected dense layers are used for final heartbeat classification.


## Classes Predicted
- / :- Paced Beat
- A :- Atrial Premature Beat
- E :- Ventricular Escape Beat
- L :- Left Bundle Branch Block Beat
- N :- Normal Beat
- R :- Right Bundle Branch Block Beat
- V :- Ventricular Premature Beat

## Technologies Used
- Python                
- Flask                   
- TensorFlow/Keras        
- WFDB                    
- NumPy                   
- Html/CSS                

## How to Run
- Install requirements
- Run appps.py
- Open localhost link

## Output
<img width="923" height="903" alt="image" src="https://github.com/user-attachments/assets/386066ac-96d1-413d-aa20-a1b478b93ba5" />

