
# Project: Log Visualization and Anomaly Detection

![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-Project-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![NumPy](https://img.shields.io/badge/Numpy-1.21.2-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3.3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-green.svg)
![Anomaly Detection](https://img.shields.io/badge/Anomaly%20Detection-Enabled-brightgreen.svg)
![Attack Finding](https://img.shields.io/badge/Attack%20Finding-Active-red.svg)


## Overview
This project focuses on visualizing network traffic data and detecting anomalies, which can indicate potential cyber-attacks. The repository includes Jupyter notebooks and model files for conducting these tasks.

## Files in the Repository

### 1. `AnomalyPredictor.ipynb`
- **Purpose**: A Jupyter notebook that includes code for predicting anomalies in network traffic data. This notebook likely contains data preprocessing steps, model training, and evaluation methods.
- **Usage**: 
    - Open the notebook using Jupyter or any compatible environment.
    - Run the cells to preprocess data, train the model, and make predictions on network traffic data.
    - The notebook may require additional data files or configurations to run successfully.

### 2. `LogVisualization.ipynb`
- **Purpose**: This notebook is designed to visualize log data and highlight potential security breaches or anomalies.
- **Usage**:
    - Convert the notebook to a Python script using the command `jupyter nbconvert --to script LogVisualization.ipynb`.
    - Run the script to generate graphs that illustrate attack patterns or anomalies in network logs.
    - You can integrate this script with a Flask web application to display these graphs on a web interface.

### 3. `autoencoder_new.pth`
- **Purpose**: A pre-trained PyTorch model file (Autoencoder) that is likely used for detecting anomalies in network traffic data.
- **Usage**:
    - Load this model in a PyTorch environment using the following code:
    ```python
    import torch
    model = torch.load('autoencoder_new.pth')
    model.eval()
    ```
    - Use this model within the `AnomalyPredictor.ipynb` or a similar script to detect anomalies in the dataset.

### 4. `classifier_model.pth`
- **Purpose**: A pre-trained classifier model for categorizing network traffic data or detecting specific types of attacks.
- **Usage**:
    - Load this model in a PyTorch environment using the following code:
    ```python
    import torch
    model = torch.load('classifier_model.pth')
    model.eval()
    ```
    - This model can be used in conjunction with the `AnomalyPredictor.ipynb` to classify network traffic data.

## Setting Up the Environment

1. **Install Python**: Make sure Python 3.x is installed on your system.
2. **Install Dependencies**: You can install the necessary Python packages using the following command:
    ```bash
    pip install -r requirements.txt
    ```
   Ensure that the `requirements.txt` file includes the required packages like `torch`, `matplotlib`, and `flask`.

3. **Run the Notebooks**: You can run the provided notebooks using Jupyter or convert them to Python scripts for integration into a larger project.

## Running the Flask Application

1. Convert the `LogVisualization.ipynb` notebook to a Python script using:
    ```bash
    jupyter nbconvert --to script LogVisualization.ipynb
    ```
2. Create a Flask application (`app.py`) to serve the visualizations.
3. Run the Flask app:
    ```bash
    python app.py
    ```
4. Access the web interface by navigating to `http://127.0.0.1:5000/` in your browser.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or issues, please contact the project maintainer.

