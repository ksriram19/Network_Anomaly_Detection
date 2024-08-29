from flask import Flask, render_template, request
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)

# Load the models (adjust paths if needed)
autoencoder = torch.load('models/autoencoder_new.pth')
classifier = torch.load('models/classifier_model.pth')

# Define the attack types and fit the LabelEncoder
attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']  # Update with your actual attack types
label_encoder = LabelEncoder()
label_encoder.fit(attack_types)

# Define categorical features for encoding
categorical_features = {
    'protocol_type': ['tcp', 'udp', 'icmp'],
    'service': ['http', 'ftp', 'smtp', 'private'],  # Add more services as needed
    'flag': ['S0', 'S1', 'SF', 'REJ']  # Add more flags as needed
}

# Step 1: Load and Preprocess the KDD Cup 1999 Dataset

# Define the column names as per the dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Load the dataset
data_path = 'filtered_dataset.csv'  # Adjust this path accordingly
data = pd.read_csv(data_path, header=None, names=column_names)

# Split the dataset into features and labels
X = data.drop(columns=['label'])
y = data['label']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit the scaler on the training data and transform the training and test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure the test data has the same columns as the train data
# Reindex to align with the training data and fill missing columns with zeros
X_test_transformed = pd.DataFrame(X_test_transformed).reindex(
    columns=pd.DataFrame(X_train_transformed).columns, fill_value=0
).values

def predict_attack_type_pytorch(classifier, autoencoder, label_encoder, categorical_features, user_input):
    # Convert user input to DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Align user input with training data columns and fill missing columns
    user_df = pd.get_dummies(user_df, columns=categorical_features)
    # Assuming X_train_transformed is the transformed training data used for alignment
    user_df = user_df.reindex(columns=pd.DataFrame(X_train_transformed).columns, fill_value=0)

    # Handle any potential misalignment in dtypes
    for col in user_df.columns:
        if user_df[col].dtype == 'O':
            user_df[col] = user_df[col].astype(float)

    # Transform using the preprocessor (ensure compatibility)
    scaled_user_input = user_df.values

    # Convert to tensor for prediction
    user_input_tensor = torch.tensor(scaled_user_input, dtype=torch.float32)
    
    # Reduce the dimensions using the autoencoder
    with torch.no_grad():
        encoded_input = autoencoder.encode(user_input_tensor)
    
    # Predict the attack type using the classifier
    classifier.eval()
    with torch.no_grad():
        output = classifier(encoded_input)
        _, predicted = torch.max(output, 1)
    
    # Decode the predicted label back to the attack type
    predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]
    
    return predicted_label



@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_attack = None
    if request.method == 'POST':
        # Extract input data from the form
        user_input = {key: request.form[key] for key in request.form.keys()}
        
        # Predict the attack type
        predicted_attack = predict_attack_type_pytorch(classifier, autoencoder, label_encoder, categorical_features, user_input)

    return render_template('index.html', predicted_attack=predicted_attack)

if __name__ == '__main__':
    app.run(debug=True)
