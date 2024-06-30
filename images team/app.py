# Import required libraries
import os
import subprocess
import numpy as np
import librosa
import joblib
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all origins

# Define upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Load the saved scaler and model for genre recognition
scaler = joblib.load("/home/rahul/Desktop/BeatBot/Models/scaler.pkl")
clfr = joblib.load("/home/rahul/Desktop/BeatBot/Models/model.pkl")

# Function to extract audio features for genre recognition
def AudioFeatureExtraction(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y, sr=sr)
    
    features = [
        np.mean(zcr), np.std(zcr),
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.mean(spectral_rolloff), np.std(spectral_rolloff)
    ]
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    return features

# Function to predict genre
def predict_genre(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    features = AudioFeatureExtraction(y, sr)
    features_scaled = scaler.transform([features])
    prediction = clfr.predict(features_scaled)
    return prediction

def load_speaker_model(model_path="/home/rahul/Desktop/BeatBot/Models/speaker_identification_model.pkl"):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Error loading speaker identification model:", str(e))
        return None

speaker_model = load_speaker_model()


# Function to extract MFCC features from audio data
def extract_featuresspeak(audio_data, sample_rate=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


def predict_speaker(audio_file, model):
    try:
        # Load the audio file and extract features
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        features = extract_featuresspeak(audio_data, sample_rate)

        # Predict the speaker
        predicted_speaker = model.predict([features])[0]

        return predicted_speaker
    except Exception as e:
        print("Error predicting speaker:", str(e))
        return None

# Define route for uploading audio
# Define route for uploading audio
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'voiceFile' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['voiceFile']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded audio file as input_audio.wav
        file_path = os.path.join(UPLOAD_FOLDER, "input_audio.wav")
        audio_file.save(file_path)

        return jsonify({'result': 'success', 'message': 'Audio uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define the route for performing genre recognition
@app.route('/perform_genre_recognition', methods=['POST'])
def perform_genre_recognition():
    try:
        # Check if audio file is provided
        if 'voiceFile' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['voiceFile']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded audio file
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(file_path)

        # Predict the genre
        prediction = predict_genre(file_path)
        prediction_list = prediction.tolist()  # Convert NumPy array to list

        return jsonify({'result': 'success', 'prediction': prediction_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Define the route for performing speaker identification
@app.route('/perform_speaker_identification', methods=['POST'])
def perform_speaker_identification():
    try:
        # Check if audio file is provided
        if 'voiceFile' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['voiceFile']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded audio file
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(file_path)

        # Predict the speaker
        predicted_speaker = predict_speaker(file_path, speaker_model)
        
        if predicted_speaker is not None:
            return jsonify({'predicted_speaker': predicted_speaker}), 200
        else:
            return jsonify({'error': 'Failed to predict speaker'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
