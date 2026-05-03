import pickle
import numpy as np
import os
from django.shortcuts import render

# Build paths to pkl files (they're in the root directory, one level above spotify_app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = pickle.load(open(os.path.join(BASE_DIR, 'spotify_popularity_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
feature_columns = pickle.load(open(os.path.join(BASE_DIR, 'feature_columns.pkl'), 'rb'))

def predict(request):
    prediction = None

    if request.method == 'POST':
        try:
            # Grab inputs from the form
            danceability = float(request.POST.get('danceability'))
            energy = float(request.POST.get('energy'))
            loudness = float(request.POST.get('loudness'))
            speechiness = float(request.POST.get('speechiness'))
            acousticness = float(request.POST.get('acousticness'))
            instrumentalness = float(request.POST.get('instrumentalness'))
            liveness = float(request.POST.get('liveness'))
            valence = float(request.POST.get('valence'))
            tempo = float(request.POST.get('tempo'))
            explicit = int(request.POST.get('explicit'))
            key = int(request.POST.get('key'))
            mode = int(request.POST.get('mode'))
            time_signature = int(request.POST.get('time_signature'))
            genre_encoded = float(request.POST.get('genre_encoded'))
            duration_min = float(request.POST.get('duration_min'))

            # Engineered features
            energy_acoustic_ratio = energy / (acousticness + 1e-6)
            dance_valence = danceability * valence
            loudness_norm = loudness * -1

            # Build input array in the same column order as training
            input_data = {
                'danceability': danceability,
                'energy': energy,
                'key': key,
                'loudness': loudness,
                'mode': mode,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': tempo,
                'time_signature': time_signature,
                'explicit': explicit,
                'genre_encoded': genre_encoded,
                'energy_acoustic_ratio': energy_acoustic_ratio,
                'dance_valence': dance_valence,
                'loudness_norm': loudness_norm,
                'duration_min': duration_min,
            }

            input_array = np.array([[input_data[col] for col in feature_columns]])
            input_scaled = scaler.transform(input_array)
            result = model.predict(input_scaled)[0]
            prediction = round(float(result), 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render(request, 'predictor/predict.html', {'prediction': prediction})