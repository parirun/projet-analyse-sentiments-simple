from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Charger le modèle et le vectoriseur sauvegardés
model = joblib.load("model.pkl")  # Modèle RandomForestClassifier
vectorizer = joblib.load("vectorizer.pkl")  # TfidfVectorizer avec max_features=20

# Initialiser l'application FastAPI
app = FastAPI()

# Modèle de données pour valider l'entrée utilisateur
class Tweet(BaseModel):
    text: str  # Le tweet à analyser

@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    try:
        # Transformer le tweet en représentation numérique
        transformed_tweet = vectorizer.transform([tweet.text])

        # Faire une prédiction
        prediction = model.predict(transformed_tweet)

        # Convertir la prédiction en texte
        sentiment = "positif" if prediction[0] == 1 else "négatif"

        return {"tweet": tweet.text, "sentiment": sentiment}

    except Exception as e:
        return {"error": str(e)}