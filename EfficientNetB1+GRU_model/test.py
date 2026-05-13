from tensorflow.keras.models import load_model
import pickle

model = load_model("caption_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)