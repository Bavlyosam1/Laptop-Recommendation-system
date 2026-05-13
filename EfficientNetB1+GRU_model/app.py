import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
import random

# Load everything
@st.cache_resource
def load_everything():
    model = load_model("caption_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_length.pkl", "rb") as f:
        max_length = pickle.load(f)
    return model, tokenizer, max_length

model, tokenizer, max_length = load_everything()

# Generate caption function
def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([photo_feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred[0])
        word = tokenizer.index_word.get(y_pred)
        if word is None:
            break
        if in_text.split().count(word) > 2:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq", "").replace("endseq", "").strip()

def preprocess_image(img):
    img = img.resize((240, 240))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Find similar captions from dataset
def find_similar_images(caption, images_folder, model, tokenizer, max_length, top_k=3):
    caption_words = set(caption.lower().split())
    similar = []

    # Sample random images from dataset
    all_imgs = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    sampled = random.sample(all_imgs, min(100, len(all_imgs)))

    for img_name in sampled:
        img_path = images_folder + "/" + img_name
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = preprocess_image(img)
            cap = generate_caption(model, tokenizer, img_array, max_length)

            # Count word overlap
            cap_words = set(cap.lower().split())
            overlap = len(caption_words & cap_words)

            if overlap > 0:
                similar.append((img_path, cap, overlap))
        except:
            continue

    # Sort by overlap score
    similar = sorted(similar, key=lambda x: x[2], reverse=True)
    return similar[:top_k]

# App UI
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and the model will generate a caption for it.")

# Optional dataset path
images_folder = st.text_input(
    "📁 Dataset images folder path (optional — for similar image recommendations):",
    placeholder="e.g. E:/Downloads/flickr30k_images"
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    img_array = preprocess_image(img)

    with st.spinner("Generating caption..."):
        caption = generate_caption(model, tokenizer, img_array, max_length)

    st.success(f"**Generated Caption:** {caption}")

    # Show similar images from dataset
    if images_folder and os.path.exists(images_folder):
        st.subheader("🔍 Similar Images from Dataset")
        with st.spinner("Finding similar images..."):
            similar = find_similar_images(
                caption, images_folder, model, tokenizer, max_length
            )

        if similar:
            cols = st.columns(3)
            for i, (img_path, cap, score) in enumerate(similar):
                with cols[i]:
                    sim_img = Image.open(img_path)
                    st.image(sim_img, width=200)
                    st.caption(f"**Caption:** {cap}")
        else:
            st.write("No similar images found.")
    elif images_folder:
        st.warning("Dataset folder not found. Check the path.")