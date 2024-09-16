import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
import pickle
from nltk.translate.bleu_score import corpus_bleu
import os

# Load necessary files and models
tokenizer_path = 'tokenizer_trained.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

model_path = 'trained_model_trained.h5'
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy')

base_model = VGG16(include_top=False, weights='imagenet')
base_model.trainable = False
image_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

max_length_path = 'max_length_trained.pkl'
with open(max_length_path, 'rb') as handle:
    max_length = pickle.load(handle)

genre_to_int_path = 'genre_to_int_trained.pkl'
with open(genre_to_int_path, 'rb') as handle:
    genre_to_index = pickle.load(handle)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) / 255.0
    return img_array

# Function to generate captions
def generate_caption(model, tokenizer, image, genre, max_length, num_captions=5, temperature=0.7):
    def sample(preds, temperature=0.7):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    captions = []
    for _ in range(num_captions):
        caption = '<start>'
        for _ in range(max_length):
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding='post')
            genre_seq = np.array([genre])
            yhat = model.predict([image, seq, genre_seq], verbose=0)[0]
            yhat = sample(yhat, temperature)
            word = tokenizer.index_word.get(yhat, None)
            if word is None:
                break
            caption += ' ' + word
            if word == 'end':
                break
        final_caption = caption.split()
        final_caption = final_caption[1:-1]  # Remove 'startseq' and 'endseq'
        captions.append(' '.join(final_caption))
    return captions

# Function to evaluate BLEU score
def evaluate_bleu_score(df, num_captions=5, temperature=0.7):
    references = []
    hypotheses = []

    for index, row in df.iterrows():
        image_name = row['img_name']
        true_caption = row['caption']
        genre = row['genre'].strip()  # Strip leading/trailing spaces

        # Check if genre exists in genre_to_index
        if genre not in genre_to_index:
            print(f"Warning: Genre '{genre}' not found in genre_to_index.")
            continue

        # Preprocess the image and generate captions
        image_path = os.path.join('images/images_small', image_name)
        image = preprocess_image(image_path)
        image_features = image_model.predict(image, verbose=0)
        genre_index = genre_to_index[genre]

        # Generate multiple captions
        predicted_captions = generate_caption(model, tokenizer, image_features, genre_index, max_length, num_captions, temperature)

        # Add the true caption and predicted captions to the BLEU score calculation
        references.append([true_caption.split()])
        hypotheses.append(predicted_captions[0].split())  # Using the first generated caption for comparison

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

# Load the dataset
df = pd.read_csv('captions_small.csv')

# Evaluate BLEU score
bleu_score = evaluate_bleu_score(df, num_captions=5, temperature=0.7)
print(f"BLEU Score: {bleu_score:.4f}")

