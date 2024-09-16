import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, concatenate, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import VGG16
import pickle

image_folder_path = 'C:/Users/Bhavana/Documents/GitHub/image-caption-generator/images/images_small'
csv_file_path = 'C:/Users/Bhavana/Documents/GitHub/image-caption-generator/captions_small.csv'

df = pd.read_csv(csv_file_path)

image_captions_genres = {}

for index, row in df.iterrows():
    image_name = row['img_name'].strip()
    caption = row['caption'].strip()
    genre = row['genre'].strip()
    key = (image_name, genre)
    if key in image_captions_genres:
        image_captions_genres[key].append(caption)
    else:
        image_captions_genres[key] = [caption]

def load_images(image_captions_genres, image_folder_path, target_size=(224, 224)):
    images, captions, genres = [], [], []
    for (image_name, genre), caption_list in image_captions_genres.items():
        image_path = os.path.join(image_folder_path, image_name)
        try:
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            images.append(img_array)
            captions.append(caption_list)
            genres.append(genre)
        except FileNotFoundError:
            print(f"Image {image_name} not found in {image_folder_path}. Skipping.")
        except OSError as e:
            print(f"Error loading image {image_name}: {e}")
    return np.array(images), captions, genres

images, captions, genres = load_images(image_captions_genres, image_folder_path)

tokenizer = Tokenizer()
all_captions = [caption for sublist in captions for caption in sublist]
tokenizer.fit_on_texts(['<start> <end>'] + all_captions)
vocab_size = len(tokenizer.word_index) + 1

num_samples = len(images)
input_sequences = []
target_sequences = []
image_ids = []

for img_id, caption_list in enumerate(captions):
    for caption in caption_list:
        sequence = tokenizer.texts_to_sequences([caption])[0]
        if len(sequence) == 0: 
            continue
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            target_sequences.append(sequence[i])
            image_ids.append(img_id)

if not input_sequences or not target_sequences:
    raise ValueError("No valid sequences found. Please check your captions and tokenization.")

max_length = max(len(seq) for seq in input_sequences)

max_length_path = 'max_length_trained.pkl'
with open(max_length_path, 'wb') as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')

one_hot_targets = np.zeros((len(target_sequences), vocab_size))
for i, target in enumerate(target_sequences):
    if target is not None:
        one_hot_targets[i, target] = 1

padded_input_sequences = np.array(padded_input_sequences)
image_features = np.array([images[i] for i in image_ids])

genres_set = set(genres)
num_genres = len(genres_set)
genre_to_index = {genre: index for index, genre in enumerate(genres_set)}

genres = [genre_to_index[genre] for genre in genres]
genres = np.array(genres)[:, np.newaxis]
genre_sequences=np.array([genres[i] for i in image_ids])
genre_to_int_path = 'genre_to_int_trained.pkl'
with open(genre_to_int_path, 'wb') as handle:
    pickle.dump(genre_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

base_model = VGG16(include_top=False, weights='imagenet')
base_model.trainable = False
image_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

extracted_image_features = image_model.predict(image_features)

assert extracted_image_features.shape[0] == padded_input_sequences.shape[0] == one_hot_targets.shape[0] == genre_sequences.shape[0], "Mismatch in the number of samples among inputs and targets."

image_input = Input(shape=(extracted_image_features.shape[1],))
image_features = Dense(512, activation='relu')(image_input)
# image_features = (max_length)(image_features)

caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(vocab_size, 512)(caption_input)
caption_lstm = LSTM(512)(caption_embedding)

genre_input = Input(shape=(1,), dtype='int32')
genre_embedding = Embedding(input_dim=num_genres, output_dim=50)(genre_input)
genre_embedding_flat = Flatten()(genre_embedding)

decoder_input = concatenate([image_features, caption_lstm, genre_embedding_flat])
decoder_hidden = Dense(512, activation='relu')(decoder_input)
output = Dense(vocab_size, activation='softmax')(decoder_hidden)

model = Model(inputs=[image_input, caption_input, genre_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit([extracted_image_features, padded_input_sequences,genre_sequences], one_hot_targets, epochs=75)

tokenizer_path = 'tokenizer_trained.pkl'
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_path = 'trained_model_trained.h5'
model.save(model_path)