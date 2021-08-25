from extract_training_data import FeatureExtractor
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Embedding, Dense


# Last Modified by Zhaoze Zhang @ 08:49 pm Nov/15

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()
    # model.add(...)
    model.add(Embedding(input_dim=word_types, output_dim=32, input_length=6))
    model.add(Flatten())
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=91, activation="softmax"))
    model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")
    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")

    # Now train the model
    # To make the model perform better (higher score in evaluation,
    # I changed epochs to 10 and batch_size to 1000
    model.fit(inputs, outputs, epochs=10, batch_size=1000)

    model.save(sys.argv[3])
