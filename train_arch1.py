import logging
import time
import os
import pickle


import numpy as np
import matplotlib.pyplot as plt

#import tensorflow_datasets as tfds
import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
#import tensorflow_text  # pylint: disable=unused-import
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib as plt
from tqdm.keras import TqdmCallback


from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model,  Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dropout, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding, Concatenate
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam, Adagrad
from keras.losses import sparse_categorical_crossentropy
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
import random


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name, default):
    value = os.getenv(name)
    return value if value is not None else default

with open("/Users/dominicgoj/coding/pichai/PichiaCLM/Model_PichiaCLM/Training/AllData/Pichia_All_2Target.pkl", "rb") as fp:
    Data_AllOrg = pickle.load(fp)

AA_tr = Data_AllOrg['AA_tr']
Cds_tr = Data_AllOrg['Cds_tr']
AA_ts = Data_AllOrg['AA_ts']
Cds_ts = Data_AllOrg['Cds_ts']

# Token IDs are small; compact integer dtype can reduce RAM significantly.
AA_tr = AA_tr.astype(np.int16, copy=False)
Cds_tr = Cds_tr.astype(np.int16, copy=False)
AA_ts = AA_ts.astype(np.int16, copy=False)
Cds_ts = Cds_ts.astype(np.int16, copy=False)

# Free the source container after extraction to reduce peak memory.
del Data_AllOrg

Settings = pd.read_csv('./BO_forHyperParameter/Arch1/Round3.csv').iloc[:, 1:]
Setting_no = 1


    
Max_length = 1000
learning_rate = 0.001

# Use TRAIN_PRESET for quick runtime modes and override single values via env vars if needed.
train_presets = {
    "low": {
        "batch_size": 16,
        "epochs": 20,
        "train_subset_fraction": 0.1,
        "shuffle_buffer": 300,
        "prefetch_buffer": 1,
        "use_mixed_precision": False,
    },
    "medium": {
        "batch_size": 64,
        "epochs": 60,
        "train_subset_fraction": 0.5,
        "shuffle_buffer": 3000,
        "prefetch_buffer": 2,
        "use_mixed_precision": False,
    },
    "full": {
        "batch_size": 150,
        "epochs": 100,
        "train_subset_fraction": 1.0,
        "shuffle_buffer": 5000,
        "prefetch_buffer": 4,
        "use_mixed_precision": False,
    },
}

train_preset = _env_str("TRAIN_PRESET", "full").strip().lower()
if train_preset not in train_presets:
    raise ValueError(
        f"Unknown TRAIN_PRESET='{train_preset}'. Use one of: {', '.join(train_presets)}"
    )

preset = train_presets[train_preset]

batch_size = _env_int("BATCH_SIZE", preset["batch_size"])
epochs = _env_int("EPOCHS", preset["epochs"])
validation_split = _env_float("VALIDATION_SPLIT", 0.2)
train_subset_fraction = _env_float("TRAIN_SUBSET_FRACTION", preset["train_subset_fraction"])
shuffle_buffer = _env_int("SHUFFLE_BUFFER", preset["shuffle_buffer"])
prefetch_buffer = _env_int("PREFETCH_BUFFER", preset["prefetch_buffer"])
seed = _env_int("SEED", 42)

if not 0.0 < validation_split < 1.0:
    raise ValueError("VALIDATION_SPLIT must be between 0 and 1.")
if not 0.0 < train_subset_fraction <= 1.0:
    raise ValueError("TRAIN_SUBSET_FRACTION must be in (0, 1].")

# Optional speed/memory toggle for supported GPUs.
if _env_bool("USE_MIXED_PRECISION", preset["use_mixed_precision"]):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

print(
    "Training config:",
    {
        "preset": train_preset,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_subset_fraction": train_subset_fraction,
        "validation_split": validation_split,
        "shuffle_buffer": shuffle_buffer,
        "prefetch_buffer": prefetch_buffer,
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
    },
)

aa_vocab_size = 25
dna_vocab_size = 67


hidden_size_enc = int(Settings['Enc hidden size'][Setting_no])
hidden_size_enc_aa = int(Settings['Enc hidden size'][Setting_no])
embedding_size_enc = int(Settings['Enc Embedding size'][Setting_no])
embedding_size_dec = int(Settings['Dec Embedding size'][Setting_no])
Dense_layer_size = int(Settings['Dense Layer size'][Setting_no])
Dense_layer_size_aa = int(Settings['Dense Layer size aa'][Setting_no])

drop_rate = Settings['Drop rate'][Setting_no]
drop_rate_aa = Settings['Drop rate aa'][Setting_no]

input_sequence = Input(shape=(Max_length,))
encod_emb = Embedding(input_dim= aa_vocab_size, output_dim = embedding_size_enc,trainable=True, mask_zero = True)
embedding = encod_emb(input_sequence)

encoder = Bidirectional(GRU(hidden_size_enc, return_sequences=True, return_state = True),
                        merge_mode="concat", weights=None)

encoder_sequence, encoder_final_f, encoder_final_b  = encoder(embedding)

encoder_final = Concatenate(axis=-1)([encoder_final_f, encoder_final_b])



decoder_inputs = Input(shape=(Max_length -1, ))
decoder_inputs_aa = Input(shape=(Max_length, ))

dex=  Embedding(input_dim = dna_vocab_size, output_dim = embedding_size_dec, trainable=True, mask_zero = True)


final_dex= dex(decoder_inputs)
final_dex_aa =  encod_emb(decoder_inputs_aa)


decoder = GRU(2*hidden_size_enc, return_sequences = True, return_state = True)
decoder_aa =  GRU(2*hidden_size_enc_aa, return_sequences = True, return_state = True)

decoder_sequence, decoder_final = decoder(final_dex, initial_state=encoder_final)
decoder_sequence_aa, decoder_final_aa = decoder_aa(final_dex_aa, initial_state=encoder_final)


attn_layer = Attention()
attn_out = attn_layer([decoder_sequence, encoder_sequence])
attn_layer_aa = Attention()
attn_out_aa = attn_layer_aa([decoder_sequence_aa, encoder_sequence])

decoder_concat_input = Concatenate(axis=-1)([decoder_sequence, attn_out]) #decoder_sequence, 
decoder_concat_input_aa = Concatenate(axis=-1)([decoder_sequence_aa, attn_out_aa]) #decoder_sequence,


Intermediate_layer = TimeDistributed(Dense(Dense_layer_size, activation='tanh'))
Intermediate_layer_aa= TimeDistributed(Dense(Dense_layer_size_aa, activation='tanh'))

Intemediate_output = Intermediate_layer(decoder_concat_input) #decoder_concat_input
Intemediate_output_aa = Intermediate_layer_aa(decoder_concat_input_aa) #decoder_concat_input


dropout_layer = Dropout(drop_rate)
dropout_output = dropout_layer(Intemediate_output)

dropout_layer_aa = Dropout(drop_rate_aa)
dropout_output_aa = dropout_layer_aa(Intemediate_output_aa)

dense_layer = TimeDistributed(Dense(dna_vocab_size, activation='softmax', dtype='float32'))
logits = dense_layer(dropout_output)

dense_layer_aa = TimeDistributed(Dense(aa_vocab_size, activation='softmax', dtype='float32'))
logits_aa = dense_layer_aa(dropout_output_aa)

enc_dec_model = Model([input_sequence, decoder_inputs, decoder_inputs_aa], [logits, logits_aa])

enc_dec_model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(learning_rate = learning_rate),
              metrics=['accuracy', 'accuracy'])
enc_dec_model.summary()

checkpoint_path = "/Users/dominicgoj/coding/pichai/PichiaCLM/Model_PichiaCLM/2Target_AllData/FinArch1_AttnCorr_cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience = 3,
    verbose=0, mode="auto", baseline=None, restore_best_weights=False)

# Prepare fixed-width training arrays first to avoid repeated slicing in tf.data.
x_aa_input = AA_tr[:, 1:Max_length+1]
x_cds_input = Cds_tr[:, 0:Max_length-1]
x_aa_decoder = AA_tr[:, 0:Max_length]
y_cds_target = Cds_tr[:, 1:Max_length]
y_aa_target = AA_tr[:, 1:Max_length+1]

# Test arrays are not used in this training script.
del AA_ts, Cds_ts

num_samples = x_aa_input.shape[0]
all_indices = np.arange(num_samples)

if 0 < train_subset_fraction < 1.0:
    subset_size = max(2, int(num_samples * train_subset_fraction))
    rng = np.random.default_rng(seed)
    all_indices = rng.choice(all_indices, size=subset_size, replace=False)

train_idx, val_idx = train_test_split(
    all_indices,
    test_size=validation_split,
    random_state=seed,
    shuffle=True,
)

def _build_dataset(indices, training):
    dataset = tf.data.Dataset.from_tensor_slices((
        (
            x_aa_input[indices],
            x_cds_input[indices],
            x_aa_decoder[indices],
        ),
        (
            y_cds_target[indices],
            y_aa_target[indices],
        ),
    ))
    if training:
        dataset = dataset.shuffle(
            min(shuffle_buffer, len(indices)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(prefetch_buffer)

train_ds = _build_dataset(train_idx, training=True)
val_ds = _build_dataset(val_idx, training=False)

## Train the model
model_results = enc_dec_model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[cp_callback, early_stop, TqdmCallback()],
    verbose=0,
)

