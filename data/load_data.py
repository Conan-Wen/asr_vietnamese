import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import string

from data.data_params import *


lowercase_chars = string.ascii_lowercase
accented_chars = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý"
punctuation_chars = string.punctuation
final_chars = lowercase_chars + accented_chars + punctuation_chars + " "
characters = [x for x in final_chars]

char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

padded_shapes = (tf.TensorShape([None, num_mel_bins]), tf.TensorShape([None]))


def load_from_file(file_path, transcription):
    mfcc_binary_data = tf.io.read_file(file_path)
    mfcc_data = tf.io.parse_tensor(mfcc_binary_data, out_type=tf.float32)
    
    label = tf.strings.lower(transcription, encoding="utf-8")
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    
    return mfcc_data, label
    
def load_data(dataSet):
    """Load train dataset or test dataset as tf.data.Dataset.

    Args:
        dataSet (String): "train" or "test" to load train datset or test dataset.

    Raises:
        ValueError: if the dataSet is not "train" or "test", the ValueError will be raised.

    Returns:
        _type_: the tf.data.Dataset.
    """
    if dataSet=="train":
        mfcc_list = train_mfcc_list
    elif dataSet=="test":
        mfcc_list = test_mfcc_list
    else:
        raise ValueError("dataSet parameter must be train or test!")
    
    df = pd.read_csv(mfcc_list)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            list(df["file_path"]),
            list(df["transcription"]),
        )
    )
    
    dataset = (dataset.map(
        load_from_file,
        num_parallel_calls=tf.data.AUTOTUNE
    )).padded_batch(batch_size, padded_shapes=padded_shapes).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
