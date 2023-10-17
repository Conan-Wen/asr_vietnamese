import tensorflow as tf
import numpy as np
from jiwer import wer


def decode_batch_predictions(model, pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for result in results:
        result = tf.strings.reduce_join(model.num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)

    return output_text

class CallbackEval(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = decode_batch_predictions(self.model, batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(self.model.num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)