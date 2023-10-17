import tensorflow as tf
from model.eval_callback import CallbackEval
from model.train_params import *

class Train:
    def __init__(self, model, loss_fn, optimizer, train_dataset, validation_dataset=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
        )
        
    def get_callbacks(self):
        validation_callback = CallbackEval(self.model, self.validation_dataset)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        
        callbacks_list = [validation_callback, model_checkpoint_callback]
        
        return callbacks_list
    
    def train(self, epochs=default_epochs):
        callbacks_list = self.get_callbacks()
        
        history = self.model.fit(
            self.train_dataset,
            # validation_split=validation_split_rate,
            validation_data=self.validation_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
        )
        
        return history