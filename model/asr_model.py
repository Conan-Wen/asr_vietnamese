import tensorflow as tf


class Asr_model(tf.keras.Model):
    def __init__(self, input_dim, output_dim, rnn_layers_num=5, rnn_units_num=128, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dum = input_dim
        self.output_dim = output_dim
        self.rnn_layers_num = rnn_layers_num
        self.rnn_units_num = rnn_units_num
        
        self.input_layer = tf.keras.layers.Reshape(
            target_shape=(-1, self.input_dum, 1),
            name="input_layer",
            input_shape=(None, self.input_dum),
        )
        self.conv2D_layer1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[14, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )
        self.bn_layer1 = tf.keras.layers.BatchNormalization(
            name="bn1",
        )
        self.relu1 = tf.keras.layers.ReLU(
            name="relu1",
        )
        self.conv2D_layer2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[14, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )
        self.bn_layer2 = tf.keras.layers.BatchNormalization(
            name="bn2",
        )
        self.relu2 = tf.keras.layers.ReLU(
            name="relu2",
        )
        self.reshape_layer = tf.keras.layers.Reshape(
            target_shape=(1, -1),
            name="reshape_layer",
        )
        
        self.rnn_layers = []
        for i in range(self.rnn_layers_num):
            recurrent = tf.keras.layers.GRU(
                units=self.rnn_units_num,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_layer{i+1}",
            )
            self.rnn_layers.append(tf.keras.layers.Bidirectional(
                recurrent,
                name=f"bidirectional_{i+1}",
                merge_mode="concat",
            ))
            
            if i < self.rnn_layers_num:
                self.rnn_layers.append(tf.keras.layers.Dropout(
                    rate=0.5
                ))
                
        self.dense_layer1 = tf.keras.layers.Dense(
            units=self.rnn_units_num*2,
            name="dense_layer1",
        )
        
        self.relu3 = tf.keras.layers.ReLU(
            name="relu3",
        )
        
        self.dense_layer_drop = tf.keras.layers.Dropout(
            rate=0.5,
            name="dense_layer_drop",
        )
           
        self.output_layer = tf.keras.layers.Dense(
            units=self.output_dim+1,
            activation="softmax",
            name="output_layer"
        )
        
    def call(self, inputs):
        x = self.input_layer(inputs)
        
        x = self.conv2D_layer1(x)
        x = self.bn_layer1(x)
        x = self.relu1(x)
        
        x = self.conv2D_layer2(x)
        x = self.bn_layer2(x)
        x = self.relu2(x)
        
        x = self.reshape_layer(x)
        
        for layer in self.rnn_layers:
            x = layer(x)
        
        x = self.dense_layer1(x)
        x = self.relu3(x)
        x = self.dense_layer_drop(x)
        x = self.output_layer(x)
        
        return x
