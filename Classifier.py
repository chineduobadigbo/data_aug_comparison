import tensorflow as tf

# used to evaluate how effective the generators are for creating new training data
class Classifier(tf.keras.Model):
    def __init__(self, input_dim, label_num, name):
        super(Classifier, self).__init__()
        self.train_info = {}
        self.model_name = name
        self.model = tf.keras.models.Sequential(name=name)
        self.model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_dim))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(label_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy']) # lr=0.001

    def delete_models(self):
        self.model.reset_states()
        del self.model

    def __del__(self):
        self.delete_models()
        print("Deleted Classifier.")