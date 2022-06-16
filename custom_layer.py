import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, layers, Sequential
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


class RBFLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=initializers.RandomUniform(0, 1),
                                       trainable=True)
        self.sigmas = self.add_weight(name='sigmas',
                                     shape=(self.output_dim,),
                                     initializer='ones',
                                     trainable=True)
        super().build(input_shape)

    def call(self, x):
        # here forward propagation takes place:
        C = tf.expand_dims(self.centers, -1)  
        temp = tf.norm(tf.transpose(C-tf.transpose(x)), axis=1)  
        return tf.exp(-1 / self.sigmas * temp)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Reading .xlsx file:
df = pd.read_excel("./iris.xlsx", header=None)
data_raw = df.to_numpy()

# preprocessing:
X = data_raw[:, 0:4]
y = data_raw[:, 4] - 1  

class1_X, class1_y = X[0:50], y[0:50]
class2_X, class2_y = X[50:100], y[50:100]
class3_X, class3_y = X[100:150], y[100:150]

X_train1, X_test1, y_train1, y_test1 = train_test_split(class1_X, class1_y, test_size=0.25, random_state=2, shuffle=True)
X_train2, X_test2, y_train2, y_test2 = train_test_split(class2_X, class2_y, test_size=0.25, random_state=2, shuffle=True)
X_train3, X_test3, y_train3, y_test3 = train_test_split(class3_X, class3_y, test_size=0.25, random_state=2, shuffle=True)

X_train = np.append(X_train1, X_train2, axis=0)
X_train = np.append(X_train3, X_train, axis=0)
X_test = np.append(X_test1, X_test2, axis=0)
X_test = np.append(X_test, X_test3, axis=0)
y_train = np.append(y_train1, y_train2, axis=0)
y_train = np.append(y_train3, y_train, axis=0)
y_test = np.append(y_test1, y_test2, axis=0)
y_test = np.append(y_test, y_test3, axis=0)

# one hot encoding:
enc = OneHotEncoder(handle_unknown='ignore')
y_train_ohe = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test_ohe = enc.fit_transform(y_test.reshape(-1,1)).toarray()


# ceating a model and training:
model = Sequential([RBFLayer(3, input_shape=(4,)),
                    layers.Dense(units=3, activation="softmax"), ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_ohe, epochs=500,
                validation_data=(X_test, y_test_ohe))

# maximum neuron determines the class
y_train_predicted = np.argmax(model.predict(X_train), axis=1)
y_test_predicted = np.argmax(model.predict(X_test), axis=1)

print('\nconfusion matrix(Train):')
print(confusion_matrix(y_train, y_train_predicted), '\n')
print('confusion matrix(Test):')
print(confusion_matrix(y_test, y_test_predicted))


# a little of visualization to understand dataset:

# import matplotlib.pyplot as plt
# from seaborn import pairplot

# df = df.rename(columns={0:"Sepal.Length", 1:"Sepal.Width",
#                          2:"Petal.Length", 3:"Petal.Width", 4:"Species"})
# df['Species'] = df["Species"].replace(1,"setosa")
# df['Species'] = df["Species"].replace(2,"versicolor")
# df['Species'] = df["Species"].replace(3,"virginica")

# pairplot(df, hue="Species")
# plt.show()
