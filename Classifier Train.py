#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print(X_train[1].shape)


# In[ ]:


print(y_train[:3])


# In[ ]:


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


# In[ ]:


model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])


# In[ ]:


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[ ]:


model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epoch=2, batch_size=32)


# In[ ]:


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)


# In[ ]:




