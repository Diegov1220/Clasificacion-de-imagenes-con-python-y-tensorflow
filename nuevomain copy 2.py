# %%
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import layers
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# %%
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

# %%
#Crear el dataset generador
width = 80
height = 80
batch=32
data = tf.keras.utils.image_dataset_from_directory('data')

# %%
#Generadores para sets de entrenamiento y pruebas
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(width,height))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])



# %%
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

train_size = int(len(data)*.6)
val_size = int(len(data)*.3)
test_size = int(len(data)*.1)

train_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)



train
    
# datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     rotation_range = 30,
#     width_shift_range = 0.25,
#     height_shift_range = 0.25,
#     shear_range = 15,
#     zoom_range = [0.5, 1.5],
#     validation_split=0.4 #20% para pruebas
# )
    
# train = datagen.flow_from_directory('data/', target_size=(256,256),batch_size=batch, shuffle=True, subset='training')
# val = datagen.flow_from_directory('data/', target_size=(256,256),batch_size=batch, shuffle=True, subset='validation')

# train


# %%
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(Dropout(0.8))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(Dropout(0.8))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(Dropout(0.8))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))


# %%
#modelo.compile(loss='binary_crossentropy',optimizer='adam',metrics =['accuracy'])

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()


# %%
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
historial=model.fit(train, epochs=20,batch_size=batch, validation_data=val, callbacks=[early_stop])

  # %%
#Graficas de precisión
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(20)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

# %%

model = keras.models.load_model('tesis.h5')
img = cv2.imread('D:\\IA_PERROS_GATOS\\test\\000377998W.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat)

if yhat > 0.5: 
    print(f'La prediccion es normal')
else:
    print(f'La prediccion es delito')




# %%
model.save('tesis.h5')

# %%

from keras.metrics import Precision, Recall, F1Score
pre = Precision()
re = Recall()

test = data.skip(train_size+val_size).take(test_size)
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)



print(pre.result() )
print(re.result())
# %%
f1 = 2 * (0.9787234 * 1) / (0.9787234 + 1)
print(f1)
import pandas as pd
metrics_df = pd.DataFrame({
    'Métrica': ['Precisión'],
    'Valor': [pre.result()]
})
metrics_df2 = pd.DataFrame({
    'Métrica': ['Recall',],
    'Valor': [re.result()]
})
metrics_df3 = pd.DataFrame({
    'Métrica': ['F1-score'],
    'Valor': [f1]
})

# %%
# Mostrar el DataFrame
print(metrics_df)
print(metrics_df2)
print(metrics_df3)
# %%
