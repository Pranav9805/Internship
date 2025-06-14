import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input,Dropout,BatchNormalization


 

dataset = "D:\Sign_Language_Detector\Indian"


img_width = 32
img_height = 32

batch_size = 64
seed = 123

train_ds = image_dataset_from_directory(
    dataset,
    validation_split = 0.2,
    subset = "training",
    seed = seed,
    image_size = (img_width, img_height),
    batch_size = batch_size

)

validation_ds = image_dataset_from_directory(
    dataset,
    validation_split = 0.2,
    subset = "validation",
    seed = seed,
    image_size = (img_width,img_height),
    batch_size=batch_size

)


val_batches = tf.data.experimental.cardinality(validation_ds)

val_ds = validation_ds.take(val_batches//2)
test_ds = validation_ds.take(val_batches//2)


model = Sequential([
     Input(shape=(32, 32, 3)),
     Conv2D(512,(3,3),activation = "relu", padding ="same",),
     BatchNormalization(),
     MaxPooling2D(2,2),
     Dropout(0.4),
     Conv2D(256,(3,3),activation = "relu",padding ="same"),
     BatchNormalization(),
     MaxPooling2D(2,2),
     Dropout(0.3),
     Conv2D(128,(3,3),activation = "relu",padding ="same"),
     BatchNormalization(),
     MaxPooling2D(2,2),
     Dropout(0.2),
     Conv2D(64,(3,3),activation = "relu",padding ="same"),
     BatchNormalization(),
     MaxPooling2D(2,2),
     Dropout(0.1),
      Conv2D(32,(3,3),activation = "relu",padding ="same"),
     BatchNormalization(),
     MaxPooling2D(2,2),
     Dropout(0.08),
     Flatten(),
     Dense(512,activation ="relu"),
     BatchNormalization(),
     Dropout(0.4),
     Dense(256,activation ="relu"),
     BatchNormalization(),
     Dropout(0.3),
     Dense(128,activation ="relu"),
     BatchNormalization(),
     Dropout(0.2),
     Dense(64,activation ="relu"),
     BatchNormalization(),
     Dropout(0.1),
     Dense(35,activation ="softmax"),
    
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10,
    batch_size = 35
)

model.save("model.h5")       

test_acc,test_loss = model.evaluate(test_ds)
print("Accuracy {test_acc:.4f}")