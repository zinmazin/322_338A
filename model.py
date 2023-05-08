import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import layers
import os
from PIL import Image

# กำหนด path ของโฟลเดอร์ที่เก็บไฟล์ภาพ
path = r'E:\Amodel\New folder\pictrain'

# กำหนดขนาดของรูปภาพ
width, height = 128, 128

# สร้าง list สำหรับเก็บข้อมูลและ label
data = []
labels = []

# วนลูปอ่านไฟล์ภาพในโฟลเดอร์แต่ละโฟลเดอร์
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        # วนลูปอ่านไฟล์ภาพในโฟลเดอร์นั้นๆ
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # เปิดไฟล์ภาพและแปลงเป็นตัวเลข
            with Image.open(file_path) as img:
                img = img.resize((width, height))
                img = np.array(img)
                # เพิ่มข้อมูลและ label ลงใน list
                data.append(img)
                labels.append(int(folder)-10)

# แปลง list เป็น numpy array
data = np.array(data)
labels = np.array(labels)

# ตรวจสอบขนาดของ data และ labels
# print('Data shape:', data.shape)
# print('Labels shape:', labels.shape)



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


from keras.utils import to_categorical

# กำหนด num_classes
num_classes = np.unique(labels)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import numpy as np
from keras.utils import to_categorical


labels = labels -1
y_onehot = to_categorical(labels, 4)


print(len(y_onehot))
print(len(data))
print(len(labels))
# Split data into train and validation set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, y_onehot, test_size=0.2, random_state=42,)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Define hyperparameters
batch_size = 32
epochs = 100
learning_rate = 0.001


# print(labels)
# print(len(labels))

# print(labels)
# print(len(labels))


# Build and compile model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test_scaled, y_test))
model.save("model")

model.save("model.h5")



import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ใช้โมเดลเดิมที่คุณสร้างไว้เพื่อทำนายข้อมูล validation
y_pred = model.predict(X_test_scaled)

# แปลงค่าความน่าจะเป็นในการทำนายของโมเดลเป็น label
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# คำนวณค่า accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# คำนวณ confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(confusion_mtx)

# คำนวณ precision, recall, F1-score สำหรับแต่ละ class
class_report = classification_report(y_true, y_pred)
print("Classification report:")
print(class_report)

#save model


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_true = label_binarize(y_true, classes=[0, 1, 2, 3])  # แปลงเป็น binary format
n_classes = 4  # จำนวนคลาส
y_score = label_binarize(y_pred, classes=[0, 1, 2, 3])

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# แสดงกราฟ ROC curve สำหรับแต่ละคลาส
import matplotlib.pyplot as plt

for i in range(n_classes):
    plt.figure()
    lw = 2
    plt.plot(fpr[i], tpr[i], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class %d' % i)
    plt.legend(loc="lower right")
    plt.show()
