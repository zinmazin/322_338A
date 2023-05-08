from keras.models import load_model
import numpy as np
import cv2

# โหลดโมเดล
model = load_model('model.h5')

# กำหนด path ของไฟล์ภาพที่ต้องการทำนาย
# image_path = r'E:\Amodel\pictrain\train\12\12 (57).jpg'
image_path = r'E:\Amodel\1.jpg'
# โหลดภาพและปรับขนาด
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))

# แปลงข้อมูลภาพเป็น array
img_array = np.array(img)

# ปรับ shape ของ array เพื่อให้เหมาะกับ input ของโมเดล
img_array = np.expand_dims(img_array, axis=0)

# ทำนายภาพ
prediction = model.predict(img_array)

# แสดงผลลัพธ์
print(prediction)

labels = ['十一', '十二', '十三', '十四']

# หา index ของค่าที่มากที่สุดในแต่ละ row ของผลลัพธ์
predicted_index = np.argmax(prediction, axis=1)

# นำ index ไป map กับรายการชื่อ labels เพื่อแปลงเป็นชื่อ labels ที่ตรงกับค่าที่ได้จากโมเดล
predicted_labels = [labels[i] for i in predicted_index]

# แสดงผลลัพธ์
print(predicted_labels)