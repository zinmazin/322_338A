from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import re
import base64
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
labels = ['十一', '十二', '十三', '十四']
def convert_image(img_data):
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(img_str))
    img = Image.open('output.png').convert('L')
    img = img.resize((28,28))
    x = np.invert(np.array([img.getdata()]))
    x = x.reshape(1, 28, 28, 1)
    return x

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        img_data = request.form['img_data']
        x = convert_image(img_data)
        prediction = np.argmax(model.predict(x))
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict_page():
    img_data = request.form['img_data']
    x = convert_image(img_data)
    prediction = np.argmax(model.predict(x))
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
