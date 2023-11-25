from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np
import cv2 as cv

class_names = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised',
}

haar = True
if haar:
    model = models.load_model("emotion_detection_model_with_face_detection.keras")
else:
    model = models.load_model("emotion_detection_model.keras")

def predict_mood(model,path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((48, 48))
    data = np.asarray(img)
    data = data / 255
    probs = model.predict(np.array([data])[:1])

    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    
    return top_prob, top_pred


face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


img_path="placeholder.png"
content=""
pred=""
prob=0
index="""
<|text-center|

<|{"logo.png"}|image|width=10vw|>

<|{content}|file_selector|extensions=.jpg,.jpeg,.gif,.png|>
Select an image

<|{img_path}|image|width=20vw|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

<|{pred}|>

>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_mood(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "Mood: " + top_pred
        state.img_path = var_val
        img = cv.imread(state.img_path, cv.IMREAD_UNCHANGED)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        colors = {
            'angry': (0, 0, 255),       
            'disgusted': (0, 255, 0),   
            'fearful': (255, 0, 255),   
            'happy': (255, 255, 0),     
            'neutral': (128, 128, 128), 
            'sad': (0, 128, 255),       
            'surprised': (0, 255, 255)  
        }
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv.putText(img, top_pred, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, colors[top_pred], 2)

        state.img_path = cv.imencode(".jpg", img)[1].tobytes()

app = Gui(page=index)

if __name__ == "__main__":

    app.run(use_reloader=True, port=1050)
        
