from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np
import cv2 as cv
import time

class_names = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised',
}
colors = {
    'angry': (0, 0, 255),       
    'disgusted': (0, 255, 0),   
    'fearful': (255, 0, 255),   
    'happy': (255, 255, 0),     
    'neutral': (128, 128, 128), 
    'sad': (0, 128, 255),       
    'surprised': (0, 255, 255)  
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
captured_image = None
value=""
is_webcam_active=False
index="""
<|text-center|

<|{"logo.png"}|image|width=30vw|>
# Emotion : **<|{pred}|>**{: .color-primary; .font-size-h1}

<|{content}|file_selector|extensions=.jpg,.jpeg,.gif,.png|>
Select an image
<br/>
<|{img_path}|image|width=20vw|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

<|{pred}|>
<br/>
<container|container|part|


<br/>

<card|card p-half|part|
## **Webcam**{: .color-primary} component

<|text-center|part|
<|{value}|toggle|lov=Files;Webcam|>
<|{is_webcam_active}|>
|card>
|container>
>
"""
def predict_mood_from_frame(frame):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (48, 48))
    data = np.asarray(img) / 255
    probs = model.predict(np.array([data]))

    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]

    return top_prob, top_pred

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_mood(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = top_pred
        state.img_path = var_val
        img = cv.imread(state.img_path, cv.IMREAD_UNCHANGED)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,minNeighbors=1)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv.putText(img, top_pred, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, colors[top_pred], 2)

        state.img_path = cv.imencode(".jpg", img)[1].tobytes()
    if var_name == "value":
        if var_val=="Files":
            state.img_path = "placeholder.png"
            state.pred = ""
            state.is_webcam_active = False
        elif var_val == "Webcam":
            state.is_webcam_active = True

            while state.is_webcam_active:
                cap = cv.VideoCapture(0)
                ret, frame = cap.read()
                state.refresh("img_path")

                if frame.size != 0:
                    img_path = cv.imencode(".jpg", frame)[1].tobytes()
                    state.img_path = img_path
                    state.refresh("img_path")
                

                if state.value == "Files":
                    state.is_webcam_active = False
                    break

                if ret:
                    top_prob, top_pred = predict_mood_from_frame(frame)
                    state.prob = round(top_prob * 100)
                    state.pred = "Mood: " + top_pred
                    img = frame.copy()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, minNeighbors=1)
                    for (x, y, w, h) in faces:
                        cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        cv.putText(img, top_pred, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, colors[top_pred], 2)

                    captured_image = img
                    time.sleep(0.1)
                cap.release()
            state.img_path = "placeholder.png"
            state.refresh("img_path")

            state.is_webcam_active = False

app = Gui(page=index)
if __name__ == "__main__":

    app.run(use_reloader=True, port=1050)
        
