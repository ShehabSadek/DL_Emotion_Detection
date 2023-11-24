from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image

model = models.load_model("baseline_mariya.keras")


def predict_mood(model,path_to_img):
    print(model.summary())
    print(path_to_img)



img_path="placeholder.png"
content=""

index="""
<|text-center|
<|{"logo.png"}|image|width=10vw|>

<|{content}|file_selector|extensions=.png, .jpg, .jpeg|>
Select an image

<|{img_path}|image|width=20vw|>

<|lebel here|indicator|value=0|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        state.img_path = var_val
        predict_mood(model, var_val)
    #print(var_name,var_val)

app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)