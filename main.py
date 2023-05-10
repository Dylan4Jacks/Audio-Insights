# import tensorflow as tf
# from tensorflow import keras
# import base64
# import io
import os
import numpy as np
from fastapi import FastAPI, Request, Response, Form
from requests.auth import HTTPBasicAuth
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator
from typing import Optional

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

# Load the saved model
# imported = tf.saved_model.load('./models/GUI_model')

@app.get("/")
async def root():
    return RedirectResponse(url="/index")


@app.get("/index", response_class=HTMLResponse)
async def write_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/html_render/{id}", response_class=HTMLResponse)
async def write_home(request: Request, id: int):
    return templates.TemplateResponse("home.html", {"request": request, "id": id })

class AudioData(BaseModel):
  input_text: str
  audio: str

def base64_to_wav(base64_string):
    # audio_bytes = io.BytesIO(base64.b64decode(base64_string))
    return "TEST" #tf.audio.decode_wav(audio_bytes.getvalue())

@app.post("/make_post")
async def make_request(audiodata: AudioData):
  # audio_string = tf.constant(audiodata.audio, dtype=tf.string)
  # encode_string = tf.io.decode_base64(audio_string)
  # audio_tensor, sample_rate = tf.audio.decode_wav(encode_string)
  # Get the prediction dictionary
  # prediction = imported(audio_tensor)

  # # Get the predicted class name and probability
  # #predicted_class_name = prediction['class_names'][0].numpy().decode('utf-8')
  # predicted_class_id = prediction['class_ids'][0].numpy()
  # predicted_probability = 100 * prediction['predictions'][0][0].numpy()
  # label_names = ['Piano','Voice', 'Trumpet', 'Saxophone', 'Organ' ,'Clarinent' ,'Acoutic Guitar' ,'Violin', 'Flute', 'Electric Guitar', 'Cello']
  # # Print the predicted class name and probability
  # print("Predicted class:", label_names[predicted_class_id])
  # print("Predicted probability:", '{:.2f}%'.format(predicted_probability))

  # print("prediction")
  return "Endpoint currently Yields Error due to Base64 -> Wav file conversion issues"  #str("Predicted probability:", '{:.2f}%'.format(predicted_probability))