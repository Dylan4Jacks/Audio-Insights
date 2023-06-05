import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import requests
import json
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

app = FastAPI()

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

# Load the saved model
# imported = tf.saved_model.load('./models/GUI_model')
fixed_length = 48000
MODEL_NAME_DEPLOY= './models/production/resnet_model_74v2.h5' # This is the model which us used in the App
model = keras.models.load_model(MODEL_NAME_DEPLOY)

#Spectrogram Content
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels
def squeeze_duel_channel(audio, labels):
  audio = tf.squeeze(tf.reduce_mean(audio, axis=-1, keepdims=True), axis=-1)
  return audio, labels

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

@app.get("/")
async def root():
    return RedirectResponse(url="/index")


@app.get("/index", response_class=HTMLResponse)
async def write_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

@app.post("/uploadfile/")
async def create_upload_file(audio_file: UploadFile = Form(...)):
    try:
        audio_bytes = audio_file.file.read()
        x, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1, desired_samples=fixed_length,)
        x = tf.squeeze(tf.reduce_mean(x, axis=-1, keepdims=True), axis=-1)
        waveform = x
        x = get_spectrogram(x)
        x = x[tf.newaxis,...]

        x_labels = ['pia','voi' ,'tru' ,'sax', 'org', 'cla', 'gac', 'vio', 'flu', 'gel','cel']
        x_labels_full = ['Piano','Voice', 'Trumpet', 'Saxophone', 'Organ' ,'Clarinent' ,'Acoutic Guitar' ,'Violin', 'Flute', 'Electric Guitar', 'Cello']
        prediction = model(x)
        predicted_class_id = tf.argmax(prediction[0]).numpy()
        predicted_probability = 100 * prediction[0][predicted_class_id].numpy()
        print("Predicted class ID:", x_labels[predicted_class_id])
        print("Predicted probability:", '{:.2f}%'.format(predicted_probability))


        plt.bar(x_labels, prediction[0])
        plt.title(x_labels[predicted_class_id])
        # plt.show()

        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Encode PNG image to base64 string
        image_base64 = base64.b64encode(buf.getvalue()).decode()
        # top_3_indices = np.argsort(predicted_probability)[-3:][::-1]
        # top_3_predictions = predicted_probability[top_3_indices]
        all_results = {"Prediction": "Predicted class ID:" + x_labels[predicted_class_id] + "Predicted probability:" + '{:.2f}%'.format(predicted_probability), "Image": image_base64}
        return all_results
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))