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
MODEL_NAME_DEPLOY= './models/production/resnet_model_74.h5' # This is the model which us used in the App
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

@app.post("/uploadfile/")
async def create_upload_file(audio_file: UploadFile = Form(...)):
    try:
        audio_bytes = audio_file.file.read()
        audio_tensor, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1, desired_samples=fixed_length)
        audio_tensor = tf.squeeze(audio_tensor, axis=-1)
        audio_tensor = tf.ensure_shape(audio_tensor, (fixed_length,))
        audio_tensor = tf.cast(audio_tensor, dtype=tf.float32)
        print(type(audio_tensor))
        spectrogram = tf.signal.stft(
            audio_tensor, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]

        # prediction = imported(audio_file.file)
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        spectrogram_list = spectrogram.numpy().tolist() #errrrror 

        # Define headers with the API token
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': 'D6b4lE0eG672DdZtlbobS7pTaXBWF3Oj4obwMwWk'
        }

        # Create the payload for the Lambda function
        payload = json.dumps({
            'input_data': spectrogram_list
        })
        # Send the data to the Lambda function
        # response = requests.post('https://epqxpvb6xc.execute-api.ap-southeast-2.amazonaws.com/test/dlcnn-audio-insights-lambda', headers=headers, data=payload)

        print(type(spectrogram))
        predictions = model(spectrogram)
        softmax_predictions = tf.nn.softmax(predictions[0]).numpy()
        max_index = np.argmax(softmax_predictions)

        label_names = ['Piano','Voice', 'Trumpet', 'Saxophone', 'Organ' ,'Clarinent' ,'Acoutic Guitar' ,'Violin', 'Flute', 'Electric Guitar', 'Cello']
        x_labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        title = x_labels[max_index]

        plt.bar(x_labels, softmax_predictions)
        plt.title(title)
        
        # Convert plot to PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Encode PNG image to base64 string
        image_base64 = base64.b64encode(buf.getvalue()).decode()

        all_results = {"Predictions": softmax_predictions.tolist(), "Image": image_base64}
        return all_results
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))