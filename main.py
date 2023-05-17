import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

app = FastAPI()

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

# Load the saved model
# imported = tf.saved_model.load('./models/GUI_model')
fixed_length = 48000
MODEL_NAME_DEPLOY= './models/vgg models/vgg_spectro_model.h5' # This is the model which us used in the App
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
        predictions = model(spectrogram)
        label_names = ['Piano','Voice', 'Trumpet', 'Saxophone', 'Organ' ,'Clarinent' ,'Acoutic Guitar' ,'Violin', 'Flute', 'Electric Guitar', 'Cello']
        print(label_names)
        x_labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        print(x_labels)
        # Get the top 2 predictions with labels as a dictionary
        prediction_num = predictions[0].numpy()
        print(prediction_num)
        top_indices = np.argsort(prediction_num)[-2:]
        # top_indices = np.where(top_indices > 0.3)
        print(top_indices)
        all_results = {label_names[i]: prediction_num[i] for i in range(len(prediction_num))}
        print(all_results)
        top_results = {label_names[i]: prediction_num[i] for i in top_indices}
        print(top_results)

        return str({"Predictions": top_results})
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))