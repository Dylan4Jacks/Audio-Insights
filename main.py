import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from typing import Annotated
from io import BytesIO
from pydub import AudioSegment

app = FastAPI()

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

# Load the saved model
imported = tf.saved_model.load('./models/GUI_model')
# MODEL_NAME_DEPLOY= 'Deploy' # This is the model which us used in the App
# model = keras.models.load_model('./models/' + MODEL_NAME_DEPLOY)

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
        audio_tensor, sample_rate = tf.audio.decode_wav(audio_bytes, desired_channels=1, desired_samples=16000)
        audio_tensor = tf.squeeze(audio_tensor, axis=-1)
        audio_tensor = tf.ensure_shape(audio_tensor, (16000,))
        audio_tensor = tf.cast(audio_tensor, dtype=tf.float32)
        print(type(audio_tensor))
        spectrogram = tf.signal.stft(
            audio_tensor, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        #plt Waveform and Model
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(audio_tensor.shape[0])
        axes[0].plot(timescale, audio_tensor.numpy())
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])

        plot_spectrogram(spectrogram.numpy(), axes[1])
        axes[1].set_title('Spectrogram')
        plt.suptitle("SAX")
        plt.show()

        prediction = imported(audio_file.file)
        # prediction = model(spectrogram)

        # Get the predicted class name and probability
        predicted_class_id = prediction['class_ids'][0].numpy()
        predicted_probability = 100 * prediction['predictions'][0][0].numpy()

        label_names = ['Piano','Voice', 'Trumpet', 'Saxophone', 'Organ' ,'Clarinent' ,'Acoutic Guitar' ,'Violin', 'Flute', 'Electric Guitar', 'Cello']
        # Print the predicted class name and probability
        print("Predicted class:", label_names[predicted_class_id])
        print("Predicted probability:", '{:.2f}%'.format(predicted_probability))
        
        prediction_list = []
        # Check if there's only one class with high probability
        for idx, val in enumerate(prediction['predictions'][0].numpy()):
            if val > 0.3:  # Set threshold based on your dataset
                prediction_list.append({"Predicted class": label_names[idx], 
                                        "Predicted probability": '{:.2f}%'.format(100 * prediction['predictions'][0][idx].numpy())})
        return {"Predictions": prediction_list}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))