import requests
import json
import tensorflow_io as tfio
import base64
import io
import os
from tensorflow import keras
import tensorflow as tf
from fastapi import FastAPI, Request, Response, Form
from requests.auth import HTTPBasicAuth
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator
from typing import Optional

app = FastAPI()

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

MODEL_NAME_DEPLOY= 'Deploy' # This is the model which us used in the App

model = keras.models.load_model('./models/' + MODEL_NAME_DEPLOY)

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

@app.get("/")
async def root():
    return RedirectResponse(url="/index")


@app.get("/index", response_class=HTMLResponse)
async def write_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/html_render/{id}", response_class=HTMLResponse)
async def write_home(request: Request, id: int):
    return templates.TemplateResponse("home.html", {"request": request, "id": id })

class Input(BaseModel):
    inputString: str

@app.post("/make_post")
async def make_request(audio_data: Input):
	audio_data_string = audio_data.inputString
	print(audio_data_string)
        
	# Decode the base64-encoded audio data
	audio_bytes = base64.b64decode(audio_data_string)

	# Load the audio data into a TensorFlow tensor
	audio_tensor = tfio.audio.decode_wav(audio_bytes)

	# Convert the TensorFlow tensor to a NumPy array
	audio_array = audio_tensor.numpy()

	# Create a file-like object to hold the decoded audio data
	audio_file = io.BytesIO()

	# Write the decoded audio data to the file-like object in .wav format
	tf.io.write_file(audio_file, audio_array, name='audio.wav')

	# Reset the file-like object to the beginning
	audio_file.seek(0)

    # Save the .wav file to a folder
	folder_path = os.path.join(os.getcwd(), 'data')
	if not os.path.exists(folder_path):
			os.mkdir(folder_path)
    
	file_path = os.path.join(folder_path, 'audio.wav')
	with open(file_path, 'wb') as f:
			f.write(audio_file.getbuffer())

	x, sample_rate = tf.audio.decode_wav(audio_file, desired_channels=1, desired_samples=16000,)
	x = tf.squeeze(tf.reduce_mean(x, axis=-1, keepdims=True), axis=-1)
	waveform = x
	x = get_spectrogram(x)
	x = x[tf.newaxis,...]

	prediction = model(x)
	print(prediction)
	
	return {"message": "Received input string: " + "Revieved"} #str(prediction)
