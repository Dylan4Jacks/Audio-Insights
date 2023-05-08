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

MODEL_NAME = 'CNN_Audio_Classifyer_Testing_Model'

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

class AudioData(BaseModel):
  input_text: str
  audio: str

@app.post("/make_post")
async def make_request(audiodata: AudioData):
	print(audiodata.input_text)
	print(audiodata.audio)

	imported = tf.saved_model.load('./models/' + MODEL_NAME)

	x = audiodata.audio
	x = tf.io.read_file(str(x))
	x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
	x = tf.squeeze(tf.reduce_mean(x, axis=-1, keepdims=True), axis=-1)
	waveform = x
	x = get_spectrogram(x)
	x = x[tf.newaxis,...]

	prediction = imported(x)
	x_labels = ['cel' 'cla' 'flu' 'gac' 'gel' 'org' 'pia' 'sax' 'tru' 'vio' 'voi']
	plt.bar(x_labels, tf.nn.softmax(prediction[0]))
	plt.title('sax')
	plt.show()

	display.display(display.Audio(waveform, rate=16000))

	response_json_unprocessed = json.loads(response.text)
	print(response_json_unprocessed)
	scores = []
	if "error" in response_json_unprocessed:
		return
	for word in response_json_unprocessed["word_list"]:
		scores.append(word['mean'])
	response_json = {
	'scores' : scores,
	'av_score' : response_json_unprocessed['sentence_mean'],
	}

	return response_json
