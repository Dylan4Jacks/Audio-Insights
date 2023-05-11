from tensorflow import saved_model, audio
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from typing import Annotated
from io import BytesIO

app = FastAPI()

#html directory
app.mount("/public", StaticFiles(directory="public"), name="public")
templates = Jinja2Templates(directory="public/views")

# Load the saved model
imported = saved_model.load('./models/GUI_model')

@app.get("/")
async def root():
    return RedirectResponse(url="/index")


@app.get("/index", response_class=HTMLResponse)
async def write_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def create_upload_file(audio_file: UploadFile = Form(...)):
    try:
        contents = await audio_file.read()
        bytes_io = BytesIO(contents)
        decoded_audio, sample_rate = audio.decode_wav(bytes_io.getvalue())
        prediction = imported(decoded_audio)

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