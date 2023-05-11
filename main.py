from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from requests.auth import HTTPBasicAuth
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator
from typing import Annotated

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


@app.post("/files/")
async def create_file(audio_file: Annotated[bytes, File()]):
    try:
        return {"file_size": len(audio_file)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    


@app.post("/uploadfile/")
async def create_upload_file(audio_file: UploadFile = Form(...)):
    try:
        return {"filename": audio_file.filename}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))