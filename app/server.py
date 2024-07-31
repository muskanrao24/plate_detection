from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi_socketio import SocketManager
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import base64
import cv2
import numpy as np
from frame_processor import process_frame
from fastapi.staticfiles import StaticFiles
import socketio

origins = [
    "http://127.0.0.1:8000",
    # IMPORTANT: Add other endpoints here
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=origins)
socket_app = socketio.ASGIApp(sio)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

printed_image = 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@sio.on("image")
async def handle_image(sid, data_image):
    global printed_image
    # Decode the received image data
    decoded_data = base64.b64decode(data_image)
    np_data = np.frombuffer(decoded_data, np.uint8)
    if printed_image == 0:
        # Save the image to the disk
        with open("image.jpg", "wb") as file:
            file.write(decoded_data)
        printed_image = 1
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    try:
        processed_frame = process_frame(img)
    except Exception as e:
        print(e)
        # Emit the same image back if an error occurs during processing
        await sio.emit("response_back", data_image, to=sid)
        return

    # Encode the processed frame to base64 string
    _, imgencode = cv2.imencode(".jpg", processed_frame)
    stringData = base64.b64encode(imgencode).decode("utf-8")
    b64_src = "data:image/jpg;base64,"
    stringData = b64_src + stringData

    # Emit the processed frame back
    await sio.emit("response_back", stringData, to=sid)


app.mount("/", socket_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
