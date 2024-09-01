import os

from fastapi import FastAPI, Request, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from loguru import logger

from src.utils.predict import predict

app = FastAPI(title="Image to text API")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log the request information
    logger.info(f"Received request: {request.method} {request.url.path}")

    # Call the next middleware or route handler
    response = await call_next(request)

    return response


@app.get('/healthcheck', status_code=200)
async def healthcheck() -> bool:
    return True


@app.post("/predict")
async def image_to_text(image: UploadFile, language: str = Form(..., regex="^(english|ukrainian)$")) -> dict:
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The uploaded file is not an image!")

    # Read the contents of the file asynchronously
    file_bytes = await image.read()
    text, heatmap_base64 = predict(file_bytes, language)
    logger.info(f"For image name: {image.filename} and language: {language} generated caption: \n{text}")
    return {"text": text, "heatmap_base64": heatmap_base64}


is_production = os.getenv("ENVIRONMENT") == "production"

if is_production:
    # Mount the static files directory
    app.mount("/", StaticFiles(directory=os.environ["STATIC_DIR"], html=True), name="root")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
