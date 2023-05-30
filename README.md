# Image Captioning

This project is an Image Captioning service that generates textual captions for images. It uses a deep learning model to
analyze the content of an image and generate a descriptive caption.

## Live web app

To access the React application, go to [image-captioner.herokuapp.com](https://image-captioner.herokuapp.com/).

To access the Swagger API, go to [image-captioner.herokuapp.com/docs](https://image-captioner.herokuapp.com/docs).

## Features

- Upload an image and get a textual caption describing the image content.
- Utilizes a deep learning models with attention mechanism which are trained on Flickr30k dataset for caption
  generation.
- Provides a heatmap visualization that highlights the areas of the image that the model focuses on while generating the
  caption.
- Supports multiple languages (English and Ukrainian) for caption generation.
- Provides a health check endpoint to verify the API status.

## Technologies Used

- Python
- Tensorflow
- FastAPI
- React
- Docker

## Usage

1. Clone the repository:

```shell
   git clone <repository-url>
   ```

2. Build the Docker image:

```shell
docker build -t image-captioning:latest .
```

3. Run the Docker container:

```shell
docker run -e PORT=8000 -p 8000:8000 image-captioning:latest 
```

Or, you can run the FastAPI backend and React frontend separately from their respective folders.

## Create Heroku project

```bash
heroku login
heroku create your-app-name
heroku git:remote your-app-name
heroku stack:set container
git push heroku main
```