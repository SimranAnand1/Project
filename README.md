# Image Captioner

This project is an Image Captioning service that generates textual captions for images. It uses a deep neural networks
to analyze the content of an image and generate a descriptive caption. Models are trained on Flickr30k dataset. Inspired
by the paper "Show Attend and Tell".

## Live web app

To access the React application, go to [image-captioner.herokuapp.com](https://image-captioner.herokuapp.com/).

To access the Swagger API, go to [image-captioner.herokuapp.com/docs](https://image-captioner.herokuapp.com/docs).

## Model

The architecture of the neural network consists of an encoder-decoder neural network. The encoder is responsible for
extracting features from the image, while the decoder interprets these features to generate a sentence.

The encoder utilizes a pre-trained Inception-V3 model, with the last fully connected layers removed, followed by a
custom fully connected layer. This enables the encoder to extract meaningful features from the image.

# Image Captioner

This project is an Image Captioning service that generates textual captions for images. It uses deep neural networks to
analyze the content of an image and generate a descriptive caption. Models are trained on the Flickr30k dataset,
inspired by the paper "Show Attend and Tell".

## Live web app

To access the React application, go to [image-captioner.herokuapp.com](https://image-captioner.herokuapp.com/).

To access the Swagger API, go to [image-captioner.herokuapp.com/docs](https://image-captioner.herokuapp.com/docs).

## Model

The architecture of the neural network consists of an encoder-decoder neural network. The encoder is responsible for
extracting features from the image, while the decoder interprets these features to generate a sentence.

The encoder utilizes a pre-trained Inception-V3 model, with the last fully connected layers removed, followed by a
custom fully connected layer. This enables the encoder to extract meaningful features from the image.

The decoder comprises a Long Short-Term Memory (LSTM) network along with visual attention. Visual attention allows the
LSTM network to focus on relevant image features while predicting each word in the sentence.

For word embedding, a custom word2vec model was trained and intersected with Google's pre-trained word2vec model. This
embedding helps in representing words as continuous vectors, capturing semantic relationships.

The training and testing datasets were created using the Flickr30k dataset, which contains a diverse range of images and
corresponding captions.

## Features

- Upload an image and get a textual caption describing the image content.
- Utilizes deep learning models with an attention mechanism for caption generation.
- Provides a heatmap visualization that highlights the areas of the image that the model focuses on while generating the
  caption.
- Supports multiple languages (English and Ukrainian) for caption generation.
- Provides a health check endpoint to verify the API status.

## Technologies used

- InceptionV3: Pre-trained model used for image feature extraction.
- LSTM: Long Short-Term Memory network used for language modeling and caption generation.
- Visual Attention: Mechanism used to focus on relevant image features during caption generation.
- Word2Vec: Technique used for word embedding, representing words as continuous vectors.

## Running locally in Docker

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

## Running locally in Terminal

Alternatively, you can run the FastAPI backend and React frontend separately from their respective folders. To run the
FastAPI backend and React frontend separately, you can follow these steps:

**Backend Setup:**

1. After cloning the repository open a terminal and navigate to the directory `/backend` where FastAPI backend is
   located.
2. Activate the virtual environment (if using one) to isolate your Python environment.
3. Install the required Python packages by running the following command:

```bash
pip install -r requirements.txt
```

4. Once the dependencies are installed, start the FastAPI backend server by running the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend server should now be running and listening on port 8000.

**Frontend Setup:**

1. Open another terminal window or tab and navigate to the directory where your React frontend is located.
2. Make sure you have Node.js and npm (Node Package Manager) installed on your system.
3. Install the necessary dependencies for the React frontend by running the following command:

```bash
 npm install
```

4. After the dependencies are installed, start the React development server by running the following command:

```bash
 npm start
```

The frontend development server will compile the React code and open a browser window at `http://localhost:3000` (by
default).
The React app will automatically reload if you make any changes to the source code.

**Accessing the Application:**

- With both the backend and frontend servers running, you can access the application in your browser.
- Open a web browser and go to `http://localhost:3000` to access the React frontend.
- The React app will communicate with the FastAPI backend running at `http://localhost:8000` for data exchange and API
  calls.

## Production deployment on Heroku

```bash
heroku login
heroku create your-app-name
heroku git:remote your-app-name
heroku stack:set container
git push heroku master
```