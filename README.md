# Image Captioning

This project is an Image Captioning service that generates textual captions for images. It uses a deep learning model to
analyze the content of an image and generate a descriptive caption. Inspired by the paper "Show Attend and Tell".

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

The training and testing data sets were created using the Flickr30 dataset, which contains a diverse range of images and
corresponding captions.

## Features

- Upload an image and get a textual caption describing the image content.
- Utilizes a deep learning models with attention mechanism which are trained on Flickr30k dataset for caption
  generation.
- Provides a heatmap visualization that highlights the areas of the image that the model focuses on while generating the
  caption.
- Supports multiple languages (English and Ukrainian) for caption generation.
- Provides a health check endpoint to verify the API status.

## Technologies Used

- InceptionV3: Pre-trained model used for image feature extraction.
- LSTM: Long Short-Term Memory network used for language modeling and caption generation.
- Visual Attention: Mechanism used to focus on relevant image features during caption generation.
- Word2Vec: Technique used for word embedding, representing words as continuous vectors.

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
git push heroku master
```