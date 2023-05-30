# Image Captioning

Inspired by the paper "Show Attend and Tell," this project aims to train a neural network capable of providing
descriptive text for a given image.

## Overview

The architecture of this project consists of an encoder-decoder neural network. The encoder is responsible for
extracting features from the image, while the decoder interprets these features to generate a sentence.

The encoder utilizes a pre-trained Inception-V3 model, with the last fully connected layers removed, followed by a
custom fully connected layer. This enables the encoder to extract meaningful features from the image.

The decoder comprises a Long Short-Term Memory (LSTM) network along with visual attention. Visual attention allows the
LSTM network to focus on relevant image features while predicting each word in the sentence.

For word embedding, a custom word2vec model was trained and intersected with Google's pre-trained word2vec model. This
embedding helps in representing words as continuous vectors, capturing semantic relationships.

The training and testing data sets were created using the Flickr30 dataset, which contains a diverse range of images and
corresponding captions.

## Technologies Used

- InceptionV3: Pre-trained model used for image feature extraction.
- LSTM: Long Short-Term Memory network used for language modeling and caption generation.
- Visual Attention: Mechanism used to focus on relevant image features during caption generation.
- Word2Vec: Technique used for word embedding, representing words as continuous vectors.

## Requirements

- Python 3.8
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib

## Order of Execution

To train the image captioning model successfully, it is recommended to follow these steps in the specified order:

1. Image Processing
2. Text Processing
3. Training
