import re
import pickle
import io
import os.path
import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Encoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.den = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        x = self.den(x)
        x = tf.nn.relu(x)
        return x


class Attend(tf.keras.Model):
    def __init__(self, units):
        super(Attend, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, feat, hidden):
        hidden_ = tf.expand_dims(hidden, axis=1)
        score = tf.nn.tanh(self.W1(feat) + self.W2(hidden_))
        att_wt = tf.nn.softmax(self.V(score), axis=1)
        context = att_wt * feat
        context = tf.reduce_sum(context, axis=1)

        return context, att_wt


class Decoder(tf.keras.Model):
    def __init__(self, units, embed_M, sentence_length):
        super(Decoder, self).__init__()
        self.units = units
        self.embed = tf.keras.layers.Embedding(input_dim=embed_M.shape[0], output_dim=embed_M.shape[1],
                                               weights=[embed_M], input_length=sentence_length, trainable=False)
        self.lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
        self.den1 = tf.keras.layers.Dense(units)
        self.den2 = tf.keras.layers.Dense(embed_M.shape[0])
        self.attend = Attend(units)

    def call(self, tok, feat, hidden):
        context, att_wt = self.attend(feat, hidden)
        x = self.embed(tok)
        context_ = tf.expand_dims(context, 1)
        x = tf.concat([context_, x], axis=2)
        output, state, _ = self.lstm(x)
        x = self.den1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.den2(x)
        return x, state, att_wt

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class ImageCaptioning:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.embed_dim = 300
        self.units = 512
        self.max_length = 80
        self.min_words = 3  # Minimum number of words in the result
        self.max_words = 10  # Maximum number of words in the result
        self.word_ind_map = {}
        self.ind_word_map = {}
        self.IV3_feat = None
        self.enc = None
        self.dec = None

    def load_models(self):
        M = np.load(os.path.join(self.model_dir, 'embedB.npy'))
        cap_seq = np.load(os.path.join(self.model_dir, 'caption_vec.npy'))

        with open(os.path.join(self.model_dir, 'word_ind_map.pkl'), 'rb') as f:
            self.word_ind_map = pickle.load(f)

        with open(os.path.join(self.model_dir, 'ind_word_map.pkl'), 'rb') as f:
            self.ind_word_map = pickle.load(f)

        self.IV3_feat = tf.keras.models.load_model(os.path.join(self.model_dir, 'IV3_feat.h5'))

        self.enc = Encoder(self.embed_dim)
        self.dec = Decoder(self.units, M, self.max_length)
        self.enc.load_weights(os.path.join(self.model_dir, 'encoder/'))
        self.dec.load_weights(os.path.join(self.model_dir, 'decoder/'))
    def evaluate(self, file):
        while True:
            attention_plot = np.zeros((self.max_length, 64))
            hidden = self.dec.reset_state(batch_size=1)
            temp_input = tf.expand_dims(self.load_image(file)[0], 0)
            img_tensor_val = self.IV3_feat(temp_input)
            img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

            features = self.enc(img_tensor_val)

            dec_input = tf.expand_dims([self.word_ind_map['<start>']], 0)
            result = []

            for i in range(self.max_length):
                predictions, hidden, attention_weights = self.dec(dec_input, features, hidden)

                attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

                predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
                result.append(self.ind_word_map[predicted_id])

                if self.ind_word_map[predicted_id] == '<end>':
                    if self.min_words <= len(result) <= self.max_words:
                        attention_plot = attention_plot[:len(result), :]
                        return result, attention_plot
                    else:
                        break

                dec_input = tf.expand_dims([predicted_id], 0)

    def measure_caption_accuracy(self, result):
        # Measure the accuracy of the caption based on your criteria
        # You can implement your own scoring mechanism here
        # Return a score between 0 and 1, where 1 indicates a perfect caption

        # Example scoring mechanism: Calculate the ratio of known words in the caption
        known_words = sum(word in self.word_ind_map for word in result)
        accuracy = known_words / len(result) if len(result) > 0 else 0.0
        return accuracy

    def predict(self, file):
        result, attention_plot = self.evaluate(file)
        accuracy = self.measure_caption_accuracy(result)

        if accuracy < 0.8:
            result, attention_plot = self.evaluate(file)  # Rerun evaluation for low accuracy captions

        heatmap_base64 = self.plot_attention(file, result, attention_plot)

        text = " ".join(result)
        text = re.sub(r"<end>", "", text).strip()
        return text, heatmap_base64

    @staticmethod
    def load_image(arg):
        img = tf.convert_to_tensor(arg)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, arg

    @staticmethod
    def read_image_file(file):
        # Load the image using PIL
        pil_image = Image.open(BytesIO(file))
        # Convert PIL image to numpy array
        return np.array(pil_image)

    @staticmethod
    def plot_attention(file, result, attention_plot):
        image = ImageCaptioning.read_image_file(file)
        len_result = len(result)
        num_rows = (len_result + 1) // 2
        num_cols = min(len_result, 2)

        fig = plt.figure(figsize=(10, 10), dpi=200)

        for l in range(len_result):
            temp_att = np.resize(attention_plot[l], (8, 8))
            ax = fig.add_subplot(num_rows, num_cols, l + 1)
            ax.set_title(result[l])
            img = ax.imshow(image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()

        # Capture the plot image as bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_bytes = buf.getvalue()
        buf.close()

        # Encode the image bytes as a base64 string
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Return the plot image base64 string
        return image_base64
