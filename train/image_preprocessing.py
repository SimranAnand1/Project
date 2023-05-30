import os
import tarfile
import tensorflow as tf
import pandas as pd
import numpy as np


class ImagePreprocessor:
    def __init__(self, workdir):
        self.workdir = workdir

    def download_dataset(self, dataset_url):
        os.makedirs(self.workdir, exist_ok=True)
        dataset_filename = os.path.join(self.workdir, "flickr30k_images.tar.gz")

        # Download the dataset
        tf.keras.utils.get_file(dataset_filename, dataset_url)

        # Extract the dataset
        with tarfile.open(dataset_filename, "r") as f:
            f.extractall(self.workdir)

    def preprocess_dataset(self):
        image_path = os.path.join(self.workdir, "flickr30k_images/flickr30k_images/")
        feat_path = os.path.join(self.workdir, "image_feat/")
        text_path = os.path.join(self.workdir, "captions.csv")

        df = pd.read_csv(text_path, delimiter="|", skipinitialspace=True)

        image_name_list = list(set(df["image_name"]))
        image_path_list = list(map(lambda arg: image_path + arg, image_name_list))
        feat_path_list = list(map(lambda arg: feat_path + arg, image_name_list))

        return image_path_list, feat_path_list

    def extract_features(self, image_path_list):
        feat_path = os.path.join(self.workdir, "image_feat")
        image_path = os.path.join(self.workdir, "flickr30k_images/flickr30k_images/")

        def feat_extract():
            IV3 = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
            x_in = IV3.input
            x_out = IV3.layers[-1].output
            return tf.keras.Model(inputs=x_in, outputs=x_out)

        mod_fe = feat_extract()
        mod_fe.save(os.path.join(self.workdir, "IV3_feat.h5"))

        def load_image(arg):
            img = tf.io.read_file(arg)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (299, 299))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            return img, arg

        image_dataset = tf.data.Dataset.from_tensor_slices(image_path_list)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_dataset = image_dataset.batch(32)

        for img, path in image_dataset:
            batch_features = mod_fe(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], 8 * 8, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_ = p.numpy().decode("utf-8")
                path_ = os.path.join(feat_path, path_[len(image_path):])
                np.save(path_, bf.numpy())

        return len(os.listdir(feat_path))


if __name__ == "__main__":
    workdir = "/path/to/workdir"
    dataset_url = "https://www.kaggle.com/hsankesara/flickr-image-dataset"
    downloader = ImagePreprocessor(workdir)
    downloader.download_dataset(dataset_url)
    image_path_list, feat_path_list = downloader.preprocess_dataset()
    num_files = downloader.extract_features(image_path_list)
    print(f"Number of extracted feature files: {num_files}")
