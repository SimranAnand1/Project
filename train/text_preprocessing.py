import os
import re
import numpy as np
import pandas as pd
import pickle
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from nltk import RegexpTokenizer
import random
import copy


class TextPreprocessor:
    def __init__(self, workdir, text_dataset_name):
        self.workdir = workdir
        self.text_path = os.path.join(workdir, text_dataset_name)
        self.df = pd.read_csv(self.text_path)

    def clean_text(self, text_path):
        df = pd.read_csv(text_path)
        df['comment'] = df['comment'].apply(self.clean_text_entry)
        groups = [b for a, b in df.groupby('image_name')]
        random.shuffle(groups)
        df = pd.concat(groups).reset_index(drop=True)
        return df

    def clean_text_entry(self, text):
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(',', '')
        text = text.replace('.', '')
        text = text.strip()
        return text

    def tokenize_comments(self, comments):
        tokenizer = RegexpTokenizer(r'\w+|<start>|<end>')
        return comments.apply(tokenizer.tokenize)

    def pad_tokens(self, tokens):
        max_len = max(tokens.apply(len))
        return tokens.apply(lambda t: self.pad(t, max_len))

    def pad(self, tokens, max_len):
        len_ = len(tokens)
        len_ = max_len - len_
        lst = copy.deepcopy(tokens)
        for i in np.arange(0, len_):
            lst.append('_')
        return lst

    def train_custom_word2vec(self, tokens):
        model = Word2Vec(
            sentences=tokens,
            min_count=1,
            window=2,
            size=300,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=2,
            iter=30
        )
        return model

    def load_google_word2vec(self):
        word2vec_path = os.path.join(self.workdir, 'GoogleNews-vectors-negative300.bin.gz')
        return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def update_custom_vocab(self, custom_model, google_model):
        custom_model.intersect_word2vec_format(
            os.path.join(self.workdir, 'GoogleNews-vectors-negative300.bin.gz'),
            lockf=1.0,
            binary=True
        )
        custom_model.build_vocab([list(google_model.vocab.keys())], update=True)
        return custom_model

    def retrain_custom_word2vec(self, model, tokens):
        model.train(tokens, total_examples=len(tokens), epochs=model.iter)
        return model

    def save_custom_word2vec(self, model, df):
        model.wv.save_word2vec_format(os.path.join(self.workdir, 'w2v_imageCap.bin'), binary=True)
        model.save(os.path.join(self.workdir, 'w2v_imageCap.kv'))

        word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(self.workdir, 'w2v_imageCap.bin'),
            binary=True,
            unicode_errors='ignore'
        )

        vec = Tokenizer()
        vec.fit_on_texts(df['tokens'])
        vocab_size = len(vec.word_index)

        embedding_matrix = np.zeros((vocab_size, 300))
        for index, word in vec.index_word.items():
            try:
                embedding_matrix[index] = word2vec[word]
            except:
                embedding_matrix[index] = np.zeros((1, 300))

        embedding_matrix = np.vstack((np.zeros((1, 300)), embedding_matrix))
        seq = vec.texts_to_sequences(df['tokens_pad'])
        seq_vec = np.array(seq).astype('int32')

        df.to_csv(os.path.join(self.workdir, 'captions_pros.csv'), index=None, header=True)
        np.save(os.path.join(self.workdir, 'embedB.npy'), embedding_matrix)
        np.save(os.path.join(self.workdir, 'caption_vec.npy'), seq_vec)

        with open(os.path.join(self.workdir, 'word_ind_map.pkl'), 'wb') as f:
            pickle.dump(vec.word_index, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.workdir, 'ind_word_map.pkl'), 'wb') as f:
            pickle.dump(vec.index_word, f, pickle.HIGHEST_PROTOCOL)

    def train_word2vec_model(self):
        # Cleaning the text
        df = self.clean_text(self.text_path)

        # Tokenizing the comments
        df['tokens'] = self.tokenize_comments(df['comment'])

        # Padding the tokens
        df['tokens_pad'] = self.pad_tokens(df['tokens'])

        # Training the custom Word2Vec model
        custom_w2v_model = self.train_custom_word2vec(df['tokens'])

        # Loading Google's pre-trained Word2Vec model
        google_w2v_model = self.load_google_word2vec()

        # Updating the local model's vocabulary with Google's Word2Vec
        custom_w2v_model = self.update_custom_vocab(custom_w2v_model, google_w2v_model)

        # Training the custom Word2Vec model again
        custom_w2v_model = self.retrain_custom_word2vec(custom_w2v_model, df['tokens'])

        # Saving the custom Word2Vec model and its vocabulary
        self.save_custom_word2vec(custom_w2v_model, df)


if __name__ == "__main__":
    # Choose the dataset. Download ukrainian captions, and place them in workdir
    ukrainian_captions_dataset_url = 'https://www.kaggle.com/datasets/vladislavalerievich/flickr30kukranian'
    english_captions_dataset = 'flickr30k_images/output.csv'

    # Instantiate the TextPreprocessor class
    preprocessor = TextPreprocessor(workdir='/path/to/workdir', text_dataset_name=english_captions_dataset)

    # Train Word2Vec model and perform text preprocessing
    preprocessor.train_word2vec_model()


