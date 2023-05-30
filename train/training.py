import tensorflow as tf
import matplotlib.pyplot as plt
import os


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights


class ImageCaptioningTrainer:
    def __init__(self, X_train, y_train, image_path, tokenizer, embedding_dim, units, vocab_size, max_length,
                 batch_size, buffer_size, epochs, checkpoint_dir, loss_plot_path):
        self.X_train = X_train
        self.y_train = y_train
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.loss_plot_path = loss_plot_path

        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, units, vocab_size)

    def train_step(self, img_tensor, target):
        loss = 0

        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                predictions, hidden, context_vector, attention_weights = self.decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def train(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                         decoder=self.decoder,
                                         optimizer=self.optimizer)

        loss_plot = []

        dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        for epoch in range(1, self.epochs + 1):
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    print(
                        'Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy() / int(target.shape[1])))

            loss_plot.append(total_loss / len(self.X_train))

            if epoch % 5 == 0:
                checkpoint.save(file_prefix=os.path.join(self.checkpoint_dir, 'ckpt'))

        self._plot_loss(loss_plot)

    def _plot_loss(self, loss_plot):
        plt.plot(loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(self.loss_plot_path)

