from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# tf.debugging.set_log_device_placement(True)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

vocabulary_size = 1000
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = vocabulary_size)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# word to index mapping
word2id = reuters.get_word_index()

max_words = 200
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)

embedding_size=300
model=Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_size, input_length=max_words))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
num_epochs = 15
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train, y_train = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test accuracy:', scores[1])