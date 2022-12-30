"""
Created on Tue Oct 25 21:33:23 2022

Building a ChatBot with Deep NLP

@author: Capataz Che
"""

# Python 3.5
# Tensorflow 1.0.0 (it was a pain to setup as of Dec 2022 and I could not use the anaconda navigator nor spyder)


# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

# Part 1: Data Preprocessing

# Importing the dataset
lines = (open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n"))
conversations = (open("movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n"))

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines[:-1]:  # last line is empty
    _line = line.split(" +++$+++ ")
    id2line[_line[0]] = _line[4]

# Creating a list of the conversations
conversations_ids = []
for conversation in conversations[:-1]:  # last line is empty
    _conversation = (conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "").split(","))
    conversations_ids.append(_conversation)

# Getting separately the questions and the answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    # decided not to add: let's

    text = re.sub(r" youd ", " you would ", text)
    text = re.sub(r" youre ", " you are ", text)

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"weren't", "were not", text)

    # it's he's she's that's what's // all could be is or has
    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)

    # 'bout goin' 're makin' stayin'

    # removal // TODO: perhaps remove single quote
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?!,*]", "", text)

    return text


clean_questions = [clean_text(x) for x in questions]
clean_answers = [clean_text(x) for x in answers]

# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in clean_questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i += 1
clean_questions = []
clean_answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        clean_answers.append(answer)
        clean_questions.append(short_questions[i])
    i += 1

# Creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        try:
            word2count[word] += 1
        except:
            word2count[word] = 1
for answer in clean_answers:
    for word in answer.split():
        try:
            word2count[word] += 1
        except:
            word2count[word] = 1

# Tokenization and filtering non frequent words
# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold = 15
words2int = (
    {}
)  # Do not know why but the course made two identical dictionaries questionswords2int and answerswords2int
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        words2int[word] = word_number
        word_number += 1

# Adding the last tokens to the dictionary
# PAD is to pad up since sequences and batches need to have the same length
# OUT is to represent the words that got filtered out by our threshold
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    words2int[token] = word_number
    word_number += 1

# Creating the inverse dictionary of the words2int dictionary
int2words = {value_int: key_word for key_word, value_int in words2int.items()}

# Adding the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

# Translating all the questions and the answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        try:
            ints.append(words2int[word])
        except KeyError:
            ints.append(words2int["<OUT>"])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        try:
            ints.append(words2int[word])
        except KeyError:
            ints.append(words2int["<OUT>"])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions (to speed up training and mitigate loss)
sorted_clean_questions = []
sorted_clean_answers = []

# only sentences up to 25 words
for length in range(1, 26):
    for i in enumerate(questions_into_int):
        if length == len(i[1]):
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])

# Part 2: Building The Seq2Seq Model

# Creating placeholders for the inputs and targets (need when creating a NN with tf)
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="target")
    lr = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, lr, keep_prob


# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    # This function adds the int encoding of "<SOS" and removes the last encoding of each target in a batch_size
    left_side = tf.fill([batch_size, 1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])  # basically removing the last token (if input is answers, then removing <EOS>) ???
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Creating the Encoder RNN Layer (LSTM, could be a GRU with PyTorch)
def encoder_rnn(
    rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length
):
    """
    Parameters
    ----------
    rnn_size:
        on input tensors of the encoder RNN layer, not to be confused with # of layers
    -------
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(
        rnn_size
    )  # https://stackoverflow.com/questions/42825206/attributeerror-module-tensorflow-contrib-rnn-has-no-attribute-basiclstmcell
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=keep_prob
    )
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, sequence_length=sequence_length, inputs=rnn_inputs, dtype=tf.float32,)
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, 
                                                                                                                                    attention_option="bahdanau", 
                                                                                                                                    num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name="attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                              training_decoder_function, 
                                                                                                              decoder_embedded_input, 
                                                                                                              sequence_length, 
                                                                                                              scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state, 
                                                                                                                                    attention_option="bahdanau", 
                                                                                                                                    num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0], 
                                                                              attention_keys, 
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_function, 
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              maximum_length, 
                                                                              num_words,
                                                                              name="attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                test_decoder_function, 
                                                                                                                scope=decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1) 
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, num_words, None, scope = decoding_scope, weights_initializer = weights, biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, 
                                           decoder_cell, 
                                           decoder_embeddings_matrix, 
                                           word2int["<SOS>"], 
                                           word2int["<EOS>"], 
                                           sequence_length - 1, 
                                           num_words, 
                                           decoding_scope, 
                                           output_function, 
                                           keep_prob, 
                                           batch_size)
    return training_predictions, test_predictions

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, words2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, 
                                                              answers_num_words + 1, 
                                                              encoder_embedding_size, 
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, words2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, 
                                                         decoder_embeddings_matrix, 
                                                         encoder_state, 
                                                         questions_num_words, 
                                                         sequence_length, 
                                                         rnn_size, 
                                                         num_layers, 
                                                         words2int, 
                                                         keep_prob, 
                                                         batch_size)
    return training_predictions, test_predictions

# Part 3: Training The Seq2Seq Model
# Setting the Hyperparameters
epochs =  100
batch_size = 32 # if trainig takes too long, try 128
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001 # if it takes too long, can make this larger although it risks learning too fast (and wrong)
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5 # suggested by Geoffrey Hinton (general rule of thumb)

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), 
                                                       targets, 
                                                       keep_prob, 
                                                       batch_size, 
                                                       sequence_length, 
                                                       len(words2int),
                                                       len(words2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       words2int)

# Setting up the Loss Error, the Optimizer, and the Gradient Clipping (bounds gradient)
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, 
                                                  targets, 
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
# Question: ["Who", "are", "you"] -> ["Who", "are", "you", <PAD>, <PAD>, <PAD>, <PAD>]
# Answer: [<SOS>, "I", "am", "a", "bot", ".", <EOS>] -> [<SOS>, "I", "am", "a", "bot", ".", <EOS>, <PAD>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)//batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, words2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, words2int))
        yield padded_answers_in_batch, padded_answers_in_batch

# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15) # Hmmm, I would like this to be more randomized
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions))  // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100 # could be 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], 
                                                    {inputs: padded_questions_in_batch, 
                                                     targets: padded_answers_in_batch, 
                                                     lr: learning_rate, 
                                                     sequence_length: padded_answers_in_batch.shape[1], 
                                                     keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print("Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}/{:>6.3f}={:>6.3f}, Training Time on 100 Batches: {:d} seconds".format(epoch, 
                                                                                                                                       epochs, 
                                                                                                                                       batch_index, 
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error,
                                                                                                                                       batch_index_check_training_loss,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch, 
                                                                       targets: padded_answers_in_batch, 
                                                                       lr: learning_rate, 
                                                                       sequence_length: padded_answers_in_batch.shape[1], 
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds".format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            # early stopping
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now!")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry, I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")

# Part 4: Testing the Seq2Seq Model

# Loading the weights and runnning the session
# checkpoint = "./chatbot_weights.ckpt"
# session = tf.InteractiveSession()
# session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(session, checkpoint)

# # Converting the questions from strings to lists of encoding integers
# def convert_string2int(question, word2int):
#     question = clean_text(question)
#     return [word2int.get(word, word2int["<OUT>"]) for word in question.split()]

# # Setting up the chat
# while(True):
#     question = input("You: ")
#     if question == "Goodbye":
#         break
#     question = convert_string2int(question, words2int)
#     question = question + [words2int["<PAD>"]] * (20 - len(question))
#     fake_batch = np.zeros((batch_size, 20))
#     fake_batch[0] = question
#     predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0] # first of the fake batch
#     answer = ""
#     for i in np.argmax(predicted_answer, 1):
#         if int2words[i] == "i":
#             token = "I"
#         elif int2words[i] == "<EOS>":
#             token = "."
#         elif int2words[i] == "<OUT>":
#             token = "OUT"
#         else:
#             token = " " + int2words[i]
#         answer += token
#         if token == ".":
#             break
#     print("ChatBot: " + answer)