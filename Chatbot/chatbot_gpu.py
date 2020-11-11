# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:04:13 2020

@author: udaykiran.patnaik
"""


# Building a chatbot with Deep NLP


# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time


######### PART 1 - DATA PREPROCESSING ##########


# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# Creating a list of conversations
conversations_ids = []
for conversation in conversations[:-1]: # ignore the last row as it is empty
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(","))
    
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
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
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold_questions = 15
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
threshold_answers = 15
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1   
        
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1    
    
# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}    

# Adding the End of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)    
    
# Sorting questions and answers by the length of questions
sorted_clean_questions = []    
sorted_clean_answers = []  
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
            
# Below commented code is same code as above but using two variables i,j which is standard way of using enumerate in python  
# where i is i[0]->index of list and 
# j is i[1]->value at that index
#for length in range(1, 25 + 1):
#    for i,j in enumerate(questions_into_int):
#        if len(j) == length:
#            sorted_clean_questions.append(questions_into_int[i])
#            sorted_clean_answers.append(answers_into_int[i])
            
            
########### Part 2 - BUILDING THE SEQ2SEQ MODEL ###########
            
            
# Creating placeholder for the inputs and the targets(output of model, this will be compared with actual output)
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')  ## [None,None]: indicates dimension = 2
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate') 
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')      ## used to control dropout hyper paarmeter
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
# Format1: target must be into batches because batches of targets will go into RNN of decoder
# Format2: each batch of target must start with <SOS> token, so need to add <SOS> token at the beginning of each batch
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1]) # Ignore the last column of target as we inserted <SOS> coulmn at the beginning. This is required to maintain the same length of target inside a batch
    preprocessed_targets = tf.concat([left_side, right_side], 1) # 1: horizontal concatination
    return preprocessed_targets

# Creating the Encoder RNN Layer
# rnn_inputs = model inputs (return value of function model_inputs() created above).
# rnn_size = the number of input tensors of the encoder rnn.
# num_layers = number of layers.
# keep_prob = dropout regularisation to improve accuaracy of the stacked LSTM.
# sequence_length = the list of the length of each question in the batch.    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) # to create LSTM using 'BasicLSTMCell'
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)  #To stack multiple rnn layers with dropout applied
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    """
    encoder_output: is not used in the program, so if required it can also be replaced by underscore('_'). It is upto the programmer.
    """
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)  
    """
    prepare_attention(): to preprocess the training data in order to prepare it to the attention process.
    attention_keys: this is key to be compared with the target states
    attention_values:the values that we'll used to construct the context vectors.
    where the context vector is returned by the encoder and that should be used by the decoder as the first element of the decoding.
    attention_score_function: used to compute the similarity between the keys and the target states
    attention_construct_function:function used to build the tensor state.
    """    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    """
    attention_decoder_fn_train(): is a training function (function to decode the training set) for an attention-based sequence-to-sequence model. 
                                  It should be used when dynamic_rnn_decoder is in the training mode. 
    """
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                              training_decoder_function, 
                                                                                                              decoder_embedded_input, 
                                                                                                              sequence_length, 
                                                                                                              scope = decoding_scope)
    """
    dynamic_rnn_decoder(): is a function used to get the output, final state and the final context state of the decoder.
    decoder_final_state, decoder_final_context_state are not required in the program, so it can also be replaced as underscore('_') in the program. It is upto the programmer.
    decoder_output: used for back propagation into the neural network during training.
    """
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decoding the test/validation set
# decode_test_set(): This function will be called twice. 1. for test (to predict the answer of new questions), as well as 2. during validation of train dataset. 
# This(validation of train dataset) is done using cross validation technique where 10% of train data will be used for validating to reduce overfilling and improve accuracy
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)  
    """
    maximum_length: longest entry you can find in the batch.
    num_words:total number of words of all the answers.
    """    
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
                                                                              name = "attn_dec_inf")
    """
    attention_decoder_fn_inference(): to predict the outcome of new observations that is not used in training, i.e these new observations are called test observations.
    """
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                test_decoder_function, 
                                                                                                                scope = decoding_scope)
    """
    dynamic_rnn_decoder(): is a function used to predict the test observations.
    decoder_final_state, decoder_final_context_state are not required in the program, so it can also be replaced as underscore('_') in the program. It is upto the programmer.
    test_predictions: ultimate test predictions
    """
    return test_predictions


# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)    #To stack multiple rnn layers with dropout applied
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        # make fully connected layer once stacked lstm with droput is finished. This will get features as input from previous part of deep learning model (stacked LSTM).
        output_function = lambda x: tf.contrib.layers.fully_connected(x, # inputs
                                                                      num_words, # number of outputs
                                                                      None, # No normalizer is used, activation = relu(default)
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables() # reuse the variables that were introduced in this decoding scope.
        """
        Now let us decode_test_set() to validate our train data (90% of train data) using validation data(10% of train data)
        Note: This validation data is not fed again to train our model
        To validate we use some cross validation technique
        decode_test_set() is also used to get the final prediction which will be used for our chatbot
        """
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
        
 
# Building the seq2seq model (brain of chatbot) 
"""
The purpose of this function is to return encoder_state using encoder_rnn and to return [training_predictions and test_predictions] using decoder_rnn 
using one compined function (i.e. putting everything together)

inputs: these are questions present in the dataset which will be used to train model by asking questions to model during training
targets: the real answers present in the dataset for above questions which will be used during training of model
answers_num_words: total number of words in all the answers
question_num_words: total number of words in all the questions
encoder_embedding_size: number of dimension of the embedding_matrix of the encoder
decoder_embedding_size: number of dimension of the embedding_matrix of the decoder
num_layers: number of layers in decoder cell containing stacked lstm layers with droput applied
questionswords2int: dictionary to preprocess our targets
"""      
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    # Initialize the decoder_embedding_matrix using tensor flow Variable class which takes random numbers with uniform distribution between 0 to 1 to inialize the decoder_embedding_matrix
    # The size of the decoder_embedding_matrix will be [row = number of questions asked + 1, col = embedding size of the decoder]
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    #training_predictions and test_predictions respectively are the predictions of the observations used for the training and 
    #the predictions of new observations that won't be used for the training.
    
    #These test_predictions will either be used for cross-validation to test the predictive power of the model on new observations
    #that won't be used to train the model more or some new predictions just to chat with the chatbot.    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


################ PART 3 - TRAINING THE SEQ2SEQ MODEL ##############
    
# Setting the Hyperparameters
epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024  # number of columns in encoding embedded matrix
decoding_embedding_size = 1024  # number of columns in decoding embedded matrix
learning_rate = 0.001
learning_rate_decay = 0.9  # rate at which learning rate is reduced over the iterations of the training
min_learning_rate = 0.0001 # minimum learning rate below which it should not decay after many iterations.
keep_probability = 0.5


# Defining a session
"""
To train the Seq2Seq model we are going to define a tensorflow session on which all the of tensorflow training will be run.
So to open a session in tensorlow we're going to create an object of the interactive session class and that object will be our session.
But before we create an object we need to reset the tensorflow graphs to ensure that the graph is ready for the training.
So in general when you open a tensorflow session to do some training you have to reset the graph first.
So we're going to start by doing this resetting the graph and to do this we need to take the tensorflow library and 
then apply the reset default graph function which is a function by tensorflow that resets the tensorflow graph.
"""
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()


# Setting the maximum sequence length = 25
"""
Meaning it sets basically the maximum length of the sequences in the questions and the answers.
This maximum length is 25 which means that in the training we won't be using the questions and the answers that have more than 25 words.
"""
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length') # This name parameter enables us to reuse this sequence length afterwards during training


# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)


# getting the training and test preditions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), # basically reverses the dimensions of a tensor. This is basically used to reshape the input tensor.

                                                        targets,
                                                        keep_prob,
                                                        batch_size,
                                                        sequence_length,
                                                        len(answerswords2int),
                                                        len(questionswords2int),
                                                        encoding_embedding_size,
                                                        decoding_embedding_size,
                                                        rnn_size,
                                                        num_layers,
                                                        questionswords2int)
                                                        

# Setting up the Loss Error, the Optimizer and Gradient Clipping
"""
This is a technique actually some operations that will cap the gradient in the graph between a minimum
value and a maximum value and that's to avoid some exploding or vanishing gradient issues.
And so we're going to make sure to avoid these issues by applying gradient clipping to our optimizer.
And so we are going to define a new scope here which will contain two elements to final ultimate elements
that we'll use for the training which are going to be the loss error and the optimizer with gradient clipping.
The loss error is going to be based on a weighted cross entropy loss error which is the most relevant
loss to use when dealing with sequences as we are doing right now and in general when dealing with deep
NLP and the optimizer that we're going to use will be first an adam optimizer which is one of the 
best optimizers for stochastic gradient descent and then we will apply gradient clipping to that optimizer to 
avoid exploding or vanishing gradient issues.
"""
with tf.name_scope("opimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length])) # tensor of weights all initilized with one with correct dimensions/shape of tensors
    optimizer = tf.train.AdamOptimizer(learning_rate)
    """
    We have one gradient per neuron in the neural networks and for each of these neuron we compute the gradient of the loss error 
    with respect to the weight of that neuron and all these gradients are into a graph and are attached to variables (grad_tensor, grad_variable).
    So now the first thing that we need to do is to compute all these gradients and to compute them that's where our optimizer comes in.
    We're going to use a method from that optimizer which is the compute gradient method and 
    that will compute these gradients of the loss error with respect to the weights of each of the neurons.
    """
    gradients = optimizer.compute_gradients(loss_error)
    # Gradient clip_by_value (minimum clip = -5, maximum clip = 5)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients) # Apply new clipped gradient using optimizer
    
    
# Padding the sequences with the <PAD> token [before padding applied]
# Question: [ 'Who', 'are', 'you' ]
# Answer:   [ <SOS>, 'I', 'am', 'a', 'bot' '.', <EOS> ]
    
# [after padding applied]
# Question: [ 'Who', 'are', 'you',<PAD>, <PAD>, <PAD>, <PAD> ]
# Answer:   [ <SOS>, 'I', 'am', 'a', 'bot' '.', <EOS>, <PAD> ]    

"""
Now we're going to finally apply the padding to the sequences with the <pad> (pad token).
So first We'll answer two questions First why do we have to do that.
And second what are we going to do exactly with our questions and our answers.
So the answer to why we have to do this is that all the sentences in a batch whether they are questions
or answers must have the same length.
And now the second question what are we going to do exactly.
So basically the <pad> are added so that the length of the question sequence is equal to the length
of the answer sequence.That's the purpose of padding and that's a must do in deep NLP.
So now what we're going to have to do is for each batch and for each sentence of each batch we will
complete the length of each sentence with enough <pad> so that each sentence of the batch has the same length.
"""
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into questions and answers
"""
we're going to split the data into batches using a new function that will use the previous function because
of course inside each of these batches we will pad the questions with the pad tokens so we will make
a new function and then of course in the training we will apply that function to create the batches
and feed the neural network with these batches of inputs and targets.
I remind that the inputs will correspond to the questions and the targets will respond to the answers.
"""
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):    # number of batch = (total nunber of questions or answers/ batch size) . '//' is used as divide operator inorder to get an integer value.
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch   # yield instead of return statement. yield in Python is like a return but it's better to use it when dealing with sequences.
        

# Splitting the questions and answers into training and validation sets
"""
Cross validation is a technique that consists of during the training keeping 10 or 15 percent of the training data aside which won't be used
to train the neural network .i.e basically will not be back propagated.
And just to keep track of the predictive power of the model on these observations are like new observations.
So basically we're just testing the model simultenously.
But during the training just to keep track of what it's capable to do on new observations.
And so this validation set that we have to make for both the questions and answers is exactly this 10 or 15 percent of the data on the side.
So let's apply cross-validation and that consist exactly of splitting the questions and answers into four sets the training and validation sets.

The first step we need to do is to get the index that will separate the First 50 percent questions of all our questions [0% to 50%] and 
the next 85 percent of questions of all questions[0% to 85%]. In this way 15% will remain for validating the data simultenously.
"""
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


# Training
batch_index_check_training_loss = 100   # we will check the training loss every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1  # it is the validation loss we will check halfway and at the end of an epoch.
total_training_loss_error = 0   # used to compute the sum of the training losses on 100 batches because we chose to check the loss every 100 batches.

list_validation_loss_error = []  # Description is given below
'''It is list of validation loss error. This list is going to be used for early stopping technique.
This technique is nothing but checking if we managed to reach a loss that is below the minimum of all the losses we have got or not ?
Hence all the losses that we get we're going to put them in a list.'''

early_stopping_check = 0   # Description is given below
'''It is the number of checks each time there is no improvement of the validation loss.
So each time we don't reduce the validation loss early_stopping_check is going to be incremented by 1 until it reaches a certain threhold number given as early_stopping_stop.'''

early_stopping_stop = 100  # early_stopping_check is incremented until it reaches early_stopping_stop and then we stop our training immediately
checkpoint = "./chatbot_weights.ckpt"   # To save the weights which we will be able to load whenever we want to chat with our trained chatbot. That's going to be the file containing the weights.
session.run(tf.global_variables_initializer()) # To run our session by taking our session object and then using the run method. Inside this run method we have to initialize all the global variables.

for epoch in range(1, epochs + 1):  # loop through each epochs
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)): # loop through each batch with padded questions and padded answers
        starting_time = time.time() # to measure the time of the training of each batch , then we'll get ending time and will do the difference between the end time and starting time to get the time of the batch
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0: # So if it is divisible by 100 which we can check by seeing if the rest of the division of batch indexed by batch index check training is equal to zero then that means that we would reach 100 batches or 200 batches or 300. That's basically every 100 batches.
            # Print the epoch, the batch, the average training loss error on 100 batches and the training time on these same 100 batches.
            # {:>3}/{} is to print 3 figures over the total number of epochs
            # {:>4}/{} is to print 4 figures over the total number of batches
            # {:>6.3f} is to print 6 figures with 3 decimals (floating number) over the total training loss error
            # {:d} is to print training time on 100 batches in integer format
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0  # Description is given below
            '''re-initialize the total training loss error to zero because that was just to compute the total 
            training loss error of 100 batches and we were done dealing with these 100 batches.
            So we're going to have to do the same thing for the next 100 batches.'''
        # Calculate Validation Loss Error
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time() # to measure the time of the validation of each batch , then we'll get ending time and will do the difference between the end time and starting time to get the time of the batch
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)): # loop through each batch with padded questions and padded answers
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch, # we don't need optimizer as it is required only in case of training but not in case of validation. Note: This function (seesion.run) returns only one variable (batch_validation_loss_error) as opposed to training, so we removed _, whichyou can see in case of training)
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1}) # unlike training, in case of validation keep_probability is 1 because we need to keep all neurons activated in case of validation.
                total_validation_loss_error += batch_training_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size) # validation loss error per batch, meaning  total_validation_loss_error / total number of batches in validation set
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            # Apply decay to the learning rate
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I apeak better now!!')
                early_stopping_check = 0 # we will reset the early stopping check variable to zero because we increment it only when there is no improvement of the validation error and we reset it to zero whenever we find an improvement.
                # still if we find a lower validation error, then we want to save them all we are going to get first a saver object.
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practise more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
print("Game Over")

############# PART 4 - TESTING THE SEQ2SEQ MODEL ##############



# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint) #It's like a load function.This restore method from the saver class to connect our session and our checkpoint.


# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
    """earlier during preprocessing step we had filterout some of least frequent words from the word2int dictionary.
    That means those least frequent words are not present in the word2int dictionary, but they might be present in the 
    question list. So whenever those words occur in the question, we will try to get the '<OUT>' token which we created earlier.
    This is done using get(). So if the question words are present in the word2int dictionary then corresponding ley value will be used,
    otherwise key value corresponding to '<OUT>' token will be used"""
    
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    """Well remember the essential thing we need to do when working with neural networks it's the fact that
       this question must be into a batch. The neural networks only accept batches of inputs .
       So we need to convert this question into batch. For this we are going to create a fake batch which will contain this question 
       and then some empty questions that will only get zeros.
    """
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question # First row of the fake batch will be the question itself and remaing will be all zeros.
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0] # predicted answer that we are interseted in is the first element which is accessed by index 0.
    # Post processing of the predicted answers meaning that making it into a clean format
    # Ex: i -> I
    #    <EOS> -> .
    #    <OUT> -> Out
    # etc..
    answer = ''
    for i in np.argmax(predicted_answer, 1):  # get the token IDs in the predicted answer.
        if answersints2word[i] == 'i':
            token = 'I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out' 
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
    
            
