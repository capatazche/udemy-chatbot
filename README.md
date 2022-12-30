# Udemy - Deep Learning and NLP A-Z: How to create a ChatBot

## Course Content
Fun facts:
* Facebook’s page to create a chatbot - itsalive.io
* Movie `Her` - a very smart chatbot
* Movie `Sunspring` - written by Benjamin (an RNN, more specifically, an LSTM). Benjamin is good at writing sentences, but it lacks the big picture.

Deep Learning and NLP Applications:
* Speech Recognition (Transcription)
* Neural Machine Translation (Google used to use phrase based translation algorithms)
* Chatbots (pro: scalability for businesses)
* Q&A
* Text Summarization
* Image Captioning
* Video Captioning

## Section 2: Deep NLP Intuition
Plan of attack:
* Types of NLP
* Classical VS Deep Learning Models
* End-to-end Deep Learning Models
* Bag-of-words
* Seq2Seq Architecture
* Seq2Seq Training
* Beam Search Decoding (how Seq2Seq models comes up with outputs)
* Attention Mechanisms (augmentation of the Seq2Seq model)
* Annex 1 (artificial neural networks ANNs)
* Annex 2 (recurrent neural networks RNNs)

Note: before going into Seq2Seq, go over the annex to know about ANNs and RNNs.

### Types of NLP
Important to not confuse NLP and Deep Learning. They do not necessarily overlap. When they overlap, it is called DNLP (Deep Natural Language Processing). Seq2Seq, which is one of the strongest models, belongs to the DNLP world.

Examples:
* If/Else Rules (questions and predetermined answers, used for chatbots) - NLP
* Audio frequency component analysis (looks at soundwave shapes and compares to a pre-recorded one, used in speech recognition) - NLP
* Bag-of-words model (used for classification) - NLP
* CNN for text recognition (CNNs are mostly used for image/video processing, used for classification) - DNLP
* Seq2Seq (many applications) - DNLP

### End-to-end (ETE) Deep Learning Models
Seq2Seq is a type of ETE deep learning model. 

Companies history:
1. Use people as customer service to understand the problem and provide a solution
2. Companies implement automated menus, "For billing, press 1", etc. This is just a combination of if/else.
3. Actually talk to a robot explaining what you are calling about. It uses deep learning (sometimes simpler nlp) to understand what you are saying but it still uses a flow chart (or decision tree). Very similar to the previous but instead of having a menu, you talk to it.
4. The implementation is a combination of two deep learning models. One to understand what you are saying and one to actually understand the meaning of it. The problem is that each one trains on its own but there is a missing link between these models. This can still end in frustration for the customer. In other words, it lacks a single thing that takes the whole thing into account.
5. The current frontier, one model running the whole show - a single model that takes it all into account to provide a solution. This is considered an ETE deep learning model. These types of models exist in more areas than just NLP.

### Bag-of-words Model
Imagine we want to create a model that give s a simple answer `Yes` or `No` to an email. To do so, consider the following vector:
```
const vector = [0,0,0, ... ,0] // 20000 elements long (average native speaker knows 20000 words)
```

Therefore, we have a position for every word in the English language. In addition, consider we are reserving the first, second, and last position for special info.
- `vector[0] // position reserced for SOS (start of sentence)`
- `vector[1] // position reserved for EOS (end of sentence)`
- `vector[19999] // position reserved for Special Words (any words not included)`

The vector is effectively our bag-of-words.

```
// Consider the following email:

// Hello Kirill,
// Checking if you are back in Oz. Let me know if you are around and keen to sync on how things are going. I defo could use some of your creative thinking to help mine :)
// Cheers,
// V

// To this, Gmail provides three possible answers (consider for later):
// 1. `Yes, I'm around.`
// 2. `I'm back!`
// 3. `Sorry, I'm not.`

// Our bag of words (with a lot of zeros...)
[
3, // There are 3 start of sentences
2, // There are 2 end of sentences
0,
1, // Count for word "keen"
...
4 // ("Kirill":1 count),(",":2 counts),("V":1 count) 
]

```

To train this bag-of-words model, I would need a sample of emails and my answers (Yes or No) for each one. We could train a simple logistic regression as well as a neural network with these vectors. One represents simple NLP while the other represents DNLP. The biggest limitation of this bag-of-words approach is that it produces a simple "Yes" or "No" much less a conversation.

### Seq2Seq Architecture (Part 1)
Issues with the bag-of-words model:
* Fixed-size input (20,000 in the case described above)
* Does not take word order into account, it just counts occurences of each word
* Fixed-size output (in the case described above, it only answers "Yes" or "No")

RNNs can be used to mitigate the bag-of-words weaknesses mentioned above.

Different types of architectures for RNNs:
* one to one
* one to many
* many to one
* many to many (with delayed results, once the inputs end, the output starts)
* many to many (with immediate results, for every input, there is an output)

For our purposes, a chatbot, two of them would be useful - the two that are "many to many". The reason is because our text can have a lot of inputs (words) and output should be variable.

As an example, consider the following text: "Hello Kirill, Checking if you are back to Oz.". Instead of having a vector thats is 20,000 in length with space representing one word in the language (and the value held the amount of occurences of such word), he have a different vector. For this approach we would have a vector as long as our text (plus two for beginning and end). And the value held instead if the index of the word in our 20,000 word collection.

``` 
"Hello Kirill, Checking if you are back to Oz."
[1, 5, 0, 9, 23, 7, 41, 102, 19, 4, 0, 20, 2] 
# Remember how 0 represents words not in the "language", 1 is SOS, and 2 is EOS
```

### Seq2Seq Architecture (Part 2)
So consider the same sentence above - "Hello Kirill, Checking [...]". The neural network would take a word for unit of time and once it reaches the EOS, then it would start answering (many to many RNN). Now how does it decide what word to use out of the 20,000? As it consumes the words it updates the probabilities for each of the 20,000 words and then chooses the most fit.

### Seq2Seq Training
There are a bunch of weights that need to be "optimized" for a network to learn. When training, we have full questions and answers to provide to the network. These weights aim to maximize the chances of the network choosing the right words in the right order.

### Beam Search Decoding
Greedy Decoding vs Beam Search Decoding.

Greedy Decoding:
After the input is processed, the NN starts choosing words to answer. In a geedy decoding approach, the NN will always choose the highest probability word at every step. 

Beam Search Decoding:
In contrast, the beam search decoding approach will start with more options and study them, it will not just choose the highest probability one for each word added. For this example say we want to study 3 different options, so we have 3 beams. These 3 beams start with the 3 highest probability words to begin an answer with. Then for each, it will choose the next 3 highest probability options. Therefore, we now have 9 total combinations. It will continue to develop the tree in this fashion until they all reach EOS. With so many combinations, how do we choose one? Well, we end up choosing the path with the highest joint probability. With this approach, we can save computing power and memory space by truncating the branches growing out. It decides to truncate if when developing these branches it detects that a branch has no chance at competing (low joint probability). This way we can mitigate the exponentiality of this approach.

### Attention Mechanisms (Part 1)
The weak point of an LSTM is the last step of the encoder (hn). This last step contains the whole meaning of the input (vector meaning). Part of the issue is that hn is of a fixed dimension while the input is of variable length. For short pieces of texts, this is not much of an issue. Nevertheless, as the text becomes longer, it becomes an issue. An analogy of the issue portrayed above by a translator: "it is like reading a whole sentence, closing your eyes and spitting out what you think it translates to". As humans we have the benefit of going back and revising. Here is where the concept of "Attention" comes in. 

In a sense, attention allows the decoder to look back at information available at the encoder. When looking back, the decoder will know at every step what part of the input to pay attention to thanks to weights assigned to the steps of the encoder during training. These weight are the put together in a context vector and this context vector is a new input for the decoder step. Note that the values in the context vector are specific to the decoder step. Depending on the step, it know what part of the input to pay attention to.

### Attention Mechanisms (Part 2)
As a side note, there is global attention as well as local attention. Global attention looks at the whole input while local attention might look at just some chunk of the input.

## Section 3: Building a ChatBot with Deep NLP
### ChatBot - Step 1
Plan of Attack:
* Installing Anaconda and Getting the Dataset (movies)
* Data Preprocessing
* Building the Seq2Seq model
* Training the Seq2Seq model
* Testing the Seq2Seq model

### ChatBot - Step 2
Installing Anaconda and getting the dataset. Installed Anaconda by downloding my distribution from [Anaconda download](https://www.anaconda.com/products/distribution). Then we need to install Tensorflow inside of a conda virtual env.

### ChatBot - Step 3
This step involves getting our dataset: The Cornell Movie Corpus Dataset. This dataset contains thousand of conversations from more than 6000 movies.

For the legacy dataset (the one being used by the course): [legacy dataset](https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html)
For a more modern version of getting the dataset: [modern dataset](https://convokit.cornell.edu/documentation/movie.html)

## Section 10: Annex 1: Artificial Neural Networks
Plan of attack:
* The Neuron
* The Activation Function
* How do Neural Networks work?
* How do Neural Networks learn?
* Gradient Descent
* Stochastic Gradient Descent
* Backpropagation

### The Neuron
Physical neuron parts:
* Dendrites (receiver)
* Axon 
* Neuron

When a neuron communicates with another one, synapse occurs. In data science, we try to replicate this synapse phenomenon.

Data sience "neuron" parts:
* Input signals (can be multiple, as with dendrites)
* Output signal
* Neuron

In terms of our brains, the initial input signals are the senses. In terms of data science, the initial input values need to be standardized (mean of 0 or variance of 1) or sometimes normalized (substract min and divide over max - min) and the need to be independent. The output value can be continuous (like price), binary, or a categorical value. For training purposes, the input values along with their output value is a single observation. In terms of data science, every synapses effectively has a weight. These weight are what gets adjusted in the process of training (backpropagation, gradient descent, etc.). The neuron effectively does a weighted sum of all the input values it is getting and then it applies an activation function. Depending on the activation function, the neuron will decide to pass on the signal or not.

### The Activation Function
The activation function decides when to pass signal or not and what value to pass.

Types of activation functions:
* Threshold function (sometimes called "step")
* Sigmoid function (useful in final or output layer specially when representing probability)
* Rectifier function
* Hyperbolic tangent function (somtimes called "tanh")

A common architecture is to have a rectifier activation function for the hidden layer and a sigmoid activation function for the output layer.

### How do NNs Work?
As an example, let's look at property valuation. For this section we will focus on NNs applications and assume we have an already trained NN.

Imagine the inputs:
* x1 = area
* x2 = bedroom
* x3 = distance to city
* x4 = age

The output layer would represent the price of the property. In a case where we do not have hidden layers, the output would be a direct weighted sum of the inputs. Nevertheless, to leverage the power and increased accuracy of a NN, then we need a hidden layer. The hidden layer builds more complex features from its inputs. Important to note that some inputs might have a weight of 0. A weight of zero would mean that the specific input is irrelevant to the feature being built in the neuron. In turn this could also mean that if a single input is incredibly important for determining the price of a house (say, age) it could pass onto the hidden layer by itslef with the rest of the neuron's input's weights being 0.

### How do NNs Learn?
In a NN you give it inputs and tell it what you want as outputs. It creates the path by itself. As an example consider a single layer feat-forward, also called Peceptron, NN. 

Q: How does Perceptron learn? 
A: Imagine the training data is a single observation. It loops (or more accurately, feeds the result back) and adjusts weights trying to minimize the cost function ((1/2)(y'-y)^2) (y' is the predicition, y is the actual). Now, if we have several observations, it tries to reduce the sum of their individial const function results. This whole process is called **backpropagation**.

### Gradient Descent
How do weights get adjusted during brackpopagation? If you had a single weight to operate on, a brute force approach to reduce the cost function does not seem too crazy. Nevertheless, as more weights are used in a model, we confront the "Curse of Dimensionality"(COD). The COD implies that as more inputs, layers, and features are introduced, the amount of possible combinations increments exponentially. Therefore, brute force approaches are completely out of the question. This where Gradient Descent comes into the picture. The Gradient Descent approach differentiates the cost function ((1/2)(y'-y)^2) to finds its slope. Note that the cost function has the shape of a big U. At the bottom of the U is the minimum value for the cost function (no slope, flat). Knowing the slope at any given point informs if we need to move further right or left to reach that minimum point for the cost function. Imagine this whole process as a ball trying to roll down (in as many dimensions as there are weights!).

PS: there are parameters you can adjust to tweak how the Gradient Descent behaves.

### Stochastic Gradient Descent
Gradient Descent is an efficient method when approaching the cost function. It allows to reach the minimum faster. Nevertheless, Gradient Descent only works if the cost function is convex. If the cost function has a different shape, the Gradient Descent approach could quite possibly end up finding a local minimum instead of the global minimum. Stochastic Gradient Descent (SGD) can help us here. When Gradient Descent adjusts weight to reduce cost, it does it taking into account the aggregated cost for all observations (sometimes called batch Gradient Descent). In contrast, SGD adjusts weights taking into account the cost of a single observation at a time. Therefore, for an iteration of training, Gradient Descent adjusts weights once while SGD adjusts weights as many times as there are observations. As a consequence, SGD fluctuations are wider but it also converges faster. Gradient Descent is deterministic while SGD is not deterministic (converging to the same reult everytime you train vs not).

There exists a method in between called the Mini Batch Gradient Descent.

### Backpropagation
During the process of backpropagation, due to how the algorithm is structured, all weights are adjusted at the same time. Thanks to some complex mathematics, backpropagation understand what amount of error is coming from where and that is why this is feasible.

Steps to train an ANN with SGD:
1. Randomly initialise the weights to small numbers close to 0 (but not 0)
2. Input the first observation of dataset in input layer 
3. Forward-propagate all the way to result y
4. Comapre predicted result with actual result and calculate error
5. Backpropagate (update the weights according to how much responsibility they hold for the error)
6. Repeat 1 - 5 and update weights after each observation
7. When whole training set passed through the ANN, that makes an epoch. We want several epochs before calling the training done.

Note that the "learning rate" (a parameter that can be tweaked) decides by how much a weight is adjusted during backpropagation.

## Section 11: Annex 2: Recurrent Neural Networks
Plan of attack:
* The idea behind RNNs
* The Vanishing Gradient Problem
* Long Short-Term Memory architecture (LSTM NNs)
* Practical Intuition
* Extra: LSTM Variations

### What are RNNs?
RNNs are some of the most advanced algorithms nowadays for supervised training.

Supervised:
* ANN: Regression and Calssification
* CNN: Computer Vision
* RNN: Time-Series Analysis

Deep learning tries to simulate how the brain works. Weights represent long term memory. If we try to match parts of the brains to their corresponding NN, then we would map:

* Cerebrum (frontal, perietal, temporal, occipital)
	* Temporal lobe (recognition memory, or long term memory) -> ANN
	* Occipital (vision) -> CNN
	* Frontal (they are like short term memory) -> RNN
	* Periteal (sensation and perception)
* Cerebellum
* Brainstem

RNN add a new dimension to a standard NN, time. So, the hidden layer/layers not only feed the output layer but also feed themselves. It's as if they have "short term memory" of the values they just had. 

Let's look at a couple of uses for RNNs (notice how all of them need context and that is where short term memory comes in):
- One to many: an image feed to an algorithm to describe a picture. "black and white dog jumps over bar"
- Many to one: sentiment analysis needs to take many words and come up with a single score. "positive score: 86%"
- Many to many: Google Translator uses them for languages where nouns have gender since it needs short term memory to use the proper adjective
- Many to many: subtitles for movies

### Vanishing Gradient Problems for RNNs
This porblem was first presented by Sepp Hochreiter. To best understand this problem, consider the gradient descent approach. When it comer to RNN, we will not have a singular const function result, we will have as many cost function results as units of time (error at t-3, t-2, t-1, t, t+1). Let's isolate the error at t (Et). The question becomes, how do I backpropagate this Et? It should backpropagate towards all neurons that were involved in the production of its result. The weight involved in the recurrent aspect of the RNN (the hidden layer feeding on itself, or in other words, the time component) is called "recurrent weight" (Wrec). Therefore the hidden layer will multiply by this factor for as many units of time there are. Considering these weights are initialized to be small but not zero, when multiplied several times, the number just becomes smaller and smaller. In turn, this means that the further from Et you are (say Et-3), the slower your weights will get updated. In other words, parts of the network are being trained from parts that are not trained. This created a vicious cycle. Now, if you consider Wrec to be a large number, then instead of a vanishing problem, you would have an exploding problem!

How do we try to mitigate this issue:
1. Exploding 
	* Truncated backpropagation - stop backpropagation at some arbitary point
	* Penalties - gradient being penalized or artificailly reduced
	* Gradient clipping - max limit for the gradient
2. Vanishing
	* Weight initialization - smart initial numbers
	* Echo state networks - designed to solve the vanishing gradient issue
	* LSTMs - considered to be the go to network for RNN implementation

Important to note that when setting up an RNN, you can tweak a parameter to instruct how many steps to look back (or units of time).

### LSTMs
Previously, we discussed about the vanishing gradient problem. How do we solve this problem? In the previous lecture, we showed ways to mitigate this (truncated backpropagation, penalties, etc). In essence, the way LSTM mitigates (solves) this problem is having Wrec hold the value of 1. Therefore it does not vanish nor explode as it is applied on itself.

### Practical Intuition
Specific cells can try to keep track of different things. As the LSTM gets trained, it learns what important things it needs to keep track of (remember) and when to reset it (or forget). Examples:
* A cell can try to predict when the end of a sentence might be coming based on average sentence length
* A cell can identify what is inside quotes and what is out
* A cell that can idenfity the condition of an if statement
* A cell that can identify the depth of a nested expression in code
* A cell that can be active when it is perceving a URL

### LSTM Variations
* Peephole - memory is shared for all sigmoids (or activation functions)
* Forget valve and memory valve is combined
* Gated recurring units (GRUs) - gets rid of memory pipeline

## Out of Scope Questions
* What are CNNs (convolutional neural networks) and how do they work?
* When to standardize and when to normalize the data? - Resource: ["Efficient BackProp" by Yann LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
* More on activation functions - Resource ["Deep Sparse Rectifier Neural Netowork" by Xavier Glorot et al. (2011)](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)
* More on how does a NN learn and backpropagation - Resource ["A list of cost functions used in neural networks, alongside applications" Cross Validated (2015)](http://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)
* More on Gradient Descent - Resource ["A Neural Network in 13 lines of Python (Part 2 - Gradient Descent) by Andrew Trask (2015)"](https://iamtrask.github.io/2015/07/27/python-network-part2/)
* More on the Vanishing Gradient Problem - ["On the difficulty of training recurrent neural networks" by Razvan Pascanu et al. (2013)](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)
* More on LSTM - ["Understanding LSTM Networks" by Christopher Olah (2015)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* More on practical intuition regarding RNNs - ["The Unreasonable Effectiveness of Recurrent Neural Networks" by Andrej Karpathy (2015)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* More on variations of LSTMs - ["LSTM: A Search Space Odyssey" by Klaus Greff et al. (2015)](https://arxiv.org/pdf/1503.04069.pdf)
* More on Attention (for translation) - ["Effective Approaches to Attention-based Neural Machine Translation" by Minh-Thang Luong et al. (2015)](http://aclweb.org/anthology/D15-1166)