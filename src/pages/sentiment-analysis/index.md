---
title: "PyTorch vs. Keras: Sentiment Analysis using Embeddings"
date: "2018-05-26"
path: "/sentiment-analysis/"
category: "Projects"
thumbnail: "thumbnail.jpg"
---

In this writeup I will be comparing the implementation of a sentiment analysis model using two different machine learning frameworks: PyTorch and Keras. I will design and train two models side by side -- one written using Keras and one written using PyTorch.

Keras is a high level machine learning library that offers a suite of tools and functions for rapid experimentation and model development. PyTorch is a more low-level framework, similar in concept to Tensorflow but different in application (TensorFlow is Static computation graph, PyTorch is Dynamic). PyTorch offers numpy like functions enhanced by automatic differentation, as well as pre-built functions for model development such as pre-defined cost functions and optimization routines. 

## Outline

A rough outline of the steps I take below is as follows:

 - Download the Data
 - Do some data exploration
     - Understand dataset
     - Look at some samples
 - Prepare data for training pipeline
 - Define Model Architecure
 - Define Learning / Opimization strategy
 - Train Model
 - Review Results


```python
from keras.datasets import imdb
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
```

## Download Data
The data comes in pre-processed, where each training sample is an array of word indexes based on a list of most frequently used words. 


```python
(x_train, y_train), (x_test, y_test) = imdb.load_data(seed=7)
```

## Exploring the Data


```python
word_index = imdb.get_word_index()
```

    Downloading data from https://s3.amazonaws.com/text-datasets/imdb_word_index.json
    1646592/1641221 [==============================] - 2s 1us/step
    

#### Create key value pair to convert word index to word


```python
word_index = {value: key for key, value in word_index.items()}
```


```python
display(Markdown("**How many words in corpus?**"))
print(len(word_index))
```


**How many words in corpus?**


    88584
    


```python
display(Markdown("**Top 30 most common words:**"))
for i in range(1, 15): print(word_index[i], end='\t')
for i in range(15, 29): print(word_index[i], end='\t')
    
display(Markdown("**Top 20 least common words:**"))
for i in range(88584 - 20, 88584 - 10): print(word_index[i], end='\t')
for i in range(88584 - 10, 88584): print(word_index[i], end='\t')
```


**Top 30 most common words:**


    the	and	a	of	to	is	br	in	it	i	this	that	was	as	for	with	movie	but	film	on	not	you	are	his	have	he	be	one	


**Top 20 least common words:**


    reble	percival	lubricated	heralding	baywatch'	odilon	'solve'	guard's	nemesis'	airsoft	urrrghhh	ev	chicatillo	transacting	sics	wheelers	pipe's	copywrite	artbox	voorhees'	


```python
display(Markdown("**What's the shortest review?**"))
short_review = (0, 100000, 'start with a large number')
for idx, review in enumerate(x_train):
    if len(review) < short_review[1]:
        short_review = (idx, len(review), ' '.join([word_index.get(i - 3, '?') for i in review]))
print('Review # {}({} words):\n\t{}'.format(short_review[0], short_review[1], short_review[2]))

display(Markdown("**What's the longest review?**"))
long_review = (0, 0, 'start with a large number')
for idx, review in enumerate(x_train):
    if len(review) > long_review[1]:
        long_review = (idx, len(review), ' '.join([word_index.get(i - 3, '?') for i in review]))
print('Review # {}({} words):\n\t{}....'.format(long_review[0], long_review[1], long_review[2][:1200]))
```


**What's the shortest review?**


    Review # 729(11 words):
    	? this movie is terrible but it has some good effects
    


**What's the longest review?**


    Review # 18853(2494 words):
    	? match 1 tag team table match bubba ray and spike dudley vs eddie guerrero and chris benoit bubba ray and spike dudley started things off with a tag team table match against eddie guerrero and chris benoit according to the rules of the match both opponents have to go through tables in order to get the win benoit and guerrero heated up early on by taking turns hammering first spike and then bubba ray a german suplex by benoit to bubba took the wind out of the dudley brother spike tried to help his brother but the referee restrained him while benoit and guerrero ganged up on him in the corner with benoit stomping away on bubba guerrero set up a table outside spike dashed into the ring and somersaulted over the top rope onto guerrero on the outside after recovering and taking care of spike guerrero slipped a table into the ring and helped the wolverine set it up the tandem then set up for a double superplex from the middle rope which would have put bubba through the table but spike knocked the table over right before his brother came crashing down guerrero and benoit propped another table in the corner and tried to irish whip spike through it but bubba dashed in and blocked his broth....
    

## Observations
Most of the words in the latter review aren't really good indicators of the sentiment of a review. Super low occuring words, like 'benoit' or 'bubba' do not give any indication of the sentiment, so we probably don't need to include those in our dataset. I'm going to reload the dataset with less words.

### Reloading Dataset, dropping all but top 10k most common words


```python
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(seed=7, num_words=10000)
```


```python
plt.hist([len(i) for i in x_train if len(i) < 1000], bins=20)
plt.gca().set_xlabel('Length of Review')
plt.gca().set_ylabel('frequency of occurence')
plt.gca().set_title('How long are the reviews?')
plt.show()
```


![png](output_13_0.png)



```python
display(Markdown("**Top 30 most common words:**"))
for i in range(1, 15): print(word_index[i], end='\t')
for i in range(15, 29): print(word_index[i], end='\t')
    
display(Markdown("**Top 20 least common words:**"))
for i in range(10000 - 20, 10000 - 10): print(word_index[i], end='\t')
for i in range(10000 - 10, 10000): print(word_index[i], end='\t')
```


**Top 30 most common words:**


    the	and	a	of	to	is	br	in	it	i	this	that	was	as	for	with	movie	but	film	on	not	you	are	his	have	he	be	one	


**Top 20 least common words:**


    avant	answering	smallest	contacts	enlightenment	murphy's	employs	unforgivable	punchline	culminating	talentless	grabbing	soulless	unfairly	grail	retrospect	edged	retains	shenanigans	beaver	

### Observations: Standardize Review Length
Each sample of this dataset is a review of variable length. A dataset where each row is of variable length is no good for inputting into a model. I'm going to pad each review to be the first 100 words. This number is a bit arbitrary, and could probably be even shorter. If I find the model is overfitting, there might be too many input features. How many words does it take for you to understand the sentiment of a review? Clearly from reading the shortest review, the author made his sentiment clear within the first five words.


```python
from keras import preprocessing

X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=100)
X_test  = preprocessing.sequence.pad_sequences(X_test, maxlen=100)
```


```python
print('New train shape:', X_train.shape)
```

    New train shape: (25000, 100)
    


```python
display(Markdown("**Padded version of shortest review:**"))
print(' '.join(word_index.get(i - 3, '?') for i in x_train[729]))

display(Markdown("**Padded version of longest review:**"))
print(' '.join(word_index.get(i - 3, '?') for i in x_train[18853]))
```


**Padded version of shortest review:**


    ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? this movie is terrible but it has some good effects
    


**Padded version of longest review:**


    like taker was about to pass out but the rock broke ? hold only to find himself caught in the ? lock rock got out of the hold and watched taker ? angle rocky hit the rock bottom but taker refused to go down and kicked out angle ? taker up into the angle slam but was rock ? by the great one and ? winner and new wwe champion the rock br br finally there is a decent ? lately the ? weren't very good but this one was a winner i give this ? a a br br
    

## Prepare Data
Fortunately the data as pulled from the keras dataset is already tokenized, but not in tensor form. I will split the train so I have a validation set to evaulate and tune model against, and create a generator function for the data.


```python
from sklearn.model_selection import train_test_split
from torch.utils import data
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

# PyTorch Preperation
if torch.cuda.is_available():
    device = torch.device("cuda")                   # a CUDA device object
    x_train = torch.tensor(x_train, device=device)  # directly create a tensor on GPU
    y_train = torch.tensor(y_train, device=device)  
    x_valid = torch.tensor(x_valid, device=device)
    y_valid = torch.tensor(y_valid, device=device)
else:
    print('CUDA is not available. creating CPU tensors')
    x_train = torch.tensor(x_train, dtype=torch.long)     
    y_train = torch.tensor(y_train, dtype=torch.float)  # directly create a tensor on GPU
    x_valid = torch.tensor(x_valid, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.float)
    
print('new train shape:', x_train.shape)
print('validation size:', x_valid.shape)

train = data.TensorDataset(x_train, y_train)
valid = data.TensorDataset(x_valid, y_valid)
```

    CUDA is not available. creating CPU tensors
    new train shape: torch.Size([20000, 100])
    validation size: torch.Size([5000, 100])
    

## Create Model


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.models import Sequential 
from keras.layers import Flatten, Dense 
from keras.layers import Embedding

torch.manual_seed(1)
class Flat(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    
# PyTorch implementation
class sentimentNet(nn.Module):
    
    def __init__(self, batch_size, num_words, input_size, embedding_size):
        super(sentimentNet, self).__init__()
        
        self.embedding1 = nn.Embedding(num_words, embedding_size) 
        self.flatten = Flat()
        self.linear1 = nn.Linear(input_size * embedding_size, 1)
        
    def forward(self, inputs):
        embed1 = self.embedding1(inputs)
        flatt1 = self.flatten(embed1)
        layer1 = self.linear1(flatt1)
        output = F.log_softmax(layer1, dim=1)
        
        return output

modelT = sentimentNet(batch_size=10, num_words=10000, input_size=100, embedding_size=8)

# Keras implementation
modelK = Sequential([
    Embedding(10000, 8, input_length=100, name='embedding1'),
    Flatten(name='flatten'),
    Dense(1, activation='sigmoid', name='linear1')
]) 

print(modelT)
print('\n\nsentimateNet: Keras')
print(modelK.summary())
```

    sentimentNet(
      (embedding1): Embedding(10000, 8)
      (flatten): Flat()
      (linear1): Linear(in_features=800, out_features=1, bias=True)
    )
    
    
    sentimateNet: Keras
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding1 (Embedding)       (None, 100, 8)            80000     
    _________________________________________________________________
    flatten (Flatten)            (None, 800)               0         
    _________________________________________________________________
    linear1 (Dense)              (None, 1)                 801       
    =================================================================
    Total params: 80,801
    Trainable params: 80,801
    Non-trainable params: 0
    _________________________________________________________________
    None
    

## Define Optimization Strategy


```python
import collections

# PyTorch
hist = collections.defaultdict(lambda:[])
costFN = nn.BCELoss()
optimizer = optim.SGD(modelT.parameters(), lr=1e-3, momentum=0.9)

# Keras
modelK.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['acc']) 
```

## Train Model


```python
from torch.utils import data

datagen = data.DataLoader(train, batch_size=32, shuffle=False)

num_epochs = 3

for epoch in range(num_epochs):
    print("Epoch {} / {}".format(epoch, num_epochs))
    running_loss = 0.0
    
    for i, data_batch in enumerate(datagen, 0):
        inputs, labels = data_batch
        
        optimizer.zero_grad()  # reset grads
        
        outputs = modelT(inputs)
        loss = costFN(outputs, labels)
        loss.backward()
        optimizer.step()
        
         # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            
            hist['loss'] += [loss.item()]
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
```

    C:\Users\joshu\Anaconda3\envs\MLenv\lib\site-packages\torch\nn\functional.py:1474: UserWarning: Using a target size (torch.Size([32])) that is different to the input size (torch.Size([32, 1])) is deprecated. Please ensure they have the same size.
      "Please ensure they have the same size.".format(target.size(), input.size()))
    

    [1,   200] loss: 13.772
    [1,   400] loss: 13.941
    [1,   600] loss: 13.708
    [2,   200] loss: 13.772
    [2,   400] loss: 13.941
    [2,   600] loss: 13.708
    [3,   200] loss: 13.772
    [3,   400] loss: 13.941
    [3,   600] loss: 13.708
    Finished Training
    

### Observations
Loss output is way too high, it is also not decreasing. This is not a good sign; will need to troubleshoot the model. Next steps should be to manually calculate the loss with a single example and look at the output, take a look at the weights and compare to keras Model.


```python
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/10
    20000/20000 [==============================] - 3s 162us/step - loss: 0.1071 - acc: 0.9658 - val_loss: 0.3840 - val_acc: 0.8536
    Epoch 2/10
    20000/20000 [==============================] - 3s 128us/step - loss: 0.0931 - acc: 0.9700 - val_loss: 0.3977 - val_acc: 0.8508
    Epoch 3/10
    20000/20000 [==============================] - 3s 131us/step - loss: 0.0807 - acc: 0.9758 - val_loss: 0.4132 - val_acc: 0.8496
    Epoch 4/10
    20000/20000 [==============================] - 3s 135us/step - loss: 0.0690 - acc: 0.9800 - val_loss: 0.4283 - val_acc: 0.8482
    Epoch 5/10
    20000/20000 [==============================] - 3s 130us/step - loss: 0.0588 - acc: 0.9846 - val_loss: 0.4447 - val_acc: 0.8466
    Epoch 6/10
    20000/20000 [==============================] - 3s 136us/step - loss: 0.0499 - acc: 0.9876 - val_loss: 0.4601 - val_acc: 0.8444
    Epoch 7/10
    20000/20000 [==============================] - 3s 130us/step - loss: 0.0417 - acc: 0.9900 - val_loss: 0.4809 - val_acc: 0.8426
    Epoch 8/10
    20000/20000 [==============================] - 3s 137us/step - loss: 0.0347 - acc: 0.9917 - val_loss: 0.4983 - val_acc: 0.8412
    Epoch 9/10
    20000/20000 [==============================] - 3s 138us/step - loss: 0.0286 - acc: 0.9938 - val_loss: 0.5200 - val_acc: 0.8378
    Epoch 10/10
    20000/20000 [==============================] - 3s 136us/step - loss: 0.0236 - acc: 0.9950 - val_loss: 0.5411 - val_acc: 0.8386
    
