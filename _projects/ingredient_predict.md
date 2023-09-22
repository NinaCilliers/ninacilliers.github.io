---
layout: page
title: What's for dinner?
description: Generating and naming recipes from parital ingredient lists using LSTM.
img: /assets/img/ingredient_predict/food_stock.jpg
importance: 1
category: Featured projects
---


<h1>What's for dinner?</h1>
<h4><i> Generating and naming recipes from parital ingredient lists</i></h4>
<h4><br></h4>
<h2>Introduction</h2>
In this project we create models to (1) generate recipes from partial ingredient lists and (2) name the generated recipes. This training dataset consists of over 180k recipes covering 18 years of user interactions and uploads on Food.com and is available on [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). 

The generated ingredient lists are coherent and would result in tasty meals, and the recipe names are accurate and cleaver, even with unusual ingredient pairings as inputs. For example, when given chicken, rice, and grapefruit – a seemingly inconsistent ingredient list – our model adds Asian ingredients and calls the recipe Teriyaki Chicken. Further, the model takes the seemingly desperate pairing of black beans and chocolate and suggests creating a gluten-free brownie recipe that I have actually had and enjoyed. 

Both models employ the Long Short-Term Memory (LSTM) architecture, which is a sophisticated recurrent neural network (RNN). RNNs are designed to process input sequences and produce output that considers both the current input and the accumulated context from previous time steps, achieved by maintaining an internal state that gets updated over time. Within an LSTM, a gate cell structure is utilized to selectively forget irrelevant information, incorporate essential information, and make predictions. The LSTM structure incorporates three distinct gates, as illustrated below.

![png](/assets/img/ingredient_predict/lstm.png)
<br>
[LSTM structure](https://towardsai.net/p/deep-learning/create-your-first-text-generator-with-lstm-in-few-minutes#df05)

All input text is encoded using pre-trained vectors from the [Word2Vec](https://code.google.com/archive/p/word2vec/) model trained by Google on the Google News dataset. The word embeddings are fine-tined after the other model weights are optimized. 

The first model generates viable ingredient lists from partial ingredient lists or a single ingredient using LSTM cells. Full ingredient lists for each recipe are broken into partial lists to generate more training data. The number of LSTM layers and LSTM cells was optimized through a grid-search, and a single layer of 500 LSTM cells was identified as the top performing model architecture. Generated recipe lists  are generated sequentially, and each generated prediction is input to future predictions. 

The second model uses a seq2seq model architecture. As implied by the name, seq2seq models take sequences as inputs and generate output sequences. In our case, generated recipes from the first model are input and recipe names are output. Seq2seq models use an encoder and decoder to convert ingredients to a context vector and generate output sequences. A temperature factor is incorporated to increase the randomness of the predicted names, which resulted in more creative recipe names. Predictions are made sequentially using an encoder and decoder from updated input sequences. 

![png](/assets/img/ingredient_predict/seq2seq.png)
<br>
[Seq2seq architecture](https://www.geeksforgeeks.org/seq2seq-model-in-machine-learning/)

<h2><br></h2>
<h2>Data Cleaning</h2>
<h3>Ingredients </h3>

```python
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle

import tensorflow as tf
import gensim.downloader as api
from gensim.models import Word2Vec
import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.models import Model
from keras.preprocessing.text import Tokenizer

import seaborn as sns
from wordcloud import WordCloud
```


```python
#precleaned dataframe from authors is used to clean data.
precleaned_key = pd.read_pickle('ingr_map.pkl')
cleaning_key = dict(zip(precleaned_key['raw_ingr'],precleaned_key['replaced']))
precleaned_key[['raw_ingr','replaced']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>raw_ingr</th>
      <th>replaced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>medium heads bibb or red leaf lettuce, washed,...</td>
      <td>lettuce</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mixed baby lettuces and spring greens</td>
      <td>lettuce</td>
    </tr>
    <tr>
      <th>2</th>
      <td>romaine lettuce leaf</td>
      <td>lettuce</td>
    </tr>
    <tr>
      <th>3</th>
      <td>iceberg lettuce leaf</td>
      <td>lettuce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>red romaine lettuce</td>
      <td>lettuce</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11654</th>
      <td>soybeans</td>
      <td>soybean</td>
    </tr>
    <tr>
      <th>11655</th>
      <td>goose</td>
      <td>goose</td>
    </tr>
    <tr>
      <th>11656</th>
      <td>ajwain</td>
      <td>ajwain</td>
    </tr>
    <tr>
      <th>11657</th>
      <td>brinjals</td>
      <td>brinjal</td>
    </tr>
    <tr>
      <th>11658</th>
      <td>khoya</td>
      <td>khoya</td>
    </tr>
  </tbody>
</table>
<p>11659 rows × 2 columns</p>
</div>






```python
class CleanRecipes():
    '''Imports and cleans raw recipe data. Returns list of ingredients.'''
    def __init__(self,cleaning_key=cleaning_key): 
        self.raw_recipes = self.import_data()
        self.input_sequences = self.raw_recipes['ingredients'].apply(self.string_to_list).to_list()
        print(self.input_sequences)
        self.cleaned_ingredients = []
        self.cleaning_key = cleaning_key

    def __call__(self):
        self.cleaned_ingredients = self.clean_text()
        return self.raw_recipes, self.input_sequences, self.cleaned_ingredients

    def import_data(self):
        #'''Import cleaned recipe, raw recipe and ratings data'''
        raw_recipes = pd.read_csv('RAW_recipes.csv')
        del raw_recipes['contributor_id']
        del raw_recipes['submitted']
        del raw_recipes['tags']
        del raw_recipes['steps']
        del raw_recipes['description']
        raw_recipes = raw_recipes.set_index('id')
        return raw_recipes

    def string_to_list(self,s):
        '''Converts a string that is formatted like a list to a list'''
        l = re.findall(r'\w[\w\s]+',s)
        return l

    def clean_text(self):
        '''Clean text'''
        for recipe in self.input_sequences:
            ingredient_bundle = []
            for ingredient in recipe:
                try:
                    ingredient_bundle.append(self.cleaning_key[ingredient])
                except:
                    ingredient_bundle.append(ingredient)
            self.cleaned_ingredients.append(ingreSdient_bundle)
        return self.cleaned_ingredients
```


```python
#Import and clean all recipes
raw_recipes, input_sequences, cleaned_sequences = CleanRecipes()()
print('Example ingredient list')
cleaned_sequences[:1]
```
    Example ingredient list

    [['winter squash',
      'mexican seasoning',
      'mixed spice',
      'honey',
      'butter',
      'olive oil',
      'salt']]



<h3><br></h3>
<h3> Cleaning recipe names </h3>


```python
class CleanNames():
    '''Takes raw recipe names and returns cleaned, single words in a list with EOS and SOS tags'''
    def __init__(self):
        self.names = raw_recipes['name'].to_list()
        self.cleaned_names = []
        
    def __call__(self):
        for name in self.names:
            name = self.de_contract(name)
            name = self.add_tags(name)
            name = list(filter(self.empty_check,str(name).split(' ')))
            self.cleaned_names.append(name)
        return self.cleaned_names
        
    def de_contract(self,name):
        '''Removes common contractions from recipe names'''
        s = str(name)
        s = s.replace('can t','cannot')
        s = s.replace('don t','do not')
        s = s.replace('it s','it is')
        s = s.replace('won t','would not')
        s = s.replace('aren t','are not')
        s = s.replace('there s','there is')
        s = s.replace('you d', 'you would')
        s = s.replace('you ve', 'you have')
        s = s.replace('you ll', 'you will')
        return s      
    
    def add_tags(self,name):
        '''Adds SOS and EOS tags'''
        name = 'SOS '+name+' EOS'
        return name
    
    def empty_check(self,s):
        '''Checks if word is only a space'''
        if s != '':
            return True
        else: return False
```


```python
cleaned_names = CleanNames()()
print('Example names:')
cleaned_names[:10]
```

    Example names:

    [['SOS', 'arriba', 'baked', 'winter', 'squash', 'mexican', 'style', 'EOS'],
     ['SOS', 'a', 'bit', 'different', 'breakfast', 'pizza', 'EOS'],
     ['SOS', 'all', 'in', 'the', 'kitchen', 'chili', 'EOS'],
     ['SOS', 'alouette', 'potatoes', 'EOS'],
     ['SOS', 'amish', 'tomato', 'ketchup', 'for', 'canning', 'EOS'],
     ['SOS', 'apple', 'a', 'day', 'milk', 'shake', 'EOS'],
     ['SOS', 'aww', 'marinated', 'olives', 'EOS'],
     ['SOS', 'backyard', 'style', 'barbecued', 'ribs', 'EOS'],
     ['SOS', 'bananas', '4', 'ice', 'cream', 'pie', 'EOS'],
     ['SOS', 'beat', 'this', 'banana', 'bread', 'EOS']]



<h2><br></h2>
<h2>EDA</h2>
<h3>Ingredient frequencies</h3>


```python
#top words
def get_top(cleaned_sequences):
    '''Returns a sorted list of ingredients by occurance'''
    ingred = [x for sublist in cleaned_sequences for x in sublist]
    ingred_counts = Counter(ingred).most_common()
    #top_ingredients = [ingred_counts[n][0] for n in range(len(ingred_counts))]
    return ingred_counts
```


```python
tops = get_top(cleaned_sequences)
sorted_ingredients = [x[0] for x in tops]
sorted_counts = [x[1] for x in tops]
count_range = 10 
```
<details>
    <summary>Click to expand hidden code.</summary>
    <pre>
    plt.figure(figsize=(10,4))
    plt.style.use('ggplot')
    plt.subplot(1,2,1)
    plt.bar(sorted_ingredients[:count_range], sorted_counts[:count_range])
    plt.xticks(rotation=90)
    plt.ylim((0,90_000))
    plt.title('Most common ingredients');
    </pre>
</details>


![png](/assets/img/ingredient_predict/output_11_0.png) 
    

<h3><br></h3>
<h3>Sequence lengths</h3>

```python
#ingredient list length
lengths = [len(x) for x in cleaned_sequences]
```

<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    plt.style.use('ggplot')
    sns.histplot(pd.DataFrame(lengths),legend=False)
    plt.title('Ingredients per recipe')
    plt.xlabel('Length')
    plt.show()
    </pre>
</details>
    
![png](/assets/img/ingredient_predict/output_13_0.png) 

```python
#Recipe name length
lengths = [len(x)-2 for x in cleaned_names] #excludes SOS and EOS
```
<details>
    <summary>Click to expand hidden code.</summary>
    <pre>
    plt.style.use('ggplot')
    sns.histplot(pd.DataFrame(lengths),legend=False)
    plt.title('Words in a recipe name')
    plt.xlabel('Length')
    plt.show()
    </pre>
</details>

![png](/assets/img/ingredient_predict/output_14_0.png) 
    
<h3><br></h3>
<h3>Word clouds</h3>

```python
#word cloud by word freqency
all_words = ' '.join([x for sublist in cleaned_sequences for x in sublist])
all_names = ' '.join([x for sublist in cleaned_names for x in sublist if x != 'EOS' if x != 'SOS'])
```
<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    wordcloud = WordCloud().generate(all_words)
    namecloud = WordCloud().generate(all_names)
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Ingredient frequencies')
    plt.subplot(2,1,2)
    plt.imshow(namecloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Name frequencies');
    </pre>
</details>

![png](/assets/img/ingredient_predict/output_16_0.png) 
    

<h2><br></h2>
<h2>Preprocessing text</h2>
<h3>Tokenizing recipes and names</h3>


```python
pad_value = int(0)
```


```python
class TokenizeRecipes():
    def __init__(self, input_sequences, ngram=True):
        self.tokenizer = Tokenizer()
        self.input_sequences = input_sequences
        self.ngram = ngram
    
    def __call__(self, names=False):
        self.tokenizer, self.ngram_sequences, self.totalwords = self.get_sequence_of_tokens()
        self.max_length = self.find_max()
        self.padded_sequence = self.pad_sequences()
        self.padded_sequence = np.array(self.padded_sequence)
        self.predictors, self.label = self.padded_sequence[:,:-1],self.padded_sequence[:,-1]
        if names:
            self.predictors, self.label = self.delete_only_ones()
        return self.tokenizer, self.max_length, self.totalwords, self.label, self.predictors, self.padded_sequence

    def get_sequence_of_tokens(self):
        '''Tokenization'''    
        self.tokenizer.fit_on_texts(self.input_sequences)
        self.total_words = len(self.tokenizer.word_index) + 1

        self.ngram_sequences = []
        for line in self.input_sequences:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            if self.ngram:
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i+1]
                    self.ngram_sequences.append(n_gram_sequence)
            else: self.ngram_sequences.append(token_list)
        return self.tokenizer, self.ngram_sequences, self.total_words

    def find_max(self):
        '''find max token length'''
        self.max_length = max(len(x) for x in self.input_sequences)
        return self.max_length

    def pad_sequences(self):
        '''pad sequences (built in function returns a recursion error)'''
        self.padded_sequence = []
        for sequence_in in self.ngram_sequences:
            zeros = (self.max_length-len(sequence_in))
            padded = [pad_value for zero in range(zeros+1)]
            padded.extend(sequence_in)
            self.padded_sequence.append(padded)
        return self.padded_sequence
    
    def delete_only_ones(self):
        '''Delete names that only contain SOS'''
        to_be_deleted = []
        for i in range(len(self.predictors)):
            if self.predictors[i][-1] == 1:
                to_be_deleted.append(i)
                
        self.predictors = np.delete(self.predictors, to_be_deleted, axis=0)
        self.label = np.delete(self.label, to_be_deleted, axis=0)
        return self.predictors, self.label
```


```python
#tokenizing ingredients
tokenizer,max_length_,total_words,label,predictors,_ = TokenizeRecipes(cleaned_sequences)()
```


```python
#tokenize names
tokenizer_names, max_length_name_, total_words_names,_,_,name_sequences = TokenizeRecipes(cleaned_names, ngram=False)()
#_, _, _, _, padded_names = TokenizeRecipes(cleaned_names, ngram=False)(names=True)
_,_,_,_,_,recipe_sequences = TokenizeRecipes(cleaned_sequences, ngram=False)()
```


```python
recipe_sequences
```




    array([[   0,    0,    0, ...,    2,    6,    1],
           [   0,    0,    0, ...,    9,   18,  141],
           [   0,    0,    0, ...,    1,   74,   23],
           ...,
           [   0,    0,    0, ...,    1,   26,  657],
           [   0,    0,    0, ...,  516,  618, 2373],
           [   0,    0,    0, ...,  316,   21,   33]])




```python
name_sequences
```




    array([[   0,    0,    0, ...,  109,   45,    2],
           [   0,    0,    0, ...,  122,   72,    2],
           [   0,    0,    0, ...,  953,   56,    2],
           ...,
           [   0,    0,    0, ...,  486,  127,    2],
           [   0,    0,    0, ...,   39, 1459,    2],
           [   0,    0,    0, ...,  467,   17,    2]])



<h3><br></h3>
<h3>Embedding text using Word2Vec</h3>


```python
api.info(name_only=True)
```




    {'corpora': ['semeval-2016-2017-task3-subtaskBC',
      'semeval-2016-2017-task3-subtaskA-unannotated',
      'patent-2017',
      'quora-duplicate-questions',
      'wiki-english-20171001',
      'text8',
      'fake-news',
      '20-newsgroups',
      '__testing_matrix-synopsis',
      '__testing_multipart-matrix-synopsis'],
     'models': ['fasttext-wiki-news-subwords-300',
      'conceptnet-numberbatch-17-06-300',
      'word2vec-ruscorpora-300',
      'word2vec-google-news-300',
      'glove-wiki-gigaword-50',
      'glove-wiki-gigaword-100',
      'glove-wiki-gigaword-200',
      'glove-wiki-gigaword-300',
      'glove-twitter-25',
      'glove-twitter-50',
      'glove-twitter-100',
      'glove-twitter-200',
      '__testing_word2vec-matrix-synopsis']}




```python
embed_model = api.load('word2vec-google-news-300')
```


```python
embed_size = 300
```


```python
class EmbedWords():
    def __init__(self,cleaned_sequences, embed_size=embed_size,embed_model=embed_model, tokenizer=tokenizer):
        self.cleaned_sequences = cleaned_sequences
        self.embed_size = embed_size
        self.embed_model = embed_model
        self.word_index = tokenizer.word_index.items()
        self.unique_words = self.find_unique_word()
       
    def __call__(self):
        self.word_embeddings = self.encode_words()
        return self.unique_words, self.word_embeddings
        
    def embed_one_word(self,word):
        '''Encode a single recipe'''
        self.word_embedding = []
        try:
            self.word_embedding.append(self.embed_model[word])
        except:
            self.word_embedding.append(np.zeros((self.embed_size)))
        return self.word_embedding
    
    def find_unique_word(self):
        '''Lists unique words'''
        return list(set([x for sublist in self.cleaned_sequences for x in sublist]))
    
    def encode_words(self):
        '''Encode all recipes'''
        self.word_embeddings = np.zeros((len(self.unique_words)+1,self.embed_size))
        for word,i in self.word_index:
            self.word_embeddings[i,:] = self.embed_one_word(word)[0]
        return self.word_embeddings
    
```


```python
#creating embeding matrices
unique_ingredients,recipe_embeddings = EmbedWords(cleaned_sequences)()
unique_names, name_embeddings = EmbedWords(cleaned_names)()
```

<h3><br></h3>
<h3>Shuffle ingredient lists</h3>


```python
def shuffle(l):
    dim = max((l[0].shape[0],l[0].shape[1]))
    idx = np.random.permutation(dim)
    l = [item[idx-1] for item in l]
    return l

predictors, labels= shuffle([predictors,label])
```

<h3><br></h3>
<h3>Creating labels for recipe names</h3>


```python
name_labels_ = name_sequences[1:]
name_sequences_ = name_sequences[:-1]
recipe_sequences = recipe_sequences[:-1]

cut_name = []
label_name = []
for name in name_sequences:
    name_ = name[:-1]
    label = name[1:]
    cut_name.append(name_)
    label_name.append(label)
cut_name = np.array(cut_name)
label_name = np.array(label_name)
```

<h2><br></h2>
<h2>Defining ingredient generator architecture</h2>


```python
def create_model(LSTM_num,
                 LSTM_layers,
                 embed_size=embed_size, 
                 unique_ingredients=unique_ingredients, 
                 max_length=max_length_, 
                 recipe_embeddings=recipe_embeddings,
                 lr=0.001,
                 transfer=False,
                 mask_zero=True,
                 verbose=1):
    '''Builds LSTM model'''
    input_len = max_length
    ingredient_len = len(unique_ingredients)+1
    
    inputs = tf.keras.Input(shape=(input_len,))
    x = Embedding(ingredient_len, embed_size, weights=[recipe_embeddings], trainable=transfer, mask_zero=mask_zero, input_length=input_len)(inputs)
    if LSTM_layers > 1:
        for n in range(0,LSTM_layers-1):
            x = LSTM(LSTM_num, return_sequences=True)(x)
            x = Dropout(0.2)(x)
    x = LSTM(LSTM_num, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(ingredient_len, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model
```


```python
early_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience = 5,
    restore_best_weights=True
)
```


```python
unique_words = list(set([x for sublist in cleaned_sequences for x in sublist]))
```


```python
def train_model(LSTM_num, 
                 LSTM_layers, 
                 predictors,label,
                 unique_words,
                 max_length, 
                 recipe_embeddings, 
                 weight_name,
                 epochs=5, 
                 mask_zero=True,
                 device='/gpu:0',
                 transfer=False,
                 lr = 0.001,
                 callbacks=False):
    '''Trains model with defined architecture'''
    '''Trains and fine-tunes model using cpu'''
    with tf.device(device): #run on cpu because gpu has glitch with mask_zero=True
        model = create_model(LSTM_num,
                             LSTM_layers,
                             unique_ingredients=unique_words, 
                             max_length=max_length, 
                             recipe_embeddings=recipe_embeddings,
                             mask_zero=mask_zero,
                             transfer=transfer,
                             lr=lr)
        if transfer:
            model.load_weights('training_'+weight_name+'.h5')
    
        #model.summary()
        if callbacks:
            history = model.fit(predictors, label, validation_split=.1,epochs=epochs, verbose = 1,callbacks=[early_cb])
        else:
            history = model.fit(predictors, label, validation_split=.1,epochs=epochs, verbose = 1)
        return model, history
```


```python
#architecture grid search
nums = [20, 100, 200, 500, 1000]
num_history = []
legend_label = []
for num in nums:
    for layer in range(1,4):
        #print(' ')
        #print(f'LSTM number = {num}')
        #print(f'LSTM layers = {layer}')
        legend_label.append(f'Layers: {layer}, LSTMs: {num}')
        _, history = train_model(num,layer,predictors, label,unique_ingredients,max_length_, recipe_embeddings, 'num_weights',epochs=5, mask_zero=False)
        num_history.append(history)
```

<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    def plot_history(history, legend, show_train=True, show_valid=True):
        '''Plots single history'''
        val_loss = history.history['val_loss']
        loss = history.history['loss']
        epochs_train = len(loss)
        plt.style.use('ggplot')
        #plt.figure(figsize=(8, 4))   
        if show_train:
            plt.plot(range(1,epochs_train+1), pd.DataFrame(loss), label=('Training - '+str(legend)))
        if show_valid:
            plt.plot(range(1,epochs_train+1), pd.DataFrame(val_loss), label=('Validation - '+str(legend)))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    </pre>
    <pre>
    #show training results from different architectures
    plt.figure(figsize=(9,8))
    for n,history in enumerate(num_history):
        plt.subplot(2,1,1)
        plt.title('Validation')
        plot_history(history,legend_label[n], show_train=False)
        plt.legend(bbox_to_anchor=(1,1))
        plt.subplot(2,1,2)
        plt.title('Training')
        plot_history(history,legend_label[n], show_valid=False)
        plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    </pre>
</details>
    
![png](/assets/img/ingredient_predict/output_44_0.png) 

<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    final_val_loss = pd.DataFrame(np.zeros(len(num_history)))
    final_loss = pd.DataFrame(np.zeros(len(num_history)))

    for n,history in enumerate(num_history):
        final_val_loss.iloc[n-1] = history.history['val_loss'][-1]
        final_loss.iloc[n-1] = history.history['loss'][-1]
        

    final_scores = pd.concat([final_val_loss,final_loss,pd.Series(legend_label)],axis=1).set_axis(['Validation Loss','Loss','Label'],axis=1).set_index('Label',drop=True)

    final_scores.plot(kind='bar')
    final_scores.sort_values('Validation Loss').head()
    plt.xlabel('')
    plt.ylabel('Loss');
    </pre>
</details>

![png](/assets/img/ingredient_predict/output_46_0.png) 
        
<h2><br></h2>
<h2> Training ingredient generator </h2> 

```python
LSTM_ = 500
layers_ = 1
```


```python
def train_and_ft(predictors,label,unique_ingredients,max_length, recipe_embeddings,weight_name,callbacks,epochs=50):
    '''Trains and fine tunes LSTM model'''
    #train LSTM and dense layer
    model, history = train_model(LSTM_, layers_, 
                                predictors,
                                label,
                                unique_ingredients,
                                max_length,
                                recipe_embeddings,
                                weight_name,
                                mask_zero=True,
                                device='/cpu:0',
                                epochs=epochs,
                                callbacks=callbacks)
    model.save_weights('training_'+weight_name+'.h5')
    
    #fine tune all layers
    model, fine_history = train_model(LSTM_, layers_, 
                                predictors,
                                label,
                                unique_ingredients,
                                max_length,
                                recipe_embeddings,
                                weight_name,
                                mask_zero=True,
                                device='/cpu:0',
                                transfer=True,
                                lr = 0.0001,
                                epochs=epochs,
                                callbacks=callbacks)
    model.save_weights('fine_'+weight_name+'.h5')
    return model, history, fine_history
```


```python
model, history, fine_history = train_and_ft(predictors,label,unique_ingredients,max_length_,recipe_embeddings,'weight',callbacks=True)
```

<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    def get_history(history_, fine_history_):
        epochs_train = len(history_.history['loss'])
        history_.history['loss'].extend(fine_history_.history['loss'])
        history_.history['val_loss'].extend(fine_history_.history['val_loss'])
        epochs_ft = len(history_.history['loss'])
        return epochs_train, epochs_ft, history_
    </pre>
    <pre>
    epochs_train, epochs_ft, merged_history = get_history(history, fine_history)
    plot_history(merged_history,'Ingredients')
    plt.axvline(x=epochs_train,color='k')
    plt.legend()
    plt.text(4,5.45,'Training')
    plt.text(12,4.85,'Fine tuning');
    </pre>
</details>

![png](/assets/img/ingredient_predict/output_54_0.png) 
    

<h2><br></h2>
<h2>Building and training Seq2seq model</h2>

```python
max_length_name_ = max_length_name_
latent_dim = 128
```


```python
def build_seq2seq(embed_train=False, 
                max_length_=max_length_, 
                unique_ingredients=unique_ingredients,
                embed_size=embed_size,
                recipe_embeddings=recipe_embeddings,
                latent_dim=latent_dim,
                unique_names=unique_names,
                name_embeddings=name_embeddings,
                max_length_name_ = max_length_name_,
                summary=True
                ):
    '''builds seq2seq network'''
    encoder_inputs = Input(shape=(max_length_+1,),dtype='int32')
    encoder_embedding = Embedding(len(unique_ingredients)+1, embed_size, weights=[recipe_embeddings], trainable=embed_train, mask_zero=True, input_length=max_length_+1)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_length_name_,),dtype='int32')
    decoder_embedding = Embedding(len(unique_names)+1, embed_size, weights=[name_embeddings], trainable=embed_train, mask_zero=True, input_length=max_length_name_-1)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs,_,_ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(len(unique_names)+1,activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model_names = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    if summary:
        model_names.summary()
    return encoder_inputs,encoder_outputs, encoder_states,decoder_inputs,decoder_lstm,decoder_outputs, decoder_dense, model_names
```


```python
iterated_recipes_, name_sequences_, name_labels_ = shuffle([recipe_sequences, cut_name, label_name])
```


```python
early_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=0,
    restore_best_weights=True)

```


```python
# test_range of items
def train_seq2seq(test_range=None,epochs=500, save=True):
    #build, compile and train model using cpu due to issue with masking
    with tf.device('/cpu:0'):    
        #train LSTM components
        print('Model summary:')
        _,_,_,_,_,_,_,model_names = build_seq2seq()
        model_names.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        
        print('Training LSTMs...')
        model_names.load_weights('name_weights.h5')
        name_history = model_names.fit([iterated_recipes_[:test_range],name_sequences_[:test_range]], name_labels_[:test_range], 
                    batch_size=256, 
                    epochs=epochs, 
                    validation_split=0.1,
                    callbacks = [early_cb])
        name_weights =  model_names.save_weights('name_weights.h5')
        
        #fine tune model
        encoder_inputs,encoder_outputs, encoder_states,decoder_inputs,decoder_lstm,decoder_outputs, decoder_dense,model_names = build_seq2seq(embed_train=True, summary=False)
        
        model_names.compile(tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        
        model_names.load_weights('name_weights.h5')
        
        print('Fine tuning model...')
        fine_history = model_names.fit([iterated_recipes_[:test_range],name_sequences_[:test_range]], name_labels_[:test_range], 
                    batch_size=256, 
                    epochs=epochs, 
                    validation_split=0.1,
                    callbacks = [early_cb])
        
        name_weights =  model_names.save_weights('name_weights.h5')
        
        if save: 
            with open('name_training.pkl','wb') as f:
                pickle.dump([name_history, fine_history, model_names, name_weights, encoder_inputs,encoder_outputs, encoder_states,decoder_inputs,decoder_lstm,decoder_outputs, decoder_dense,model_names],f)
            f.close()
        
        return name_history, fine_history, model_names, name_weights, encoder_inputs,encoder_outputs, encoder_states,decoder_inputs,decoder_lstm,decoder_outputs, decoder_dense,model_names
```


```python
name_history, fine_history, model_names, name_weights, encoder_inputs,encoder_outputs, encoder_states,decoder_inputs,decoder_lstm,decoder_outputs, decoder_dense,model_names = train_seq2seq()

```

    Model summary:
    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     input_2 (InputLayer)        [(None, 45)]                 0         []                            
                                                                                                      
     input_3 (InputLayer)        [(None, 17)]                 0         []                            
                                                                                                      
     embedding_1 (Embedding)     (None, 45, 300)              3723900   ['input_2[0][0]']             
                                                                                                      
     embedding_2 (Embedding)     (None, 17, 300)              8686200   ['input_3[0][0]']             
                                                                                                      
     lstm_1 (LSTM)               [(None, 128),                219648    ['embedding_1[0][0]']         
                                  (None, 128),                                                        
                                  (None, 128)]                                                        
                                                                                                      
     lstm_2 (LSTM)               [(None, 17, 128),            219648    ['embedding_2[0][0]',         
                                  (None, 128),                           'lstm_1[0][1]',              
                                  (None, 128)]                           'lstm_1[0][2]']              
                                                                                                      
     dense_1 (Dense)             (None, 17, 28954)            3735066   ['lstm_2[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 16584462 (63.26 MB)
    Trainable params: 4174362 (15.92 MB)
    Non-trainable params: 12410100 (47.34 MB)
    __________________________________________________________________________________________________
    Training LSTMs...
    Fine tuning model...
```

<details>
    <summary>Click to show hidden code.</summary>
    <pre>
    name_loss = name_history.history['loss'][:471]
    name_val_loss = name_history.history['val_loss'][:471]
    name_accuracy = name_history.history['accuracy'][:471]
    name_val_accuracy = name_history.history['val_accuracy'][:471]
    </pre>
    <pre>
    plt.plot(range(len(name_loss)),name_loss,label='Training loss')
    plt.plot(range(len(name_val_loss)),name_val_loss,label='Validation loss')
    plt.axvline(x=149,color='k')
    plt.legend()
    plt.text(20,5.45,'Training')
    plt.text(170,4.25,'Fine tuning')
    plt.xlabel('Epochs')
    plt.ylabel('Loss');
    </pre>
</details>

![png](/assets/img/ingredient_predict/output_63_0.png) 

<h2><br></h2>
<h2>Predicting recipe names</h2>


```python
def decoder(latent_dim, embed_size=embed_size, unique_names=unique_names,name_embeddings=name_embeddings,max_length_name_=max_length_name_, decoder_inputs=decoder_inputs, decoder_lstm=decoder_lstm, decoder_dense=decoder_dense):
    '''Creates decoder model'''
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embedding = Embedding(len(unique_names)+1, embed_size, weights=[name_embeddings], trainable=False, mask_zero=True, input_length=max_length_name_)(decoder_inputs)
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    return decoder_model
```


```python
def encoder(encoder_inputs, encoder_states):
    '''Creates encoder model'''
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model
```


```python
def sample(preds, temperature=0.5):
    '''Selects next word in names with temperature controlling the variability.'''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```


```python
def decode_sequence(input_seq):
    '''Decodes context vector'''
    #load models 
    encoder_model = encoder(encoder_inputs, encoder_states)
    decoder_model = decoder(latent_dim)
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq,verbose=0)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,17))
    target_seq[0,-1] = 1 #Assigning first word to be 'sos'
    stop_condition = True
    decoded_sentence = ''
    prev_char = ' '
    while stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value,verbose=0)# Sample a token
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_token_index = np.random.choice(len(output_tokens[0,-1,:]),p=output_tokens[0,-1,:])
        sampled_token_index = sample(output_tokens[0,-1,:])
        sampled_char = list(tokenizer_names.word_index.keys())[sampled_token_index-1]
        if (sampled_char != prev_char)&(sampled_char != 'eos'):
            decoded_sentence += sampled_char+' '# Exit condition: either hit max length
        prev_char = sampled_char
        # or find stop character.

        if (sampled_char == 'eos' or len(decoded_sentence) > 100):
            stop_condition = False# Update the target sequence (of length 1).
            
        #updating target sequence
        target_seq = target_seq[0].tolist()
        target_seq.append(sampled_token_index)
        target_seq.pop(0)
        target_seq = np.array([target_seq])
        
        states_value = [h, c]
    return decoded_sentence.strip().title()
```





<h2><br></h2>
<h2>Predicting recipes</h2>


```python
def pad_sequences(token_list, max_length=max_length_):
    '''pad sequences (built in function returns a recursion error)'''
    padded_sequence = []
    zeros = (max_length-len(token_list))
    padded = [int(0) for zero in range(zeros+1)]
    padded.extend(token_list)
    return padded
```




```python
def cook_for_me(seed_text, model):
    '''Suggests additional ingredients to add'''
    next_words = 5
    words_out = []
    token_list = []

    seed_text = seed_text.split('and')
    for word in seed_text:
        word = word.strip()
        token_ = tokenizer.texts_to_sequences([word])[0]
        token_list.extend(token_)
    token_list = pad_sequences(token_list)[2:]

    while next_words > 0:
        model.load_weights('fine_weight.h5')
        proba = model.predict(np.array(token_list).reshape(-1,43),verbose=0)
        predicted = np.argmax(proba, axis=1)
        new_token = int(predicted[0])
        word = list(tokenizer.word_index.keys())[int(predicted[0]-1)]
        words_out.append(word)
        token_list.append(new_token)
        token_list.pop(0)
        next_words -= 1
    token_list = [pad_sequences(token_list)]
    words_out = list(set(words_out))
    string_out = 'Why not try adding some '+', '.join(words_out[:-1])+' and '+words_out[-1]+'?'
    return token_list, string_out
```


```python
def make_suggestions(ingredients_in_stock):
    '''Suggests additional ingredients to use and names the recipe'''
    for ingredient in ingredients_in_stock:
        greeting = 'For '+ingredient.upper() +' let me see... '
        print(greeting)
        suggestions, text = cook_for_me(ingredient,model)
        #print(suggestions)
        print(text.replace('flmy','flour'))
        print('I call it...')
        named = decode_sequence(suggestions)
        print(named)
        print(' ')
```
<h2><br></h2>
<h2>What's for dinner?</h2>

```python
#defining three levels to test suggestion models
#easy
single_ingredients = [
    'cherry',
    'lettuce',
    'mirin',
    'pineapple'
    ]

#medium
reasonable_combos = [
    'chicken and rice',
    'chicken and rice and salsa',
    'banana and cinnamon',
    'egg and sugar'
    ]

#hard
odd_pairings = [
    'chicken and rice and grapefruit',
    'chocolate and beans',
    'marshmallow and cantaloupe',
    'spinach and cinnamon'
    ]
```


```python
make_suggestions(single_ingredients)
```

    For CHERRY let me see... 
    Why not try adding some cinnamon, egg, sugar, flour and butter?
    I call it...
    Cherry Cobbler
     
    For LETTUCE let me see... 
    Why not try adding some cucumber, tomato, feta cheese, green pepper and scallion?
    I call it...
    Simple Salad
     
    For MIRIN let me see... 
    Why not try adding some sesame oil, garlic clove, soy sauce, fresh ginger and sugar?
    I call it...
    Japanese Dipping
     
    For PINEAPPLE let me see... 
    Why not try adding some lemon juice, salt, water, sugar and cornstarch?
    I call it...
    Pineapple Jam
     





```python
make_suggestions(reasonable_combos)
```

    For CHICKEN AND RICE let me see... 
    Why not try adding some chicken broth, carrot, celery, onion and potato?
    I call it...
    Chicken And Vegetables
     
    For CHICKEN AND RICE AND SALSA let me see... 
    Why not try adding some cheese, salt, tortilla, water and pepper?
    I call it...
    Chicken Tortilla
     
    For BANANA AND CINNAMON let me see... 
    Why not try adding some salt, allspice, ginger, nutmeg and clove?
    I call it...
    Banana Shake
     
    For EGG AND SUGAR let me see... 
    Why not try adding some vanilla, baking powder, salt, milk and flour?
    I call it...
    Funnel Cake
     



```python
make_suggestions(odd_pairings)
```

    For CHICKEN AND RICE AND GRAPEFRUIT let me see... 
    Why not try adding some garlic, water, soy sauce, ginger and brown sugar?
    I call it...
    Teriyaki Chicken
     
    For CHOCOLATE AND BEANS let me see... 
    Why not try adding some baking powder, sugar, egg, butter and flour?
    I call it...
    Chocolate Brownies
     
    For MARSHMALLOW AND CANTALOUPE let me see... 
    Why not try adding some blueberry, strawberry, pineapple and banana?
    I call it...
    Banana Fruit
     
    For SPINACH AND CINNAMON let me see... 
    Why not try adding some salt, milk, pepper, nutmeg and egg?
    I call it...
    Spinach And Cheese
     

