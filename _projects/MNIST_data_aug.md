---
layout: page
title: Handwritten digit classification with 99.6% accuracy using a convolutional neural network
description: A small convolutional neural network is built to decode the MNIST 784 dataset. The final model was trained in ~30 minutes and achieved 99.6% accuracy. A smart data augmentation strategy is the key to the model's high accuracy. 
img: assets/img/mnist_data_aug/output_10_0.png
importance: 4
category: fun
---

# Handwritten digit classification with 99.6% accuracy using a convolutional neural network

A small convolutional neural network is built to decode the MNIST 784 dataset. This dataset contains a subset of 70,000 size-normalized and centered hand written digits. The architecture of the network was built up iteratively until apparent improvements to performance stagnated. Then, the model was optimized by evaluating data augmentation strategies, optimizers and dropout layers. The final model was trained in ~30 minutes and achieved 99.6% accuracy on the test set. The high level of accuracy is due to careful data augmentation during preprocessing.


```python
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from collections import Counter
import pandas as pd
from functools import partial 
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
```
<h3><br></h3>
<h3>Importing data</h3>


```python
#Loading MNIST_784 dataset from OpenML
mnist = fetch_openml('mnist_784', as_frame = False, parser='auto') 
X,y = mnist.data.reshape(-1,28,28,1)/255, mnist.target.reshape(-1,1)
y = y.astype(np.int64)
mnist.DESCR
```




    "**Author**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges  \n**Source**: [MNIST Website](http://yann.lecun.com/exdb/mnist/) - Date unknown  \n**Please cite**:  \n\nThe MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples  \n\nIt is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.  \n\nWith some classification methods (particularly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. If you do this kind of pre-processing, you should report it in your publications. The MNIST database was constructed from NIST's NIST originally designated SD-3 as their training set and SD-1 as their test set. However, SD-3 is much cleaner and easier to recognize than SD-1. The reason for this can be found on the fact that SD-3 was collected among Census Bureau employees, while SD-1 was collected among high-school students. Drawing sensible conclusions from learning experiments requires that the result be independent of the choice of training set and test among the complete set of samples. Therefore it was necessary to build a new database by mixing NIST's datasets.  \n\nThe MNIST training set is composed of 30,000 patterns from SD-3 and 30,000 patterns from SD-1. Our test set was composed of 5,000 patterns from SD-3 and 5,000 patterns from SD-1. The 60,000 pattern training set contained examples from approximately 250 writers. We made sure that the sets of writers of the training set and test set were disjoint. SD-1 contains 58,527 digit images written by 500 different writers. In contrast to SD-3, where blocks of data from each writer appeared in sequence, the data in SD-1 is scrambled. Writer identities for SD-1 is available and we used this information to unscramble the writers. We then split SD-1 in two: characters written by the first 250 writers went into our new training set. The remaining 250 writers were placed in our test set. Thus we had two sets with nearly 30,000 examples each. The new training set was completed with enough examples from SD-3, starting at pattern # 0, to make a full set of 60,000 training patterns. Similarly, the new test set was completed with SD-3 examples starting at pattern # 35,000 to make a full set with 60,000 test patterns. Only a subset of 10,000 test images (5,000 from SD-1 and 5,000 from SD-3) is available on this site. The full 60,000 sample training set is available.\n\nDownloaded from openml.org."




```python
#Making test and train sets
X_train, y_train, X_valid, y_valid, X_test, y_test = X[:60000], y[:60000],X[60000:65000],y[60000:65000],X[65000:],y[65000:]
```


```python
X_train.shape, X_valid.shape, X_test.shape
```




    ((60000, 28, 28, 1), (5000, 28, 28, 1), (5000, 28, 28, 1))




```python
y_train.shape, y_valid.shape, y_test.shape
```




    ((60000, 1), (5000, 1), (5000, 1))




```python
X.shape,X[0].shape
```




    ((70000, 28, 28, 1), (28, 28, 1))


<h3><br></h3>
<h3>EDA</h3>

Examples (25) of handwritten digits with their labels are shown below. Note the variation in slanting and character orientation apparent in different handwriting styles.


```python
#Looking at 10 random digits
def show_num(input_pic):
  plt.imshow(input_pic,cmap='binary')
  plt.axis(False)

for i in range(0,25):
    ax = plt.subplot(5,5,i+1)
    show_num(X[i])
    plt.title(str(y[i].squeeze()))
plt.tight_layout()
```


    
![data preview](/assets/img/mnist_data_aug/output_10_0.png)
    

The distribution of digits is balanced with all digits occurring at approximately the same frequency.


```python
pd.DataFrame(Counter(y.squeeze()).items(),columns=['Digit','Count']).sort_values('Digit').set_index('Digit').plot(kind='bar',ylabel='Count',legend=None, figsize=(7,3))
plt.xticks(rotation=0);
```
![number distribution](/assets/img/mnist_data_aug/output_12_0.png)

<h3><br></h3>
<h3>Data Augmentation</h3>

Data augmentation is a powerful method to boost model performance. In data augmentation, variation or noise is added to the training set. By training with this more complicated artificial dataset, neural networks can become better at evaluating test cases. To enhance performance, the added variation needs to be consistent with what is encountered in the test set. For example, if digits aren't rotated 180 degrees in the test set, rotating digits 180 in the training set may not improve performance.

In our dataset, we see slight variations in digit orientation and slanting. These factors are amplified in our test set by artificially shifting and rotating our test set digits. If features are overly shifted or rotated, it may increase the bias of our model and deviate from what is observed in our test set. For this reason, the levels of shifting and rotation were tuned in our model optimization.



```python
#image rotation (mimicks variation in handwriting slant)
img_rotation = tf.keras.layers.RandomRotation(
    factor = (-0.05,0.05),
    fill_mode = 'nearest',
    seed = 10
)

for i in range(0,9):
    aug_img = img_rotation(X[i])
    ax = plt.subplot(3,3,i+1)
    show_num(aug_img)
    plt.title(str(y[i].squeeze()))
```

![rotated data](/assets/img/mnist_data_aug/output_14_0.png)




```python
#image shifting - mimicks sligh variation in image centering 
img_shift = tf.keras.layers.RandomTranslation(
    height_factor = 0.1,
    width_factor = 0.1,
    fill_mode = 'nearest',
    seed = 10
)

for i in range(0,9):
    aug_img = img_shift(X[i])
    ax = plt.subplot(3,3,i+1)
    show_num(aug_img)
    plt.title(str(y[i]))
```

![shifted data](/assets/img/mnist_data_aug/output_15_0.png)  


```python
#combining both augmentations 
for i in range(0,9):
    aug_img = img_shift(img_rotation(X[i]))
    ax = plt.subplot(3,3,i+1)
    show_num(aug_img)
    plt.title(str(y[i]))
```


![augmented data](/assets/img/mnist_data_aug/output_16_0.png)  
    

<h3><br></h3>
<h3>Model building</h3>

A small convolutional neural network was built to decode our digits. The input image is first rotated and shifted as described previously. The convolutional layers are configured with padding so that dimensionality is not reduced during convolution, use ReLU activation to introduce non-linearity, and are initialized using the He normal initialization method. The number of filters used in each convolution step is increased from 64 to 256 and the filter size is decreased from 7 to 3. 

Max pooling layers are added to extract the most prominent features from the input while decreasing the computational complexity of the model. The maximum value of each sliding window is retained in a reduced dimensional space output.

The output of the convolutional layers is flattened and fed into three fully connected layers using ReLU activation. Dropout is also added as a regularization technique, which is discussed further in the optimization section.

The final architecture of this model is the result of iterative trial and error. Layers were added following commonly observed patterns and tweaked until the model trained quickly and with a high level of accuracy (~0.9) after the first epoch.


```python
Layer = partial(tf.keras.layers.Conv2D, kernel_size=3, padding='same', activation ='relu', kernel_initializer='he_normal')

model = tf.keras.Sequential([
    tf.keras.Input(shape=[28,28,1]),
    img_rotation,
    img_shift,
    Layer(filters=64, kernel_size=7),
    tf.keras.layers.MaxPool2D(),
    Layer(filters=128),
    Layer(filters=128),
    tf.keras.layers.MaxPool2D(),
    Layer(filters=256),
    Layer(filters=256),  
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=10,activation='softmax',kernel_initializer='he_normal')
])

model.compile(loss='sparse_categorical_crossentropy',
               optimizer = 'adam',
               metrics = ['accuracy'])

model.summary()
```
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     random_rotation (RandomRota  (None, 28, 28, 1)        0         
     tion)                                                           
                                                                     
     random_translation (RandomT  (None, 28, 28, 1)        0         
     ranslation)                                                     
                                                                     
     conv2d (Conv2D)             (None, 28, 28, 64)        3200      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 14, 14, 128)       73856     
                                                                     
     conv2d_2 (Conv2D)           (None, 14, 14, 128)       147584    
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 7, 7, 256)         295168    
                                                                     
     conv2d_4 (Conv2D)           (None, 7, 7, 256)         590080    
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 3, 3, 256)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 2304)              0         
                                                                     
     dense (Dense)               (None, 128)               295040    
                                                                     
     dropout (Dropout)           (None, 128)               0         
                                                                     
     dense_1 (Dense)             (None, 64)                8256      
                                                                     
     dropout_1 (Dropout)         (None, 64)                0         
                                                                     
     dense_2 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 1,413,834
    Trainable params: 1,413,834
    Non-trainable params: 0
    _________________________________________________________________
    
<h3><br></h3>
<h3>Model optimization</h3>

The model was optimized in three phases: 
1. Image augmentaiton 
2. Optimize selection 
3. Regularization with dropout


```python
%%time
history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid))
```

Images were first rotated and then shifted independently. The augmentation ranges were fine-tuned so as not to adversely affect model bias. It was found that rotations of 0.1 were too extreme but that 0.05 worked nicely. Similarly, an image shift of 0.1 performed well. The combination of these two augmentations outperformed the individual augmentations and notably decreased the variance of the model. The observed preservation of bias and reduction of variance is optimal for an image augmentation strategy and will be implemented going forward.  

| Test case | Training Accuracy | Validation Accuracy | Run Time | Epochs | Optimizer |
| --- | --- | --- | --- |--- |--- |
| No augmentation | .9884 | .9856 | 7:23 | 3 | sgd |
| Image rotation = .1 | .9781 | .9806 | 7:34 | 3 | sgd |
| Image rotation = .05 | .9828 | .9846 | 7:38 | 3 | sgd |
| Image shift = .1 | .9821 | .9832 | 7:38 | 3 | sgd |
| Image rotation = .05 + shift = .1 | .9757 | .9878 | 7:52 | 3 | sgd |

Three optimizers were evaluated by considering training set performance and run time. Overfitting the validation set was not considered, as this could be an indicator of an optimizer working well and can be combated with regularization techniques. Stochastic gradient descent (sgd) adjusts the network's weights by taking small steps in the direction of the steepest descent of the loss function. Adam and AdamW are variations of sgd that uses an adaptive learning rate based on first and second moments of the gradients. The incorporation of momentum helps accelerate convergence by adding a fraction of the previous gradients to the current updates step. The second moment of the gradient is used to dampen the gradient in the steepest direction, which provides robustness and functionality in noisy environments. AdamW additionally includes weight decay, which is a regularization technique that reduces the size of model's weights at each training iteration. 


Adam was selected for its superior training accuracy and low runtime and is used in all cases going forward.


| Test case | Training Accuracy | Validation Accuracy | Run Time | Epochs |
| --- | --- | --- | --- | --- |
| sgd | .9757 | .9878 | 7:52 | 3 |
| adam | .9834 | .9866 | 8:30 | 3 |
| adamW | .9828 | .9882 | 13:08 | 3 |

Dropout is a regularization technique that sets a fraction of the input units to zero during each training iteration. By randomly dropping neurons, dropout forces the network to rely on different subsets of neurons for each training example, which prevents overfitting. Dropout also functions as a form of ensemble learning built from training each varying subset of the network. 

We expect to see higher validation accuracy than training accuracy for each case due to differences in the training and test data, as image augmentation will make training predictions more challenging than test predictions. If dropout is not too disruptive, we should see a further improvement to the validation accuracy. For this reason, the validation accuracy will be compared to the validation accuracy of the network with no dropout layers. 

Two dropout rates of 0.2 and 0.5 were tested in the fully connected layers and in the convolutional layers. In the convolutional layers, incorporation of 0.5 dropout significantly increased model bias, and 0.2 dropout slightly increased bias, both without a significant reduction in model variance. Thus, dropout was not included in the convolutional layers. In the fully connected layers, both 0.2 and 0.5 dropout levels increased model bias, but 0.2 dropout model's validation accuracy outperformed the baseline case. Thus, layers of 0.2 dropout were included in the fully connected layers.


| Test case | Training Accuracy | Validation Accuracy | Epochs |
| --- | --- | --- |  --- | 
| No dropout | .9926 | .9878 |  20 |
| Dropout in fully connected (FC) layers (0.2) | .9900 | .9898 |  20 |
| Dropout in FC layers (0.5) | .9872 | .9846 |  20 | 
| Dropout in FC layers (0.2) + Conv2D layers (0.5) | .9484 | .9474 |  20 |
| Dropout in FC layers (0.2) + Conv2D layers (0.2) | .9818 | .9890 |  20 | 

<h3><br></h3>
<h3>Model training</h3>

The optimized model was trained using early stopping and evaluated on the test set.


```python
#defining callbacks - early stopping and weight saving

early_stopping = EarlyStopping(monitor='val_accuracy',patience=10, restore_best_weights=True)
save_weights = ModelCheckpoint('weights_MNIST784CV.h5', save_best_only=True, save_weights_only=True)
```


```python
%%time
#model.load_weights('weights_MNIST784CV.h5')
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid), callbacks=[early_stopping, save_weights])
```

    Epoch 1/200
    1875/1875 [==============================] - 186s 99ms/step - loss: 0.1814 - accuracy: 0.9507 - val_loss: 0.0842 - val_accuracy: 0.9766
    Epoch 2/200
    1875/1875 [==============================] - 190s 101ms/step - loss: 0.0881 - accuracy: 0.9773 - val_loss: 0.0691 - val_accuracy: 0.9806
    Epoch 3/200
    1875/1875 [==============================] - 177s 94ms/step - loss: 0.0744 - accuracy: 0.9811 - val_loss: 0.0431 - val_accuracy: 0.9886
    Epoch 4/200
    1875/1875 [==============================] - 169s 90ms/step - loss: 0.0639 - accuracy: 0.9835 - val_loss: 0.0583 - val_accuracy: 0.9840
    Epoch 5/200
    1875/1875 [==============================] - 169s 90ms/step - loss: 0.0583 - accuracy: 0.9852 - val_loss: 0.0361 - val_accuracy: 0.9902
    Epoch 6/200
    1875/1875 [==============================] - 170s 91ms/step - loss: 0.0546 - accuracy: 0.9858 - val_loss: 0.0422 - val_accuracy: 0.9876
    Epoch 7/200
    1875/1875 [==============================] - 172s 92ms/step - loss: 0.0488 - accuracy: 0.9874 - val_loss: 0.0318 - val_accuracy: 0.9916
    Epoch 8/200
    1875/1875 [==============================] - 175s 93ms/step - loss: 0.0489 - accuracy: 0.9882 - val_loss: 0.0545 - val_accuracy: 0.9872
    Epoch 9/200
    1875/1875 [==============================] - 169s 90ms/step - loss: 0.0478 - accuracy: 0.9882 - val_loss: 0.0565 - val_accuracy: 0.9846
    Epoch 10/200
    1875/1875 [==============================] - 167s 89ms/step - loss: 0.0452 - accuracy: 0.9884 - val_loss: 0.0283 - val_accuracy: 0.9928
    Epoch 11/200
    1875/1875 [==============================] - 172s 92ms/step - loss: 0.0449 - accuracy: 0.9889 - val_loss: 0.0260 - val_accuracy: 0.9944
    Epoch 12/200
    1875/1875 [==============================] - 170s 91ms/step - loss: 0.0470 - accuracy: 0.9887 - val_loss: 0.0464 - val_accuracy: 0.9872
    Epoch 13/200
    1875/1875 [==============================] - 176s 94ms/step - loss: 0.0445 - accuracy: 0.9885 - val_loss: 0.0443 - val_accuracy: 0.9890
    Epoch 14/200
    1875/1875 [==============================] - 174s 93ms/step - loss: 0.0407 - accuracy: 0.9898 - val_loss: 0.0442 - val_accuracy: 0.9902
    Epoch 15/200
    1875/1875 [==============================] - 171s 91ms/step - loss: 0.0428 - accuracy: 0.9899 - val_loss: 0.0333 - val_accuracy: 0.9906
    Epoch 16/200
    1875/1875 [==============================] - 173s 92ms/step - loss: 0.0391 - accuracy: 0.9903 - val_loss: 0.0387 - val_accuracy: 0.9900
    Epoch 17/200
    1875/1875 [==============================] - 171s 91ms/step - loss: 0.0441 - accuracy: 0.9895 - val_loss: 0.0434 - val_accuracy: 0.9898
    Epoch 18/200
    1875/1875 [==============================] - 176s 94ms/step - loss: 0.0380 - accuracy: 0.9904 - val_loss: 0.0471 - val_accuracy: 0.9874
    Epoch 19/200
    1875/1875 [==============================] - 173s 92ms/step - loss: 0.0411 - accuracy: 0.9897 - val_loss: 0.0662 - val_accuracy: 0.9860
    Epoch 20/200
    1875/1875 [==============================] - 179s 96ms/step - loss: 0.0395 - accuracy: 0.9901 - val_loss: 0.0362 - val_accuracy: 0.9908
    Epoch 21/200
    1875/1875 [==============================] - 173s 92ms/step - loss: 0.0374 - accuracy: 0.9908 - val_loss: 0.0505 - val_accuracy: 0.9896
    


```python
pd.DataFrame(history.history).plot(xlim=[0,29], ylim=[0,1], grid=True, xlabel='Epoch', ylabel='Metric',style=['r--','r--.',"b-",'b-*'])
plt.show()
```

![model history](/assets/img/mnist_data_aug/output_27_0.png)
    



```python
y_pred = model.predict(X_test).argmax(axis=1)
accuracy_score(y_test,y_pred)
```

    157/157 [==============================] - 5s 35ms/step
    




    0.9956



The model accuracy on the test set was 0.9956.

<h3><br></h3>
<h3>Error analysis</h3>

In order to asses the model's performance and determine possible pathways for model improvement, we look at the performance of the model. All of the mislabeled digits are displayed with the correct label in the figure below.


```python
print('Total number of mislabeled instances in test dataset:')
print((y_test.squeeze() != y_pred).sum())
```

    Total number of mislabeled instances in test dataset:
    22
    


```python
for i in range(0,22):
    ax = plt.subplot(5,5,i+1)
    show_num(X_test[y_test.squeeze() != y_pred][i])
    plt.title('Actual: '+ str(y_test[y_test.squeeze() != y_pred][i].squeeze()) +'\nPredicted: '+str(y_pred[y_test.squeeze() != y_pred][i]))
plt.tight_layout()
```


![misclassified numbers](/assets/img/mnist_data_aug/output_32_0.png)    
    


Most of the misclassifications are reasonable. For example, classifying 5's with closed bottom curves as 6's. The high frequency of digits that appear to be faded in the misclassified set suggest that further image augmentation mimicking this fade may be helpful to future classification efforts. 

The frequency of misclassifications is shown in the confusion matrix below. It is noted that the most common misclassification was overly classifying 7's as 2's. The misclassified 7's are shown for reference. For future model development, it may be of interest to obtain more 7's and 2's, especially 7's with a bottom sherif.


```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x17c9e1a29d0>




![confusion matrix](/assets/img/mnist_data_aug/output_34_1.png)   
    



```python
for i in range(0,5):
    ax = plt.subplot(1,5,i+1)
    show_num(X_test[(y_test.squeeze()==7)&(y_pred==2)][i])
    plt.title('Acutal: '+ str(y_test[(y_test.squeeze()==7)&(y_pred==2)][i].squeeze()) +'\nPredicted: '+str(y_pred[(y_test.squeeze()==7)&(y_pred==2)][i]))
plt.tight_layout()
```


![Misclassified sevens](/assets/img/mnist_data_aug/output_35_0.png)    


<h3><br></h3>
<h3>Benchmarking model performance</h3>

LeChunn has compiled a summary of model performance on the MNIST dataset. The results for convolutional neural networks are shown in the table below. In the first column there is a description of the network, the second column describes preprocessing steps, the third collumn is the test error rate (%), and the fourth column is the reference for the model. 

![CV_perf_MNIST784.png](attachment:CV_perf_MNIST784.png)

Full table available at http://yann.lecun.com/exdb/mnist/

The error rate is 1 - the accuracy rate, and our model has an error rate percentage of 0.44, which is in line or better than the scores on these larger convenolutional networks. Likely, the preprocessing steps added to our model provide a singificant boost to model performance.
