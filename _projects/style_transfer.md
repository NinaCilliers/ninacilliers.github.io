---
layout: page
title: Painting with convolutional neural networks
description: A novel image is generated with distinct user selected style and content.
img: /assets/img/style_transfer/output_44_0.png
importance: 1
category: Other projects
---

<h1>Painting with convolutional neural networks</h1>
<h2>Introduction</h2>

In this project we generate a new image from two distinct content and style images using a pre-trained VGG neural network. Instead of training the neural network weights, as is typical in a deep learning project, the generated image is adjusted to minimize a custom cost function. This project is inspired by "Deep Learning & Art: Neural Style Transfer" by Andrew Ng and follows the 2015 paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Leon Gatys et al. 

<b>What are content and style in a neural network?</b><br>
Each layer of the VGG neural network produces a differently filtered version of the input image. High levels of a neural network capture objects and their spatial arrangement, while lower levels capture smaller scale features. While it may feel intuitive to use only the lower levels of a neural network to capture style, style is apparent in all scales of artistic representation. Thus, input from all layers is used to represent the style of an image, while input from one higher level layer is used to represent content. 

The key idea in "A Neural Algorithm of Artistic Style" is the separation of style and content inputs. The figure below shows the content and style of select layers in a VGG neural network. Style is represented as a gram matrix. <br>


![Overview](/assets/img/style_transfer/intro_fig.png)<br>
[Gatys et al. 2015](https://arxiv.org/abs/1508.06576) 
<br>
<br>
<b> What is a Gram matrix? </b><br>
The Gram matrix has seen wide spread utilization in style representation. The mechanism underlying its utility is thought to be a consequence of minimizing the maximum mean discrepancy between images (See [Li et al. 2017](https://arxiv.org/pdf/1701.01036.pdf) for full discussion). The Gram matrix is the product of multiplying a matrix with its own transpose and provides a measure of the degree of feature correlation between the matrix vectors. This transformation results in high values for similar styles and ignores feature position. 

F<sup>l</sup><sub>ij</sub> is the activation of the ith filter at position j in layer l. The Gram matrix G<sup>l</sup><sub>ij</sub> is the product of matrix multiplication of the vectorized feature map i and j in each layer l: <br>

![Gram](/assets/img/style_transfer/gram.png)<br>



```python
import os
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import keras 
from PIL import Image
import copy
import tensorflow as tf
import numpy as np
```
<h2><br></h2>
<h2>Content Image</h2>


```python
#content image

#Defining function to look at a digit
def show_num(input_pic):
  plt.imshow(input_pic,cmap='binary')
  plt.axis(False)

content_image = Image.open('tokyo.jpg').resize((300,300))
print(np.array(content_image).shape)
print('Content Image:')
content_image
#[Helena Bradbury](https://www.helenabradbury.com/blog-1/72-hours-in-tokyo-japan)
```

    (300, 300, 3)
    Content Image:
    




    
![content image](/assets/img/style_transfer/output_4_1.png)




```python
#content image formatting 
print('Content image formatting:')
print(f'Original size: {content_image.size}')

#converting to tensor
content_image = np.array(content_image)/255
content_image = content_image[None,:,:,:]
print((f'Tensor size: {content_image.shape}'))
```

    Content image formatting:
    Original size: (300, 300)
    Tensor size: (1, 300, 300, 3)
    
<h2><br></h2>
<h2>Style Image</h2>


```python
style_image = Image.open('pablo.jpg')

print('Style image formatting:')
print(f'Original size: {style_image.size}')

#resizing, preserving aspect ratio
style_image = style_image.resize((300,int(300*818/650)))
#print(f'Size after resizing: {style_image.size}')

#cropping
style_image = style_image.crop((0,10,300,310)) #left, upper, right, lower
print(f'Size after cropping: {style_image.size}')

#converting to tensor
style_image = np.array(style_image)/255
style_image = style_image[None,:,:,:]
print((f'Tensor size: {style_image.shape}'))
```

    Style image formatting:
    Original size: (650, 818)
    Size after cropping: (300, 300)
    Tensor size: (1, 300, 300, 3)
    


```python
#visualizing adjusted style image 

def show_tensor(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8) 
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
show_tensor(style_image)
```


![Style image](/assets/img/style_transfer/output_8_0.png)<br>
[Portrait of Ambroise Vollard, Pablo Picasso](https://www.theguardian.com/artanddesign/2016/may/05/spiritualist-artist-georgiana-houghton-uk-exhibition-courtauld)


<h2><br></h2>
<h2>Loading VGG-19</h2>

The layers of a pre-trained VGG-19 neural network are used to compute a custom loss function. VGG-19 is a deep convolutional network with 19 convolutional layers. In theory, any trained convolutional neural network could be used for this task. We use VGG-19 and replace the max pooling layers with average pooling layers following [Gatys et al. 2015](https://arxiv.org/abs/1508.06576).



```python
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(content_image.shape[1], content_image.shape[2], 3),
                                  weights='imagenet')
vgg.trainable = False

for layer in vgg.layers:
    print(layer.name, '         ',layer)
```

    input_28           <keras.engine.input_layer.InputLayer object at 0x000001E1217D0070>
    block1_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D2350>
    block1_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D0CA0>
    block1_pool           <keras.layers.pooling.max_pooling2d.MaxPooling2D object at 0x000001E1142D4CA0>
    block2_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E12552F4C0>
    block2_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E12495C4C0>
    block2_pool           <keras.layers.pooling.max_pooling2d.MaxPooling2D object at 0x000001E1217E3250>
    block3_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D3970>
    block3_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E126BD1750>
    block3_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10959BDF0>
    block3_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E124C82170>
    block3_pool           <keras.layers.pooling.max_pooling2d.MaxPooling2D object at 0x000001E1095991B0>
    block4_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217E1AE0>
    block4_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217E01C0>
    block4_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1058E06D0>
    block4_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E124E1ACE0>
    block4_pool           <keras.layers.pooling.max_pooling2d.MaxPooling2D object at 0x000001E109599ED0>
    block5_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E114605780>
    block5_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E109598970>
    block5_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10E0B5480>
    block5_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10E0B75B0>
    block5_pool           <keras.layers.pooling.max_pooling2d.MaxPooling2D object at 0x000001E10E0B6D10>
    


```python
#switching max pool layers to avg pool layers 

#max pool settings
poolsize = (2,2)
padding = 'valid'
strides = (2,2)
data_format = 'channels_last'


#finding MaxPool layers and switching to Average Pool
#x = vgg.layers[0].output
input = vgg.layers[0].input
x = vgg.layers[1](input)
for layer in vgg.layers[2:]:
    if (isinstance(layer, keras.layers.pooling.max_pooling2d.MaxPool2D)):
        x = tf.keras.layers.AveragePooling2D(pool_size=poolsize, padding=padding, strides=strides, data_format=data_format)(x)
        #x = tf.keras.layers.AveragePooling2D()(x)
    else: x = layer(x)
vgg = tf.keras.Model(input, x)

#checking that this has indeed happened 
for layer in vgg.layers:
    print(layer.name, '         ',layer)

```

    input_28           <keras.engine.input_layer.InputLayer object at 0x000001E1217D0070>
    block1_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D2350>
    block1_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D0CA0>
    average_pooling2d_76           <keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x000001E11429B6D0>
    block2_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E12552F4C0>
    block2_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E12495C4C0>
    average_pooling2d_77           <keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x000001E10959B340>
    block3_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217D3970>
    block3_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E126BD1750>
    block3_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10959BDF0>
    block3_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E124C82170>
    average_pooling2d_78           <keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x000001E124A0AE60>
    block4_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217E1AE0>
    block4_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1217E01C0>
    block4_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E1058E06D0>
    block4_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E124E1ACE0>
    average_pooling2d_79           <keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x000001E1217D32E0>
    block5_conv1           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E114605780>
    block5_conv2           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E109598970>
    block5_conv3           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10E0B5480>
    block5_conv4           <keras.layers.convolutional.conv2d.Conv2D object at 0x000001E10E0B75B0>
    average_pooling2d_80           <keras.layers.pooling.average_pooling2d.AveragePooling2D object at 0x000001E1217E1750>
    
<h2><br></h2>
<h2>Cost and loss functions</h2>

Cost functions are built for each style layers and the content layer. These functions are combined into one loss function following [Gatys et al. 2015](https://arxiv.org/abs/1508.06576).<br>

The content cost function compares the activation of a single selected layer F<sup>l</sup><sub>ij</sub> to the feature response of the original image P<sup>l</sup><sub>ij</sub>, where the vectors x and p are the generated and original images:

![content cost](/assets/img/style_transfer/content_loss.png)<br>

The style cost function compares the Gram matrix of a single selected layer G<sup>l</sup><sub>ij</sub> to the Gram matrix of the style image A<sup>l</sup><sub>ij</sub>, where N<sub>i</sub> and M<sub>i</sub> are the number and size of the feature maps. The final style cost function is a weighted sum of each layer’s cost function. 

![style cost](/assets/img/style_transfer/style-cost.png)<br>



```python
# compute content cost using a_C and a_G
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    #a_G dimensions
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    #reshape
    a_C_unrolled = tf.reshape(a_C, shape = [_,-1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [_, -1, n_C])
    
    #cost 
    J_content = tf.reduce_sum((a_C-a_G)**2/(4*n_H*n_W*n_C))
    
    return J_content
```


```python
def compute_layer_style_cost(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape()
    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H*n_W,n_C]))
    
    GS = tf.matmul(a_S, tf.transpose(a_S))
    GG = tf.matmul(a_G, tf.transpose(a_G))
    
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*n_C**2*((n_H*n_W)**2))
    
    return J_style_layer
```


```python
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    J_style = 0
    
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]
    
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer

    return J_style
```

The loss function is a combination of the loss and cost functions, where alpha and beta represent the relative contributions of content and style. In this project we optimize the ratio of alpha to beta. <br>

![loss function](/assets/img/style_transfer/loss-function.png)<br>

We use gradient tape in tensorflow to compute this custom loss function and update the generated image. The weights of the VGG network are unchanged.


```python
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]
```


```python
content_layer = [('block5_conv1', 1)]
```


```python
def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model
```


```python
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder
```


```python
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)
```


```python
#alpha_beta is the #ratio of alpha (content) to beta (style)
@tf.function()
def total_cost(J_content, J_style, alpha_beta):
    J = tf.math.multiply(alpha_beta,J_content)+J_style
    return J
```


```python
learning_rate = 0.01
content_layer = [('block3_conv1', 1)]
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

@tf.function()
def train_step(generated_image, alpha_beta):

    with tf.GradientTape() as tape:
        a_C = vgg_model_outputs(preprocessed_content)
        a_G = vgg_model_outputs(generated_image)

        J_style = compute_style_cost(a_S, a_G)
        J_content = compute_content_cost(a_C, a_G)

        J = total_cost(J_content, J_style, alpha_beta)
          
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    
    generated_image.assign(clip_me(generated_image))   
    return J
```
<h2><br></h2>
<h2>Initial generated image</h2>
We generate an initial image by adding a low level of noise to our content image. 


```python
def clip_me(image):
    return tf.clip_by_value(image, 0.0, 1.0)
```


```python
#initialize image to be geenrated 
#generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(content_image),-.1,.1)
generated_image = content_image + noise
generated_image = clip_me(generated_image)
generated_image = tf.Variable(generated_image)
print(f'Checking size of generated image: {generated_image.get_shape()}')

starting_point = copy.deepcopy(generated_image)
image_check = show_tensor(generated_image)
display(image_check)
```

    Checking size of generated image: (1, 300, 300, 3)
    

![initial image](/assets/img/style_transfer/output_26_1.png)
    

<h2><br></h2>
<h2>Hyperparameter tunning</h2>
We carefully tune selected hyperparameters to optimize our final generated image: <br><br>
- Style to cost ratio (alpha to beta)<br>
- Content layer representation<br>
- Learning rate <br>

<h3><br></h3>
<h3>Style to cost ratio</h3>



```python
def train(epochs, alpha_beta, progress):
    #defines new starting point for each training session
    #to pick up from last genrated image, comment out
    generated_image = copy.deepcopy(starting_point)
    #display(show_tensor(generated_image))

    epoch_show = 2*int(epochs/10)
    img_log = []
    epoch_list = []

    alpha_beta = np.array(alpha_beta,dtype=np.float32)

    if progress:
        print('Training...')
        
    for i in range(epochs+1):
        train_step(generated_image, alpha_beta)
        if i % epoch_show == 0:
            if progress:
                print(f'Epoch {i} - Done')
            img = show_tensor(generated_image)
            #display(img.resize((100,100)))
            img_log.append(generated_image)
            epoch_list.append(i)
    
    if progress:
        #display in grid
        print('Image training progression:')
        for i, image in enumerate(img_log):
            plt.subplot(int(len(img_log)/int(len(img_log)/2)), int(len(img_log)/2), i + 1)
            plt.imshow(show_tensor(image))
            plt.title(str(epoch_list[i]))
            plt.axis('off')
        plt.show()

    #return image from last traiing epoch
    return(generated_image)
```


```python
alpha_beta_range = np.array([1e-5, 1e-4, 1e-3, 1e-2, .1,1,10],dtype=np.float32)
epochs = 10
optimizer.learning_rate = 0.01
image_log = []
content_layer = [('block5_conv1', 1)]

for ratio in alpha_beta_range: 
    print('Training alpha to beta ratio',str(ratio),'...')  
    #print('Starting point')
    #display(show_tensor(starting_point).resize(((100,100))))  
    img = train(epochs, ratio, progress=False)
    image_log.append(img)

```

    Training alpha to beta ratio 1e-05 ...
    Training alpha to beta ratio 1e-04 ...
    Training alpha to beta ratio 0.001 ...
    Training alpha to beta ratio 0.01 ...
    Training alpha to beta ratio 0.1 ...
    Training alpha to beta ratio 1.0 ...
    Training alpha to beta ratio 10.0 ...
    


```python
#results of ratio fit 
plt.figure(figsize=(10,8))
rows = int(np.ceil(len(image_log)/2))
plt.subplot(2,rows,1)
plt.title('Starting point')
plt.imshow(show_tensor(starting_point))
plt.axis('off')

for i, image in enumerate(image_log):
    plt.subplot(2,rows,i+2)
    plt.title(alpha_beta_range[i])
    plt.imshow(show_tensor(image))
    plt.axis('off')

plt.tight_layout()
```


![alpha beta](/assets/img/style_transfer/output_30_0.png)
    


An alpha to beta ratio of 1 is selected.


```python
alpha_beta = 1
```
<h3><br></h3>
<h3>Content layer selection</h3>


```python
CONTENT_LAYERS = [
    ('block1_conv1', 1),
    ('block2_conv1', 1),
    ('block3_conv1', 1),
    ('block4_conv1', 1),
    ('block5_conv1', 1)]
```


```python
layer_log = []
epochs = 10
optimizer.learning_rate = .01

for i,layer in enumerate(CONTENT_LAYERS):
    content_layer = [CONTENT_LAYERS[i]]
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
    #print(vgg_model_outputs.outputs[-1])
    print('Training',str(content_layer[0]).title()[2:14],'...')
    img = train(epochs, alpha_beta, progress=False)
    layer_log.append(img)
```

    Training Block1_Conv1 ...
    Training Block2_Conv1 ...
    Training Block3_Conv1 ...
    Training Block4_Conv1 ...
    Training Block5_Conv1 ...
    


```python
#results of ratio fit 
plt.figure(figsize=(10,8))
plt.subplot(2,int((len(layer_log)+1)/2),1)
plt.title('Starting point')
plt.imshow(show_tensor(starting_point))
plt.axis('off')

for i, image in enumerate(layer_log):
    plt.subplot(2,int((len(layer_log)+1)/2),i+2)
    plt.title(str(CONTENT_LAYERS[i][0].title()[:6]))
    plt.imshow(show_tensor(image))
    plt.axis('off')

plt.tight_layout()
```


![content layers](/assets/img/style_transfer/output_36_0.png)<br>
    


Content layer selection has a subtle effect on the generated image. The highest layer sampled block5_conv1 is used in the spirit of capturing the highest level content features.

<h3><br></h3>
<h3>Learning rate</h3>


```python
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + [('block5_conv1', 1)])
optimizer.learning_rate = 0.01
epochs = 10
lr_log = []

learning_rates = [0.001, 0.01, 0.025, 0.05, 0.1]
for rate in learning_rates: 
    print('Training',rate,'...')
    optimizer.learning_rate = rate
    img = train(epochs, alpha_beta, progress=False)
    lr_log.append(img)
```

    Training 0.001 ...
    Training 0.01 ...
    Training 0.025 ...
    Training 0.05 ...
    Training 0.1 ...
    


```python
#results of ratio fit 
plt.figure(figsize=(10,8))
plt.subplot(2,int((len(lr_log)+1)/2),1)
plt.title('Starting point')
plt.imshow(show_tensor(content_image))
plt.axis('off')

for i, image in enumerate(lr_log):
    plt.subplot(2,int((len(lr_log)+1)/2),i+2)
    plt.title(learning_rates[i])
    plt.imshow(show_tensor(image))
    plt.axis('off')

plt.tight_layout()
```


![learning rates](/assets/img/style_transfer/output_40_0.png)<br>
    


Images generated with learning rates of 0.01 and 0.025 have the desirable combination of visable changes to style and clearly discernable buildings. We select 0.01 for use in our final image generation.

<h2><br></h2>
<h2>Final image generation</h2>


```python
optimizer.learning_rate = 0.01
epochs = 1000

img = train(epochs, alpha_beta, progress=True)
```

    Training...
    Epoch 0 - Done
    Epoch 200 - Done
    Epoch 400 - Done
    Epoch 600 - Done
    Epoch 800 - Done
    Epoch 1000 - Done
    Image training progression:
    

![training final image](/assets/img/style_transfer/outputs_43_1.png)   



```python
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.imshow(show_tensor(content_image))
plt.axis('off')
plt.title('Content Image')

plt.subplot(1,3,2)
plt.imshow(show_tensor(style_image))
plt.axis('off')
plt.title('Style Image')

plt.subplot(1,3,3)
plt.imshow(show_tensor(img))
plt.axis('off')
plt.title('Generated image')

plt.tight_layout()
```


![final image generation](/assets/img/style_transfer/output_44_0.png)

