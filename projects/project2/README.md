# CS-433
This folder consists of implementation for project 2 of [**EPFL CS-433**](https://edu.epfl.ch/coursebook/en/machine-learning-CS-433) course.

# Introduction
Here you can find all the code for [**Class Project 2 | Road Segmentation**](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) competition (2020). The goal of the project was to create road segmentation algorithm for sattelite images. For the best performance of neural nettwork model we imlement data augmentation process and some losses to train with.

On this competition our best result was achieved with *accuracy=0.952* and *F1-score=0.909*.

**`Team`**: **NotTheLast**

**`TeamMembers`**: **Yaroslav Kivva**, **Denys Pushkin**, **Odysseas Drosis**

# Quickstart
Before running the code make sure that you download train and test datasets to the directory ```data/```.

To reproduce our best results please follow the instructions below:
1. Download train and test datasets to the ```data/```
2. ```python run.py```

## Implementation

### **`loss.py`**
Implementations of some custom losses\metrics for training the model
- ```def soft_dice_loss(y_true, y_pred, smooth = 1):```: soft dice loss
- ```def weighted_binary_crossentropy(y_true, y_pred, weight=4):```: weighted binary crossentropy loss
- ```def F1_score(y_true, y_pred, delta=1e-8):```: f1-score

### **`train.py`**
Implementation of training process. To train different models you should change parameters defined in the beginning of the file.

### **`models.py`**
Consists of implementation of models architecture
- ```def get_Unet_model(params):```: constructs UNet model with the given parameters. ```params``` - is a dict with the next keys:
  - ```num_blocks```: number of downsampling and upsampling blocks + 1
  - ```input_side```: size of the image for the input to the model
  - ```activation```: activation which used in conv layers
  - ```regularizer_```: regularizer which used in Conv layers
  - ```starting_num_channels```: number of channels in the input
  - ```optimizer```: optimizer to update weights
  - ```loss```: loss which used for the output
  - ```metrics```: metrics which will be used while training
- ```def get_model_4_blocks(optimizer, loss, metrics, input_side=INPUT_SIDE, base_activation='elu', dropout_rate = 0.2):``` - another implementation of UNet with num_blocks = 4
- ```def get_model_5_blocks(optimizer, loss, metrics, input_side=INPUT_SIDE, base_activation='elu', dropout_rate = 0.2):``` - another implementation of UNet with num_blocks = 5



### **`model_wrapper.py`**
Consistas the class for wrapping the model constructed from **model.py**, so you can perform training, evaluation, prediction, saving and loading the model:
- ```def __init__(self, model, params):``` : initialize an instance of ModelWrapper to perform training, evaluation and prediction with given parameters. ```paramms``` is dictionary with the next keys: 
  - ```input_side```: size of the side of the images to feed to the network
  - ```image_side```: size of the initial image
  - ```reg_name```: None or str from ['l1', 'l2'] name of regularizer. It is used only to track the regularization penalty of the model
  - ```lambd```: regularization parameter
  - ```batch_size```: batch size
  - ```path```: path to save and load the model
  - ```main_metric_fn```: function which used for metric
  - ```main_metric_name```: string which names this metric
- ``` train(self, x_train, y_train, x_test=np.array([]), y_test=np.array([]), epochs=1, use_flip=True, use_rot90=True, rot_angle = None, shift=None, save=True)```: performs training of the model with given parameters
- ```def predict(self, images, shift=None):``` : make prediction for the given images
  - ```shift```: could be None or int. Specifies how we slide the window when predicting the full_size image
- ```def evaluate(self, images, y_true, shift=None):``` : evaluates performance on images (use main metric function for generation the score)
- ```def get_batch(self, x_train, y_train, use_flip=True, use_rot90=True, rot_angle = None):``` : generated batches for the training
- ```def rotate_image(self, image, angle):``` : rotates image by angle
- ```def print_state(self):``` : outputs current epoch, metric and loss scores
- ```def reg_score(self):``` : custom loss function (adds regularization)
- ```def draw_metrics(self):``` : draw metric scores
- ```def save_model(self):``` : saves model weights
- ```def load_model(self):``` : loads the model weights

### **`main.ipynb`**
Notebook which we used for visualizing results, expolring training porcess


