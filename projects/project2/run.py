import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

IMAGE_SIDE = 400


from utils import *
from losses import *
from models import *
from model_wrapper import *
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

params = {}
params['input_side'] = 128
params['image_side'] = IMAGE_SIDE
params['num_blocks'] = 5
params['activation_'] = 'elu'
params['regularizer_'] = None
params['starting_num_channels'] = 16
params['metrics'] = [F1_score]
params['metrics_names'] = ['loss', 'F1_score']
params['batch_size'] = 16
params['batches_per_epoch'] = 200
params['reg_name'] = None
params['lambd'] = 0
params['loss'] = soft_dice_loss
params['main_metric_fn'] = F1_score 
params['optimizer'] = tf.keras.optimizers.Adam()
params['path'] = '/saved_models/best' 

model_best = get_Unet_model(params)
model_best = ModelWrapper(model_best, params)

model_best.load_model()

test_data = load_test_data()
#Causion: without GPU acceleration prediction will run for a long time (maybe an hour)
#To accelerate prediction, remove parameter shift. But then output will be slightly different
#to what we submitted
preds = model_best.predict(test_data, shift=8)
save_preds(np.round(preds))