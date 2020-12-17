import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ast
from PIL import Image

FILE_PATH = "."


class ModelWrapper():

    def __init__(self, model, params):
        '''

        Parameters
        ----------
        model : tf.keras.Model
        params : dict
            Must contain the following attributes:
                params['metrics_names']: list
                    names of metrics tracked by model
                params['input_side']: int
                    model's input must have (input_side, input_side, 3) shape
                params['image_side']: int
                    side of images to be received for training
                params['reg_name']: None or str from ['l1', 'l2']
                    name of regularizer. It is used only to track the regularization
                    penalty of the model
                params['lambd']: float
                    regularization parameter
                params['batch_size']: int
                params['batches_per_epoch']: int
                params['path']: string
                    It is used to save and download model's weights and
                    training characteristics from this path
                params['main_metric_fn']: function
                    metric for evaluating results on initial (not cropped) images. 
                    It is used to compare different models to save the best model's weights
        
        '''
        
        self.model = model
        self.metrics_names = params['metrics_names']
        self.input_side = params['input_side']
        self.image_side = params['image_side']
        self.reg_name = params['reg_name']
        self.lambd = params['lambd']
        self.batch_size = params['batch_size']
        self.batches_per_epoch = params['batches_per_epoch']
        self.path = params['path']
        self.main_metric_fn = params['main_metric_fn']
        self.main_metric_name = 'Full_img_' + self.main_metric_fn.__name__

        self.metrics_per_epoch = {}
        for metric in self.metrics_names:
            self.metrics_per_epoch[metric+'_train'] = []
        for metric in self.metrics_names:
            self.metrics_per_epoch[metric+'_test'] = []
        if self.reg_name != None:
            self.metrics_per_epoch['regularizer'] = []
        self.metrics_per_epoch[self.main_metric_name] = []

        self.epoch = 0
        self.best_main_metric_score = None
    

    def train(self, x_train, y_train, x_test=np.array([]), y_test=np.array([]), epochs=1, use_flip=True, use_rot90=True, rot_angle = None, shift=None, save=True):
        '''
        
        Parameters
        ----------
        use_flip: bool
            whether to use image flipping for training data augmentation
        
        use_rot90: bool
            whether to use image rotation by multiple of 90 degrees for 
            training data augmentation
        
        rot_angle: None or float
            If float, add random rotation from (-rot_angle, rot_angle) degrees to 
            training images
        
        shift : None or int
            when predictiong and evaluating model on initial images, smaller images 
            will be cropped from them with shifting size 'shift'. Smaller shift value 
            improves prediction results at the expence of more computational cost.
            If None, it is set to the biggest possible value (self.input_side) 

        '''
        
        use_test_data = (len(x_test) > 0) and (len(y_test) > 0)
        
        if shift == None:
            shift = self.input_side
        
        if use_test_data:
            x_test_cropped, y_test_cropped = self.transform_test_data(x_test, y_test, shift)
        
        for epoch in range(epochs):
            
            metrics_per_batch = {}
            for metric in self.metrics_names:
                metrics_per_batch[metric] = tf.keras.metrics.Mean()

            for _ in range(self.batches_per_epoch):
                x_batch, y_batch = self.get_batch(x_train, y_train, use_flip, use_rot90, rot_angle)
                res = self.model.train_on_batch(x_batch, y_batch)
                for i, metric in enumerate(self.metrics_names):
                    metrics_per_batch[metric](res[i])
            
            for metric in self.metrics_names:
                self.metrics_per_epoch[metric+'_train'].append(metrics_per_batch[metric].result().numpy())
            
            if use_test_data:
                res = self.model.evaluate(x_test_cropped, y_test_cropped, verbose=0)
                for i, metric in enumerate(self.metrics_names):
                    self.metrics_per_epoch[metric+'_test'].append(res[i])
            else:
                for i, metric in enumerate(self.metrics_names):
                    self.metrics_per_epoch[metric+'_test'].append(None)
            
            if self.reg_name != None:
                self.metrics_per_epoch['regularizer'].append(self.reg_score())
            
            if use_test_data:
                self.metrics_per_epoch[self.main_metric_name].append(self.evaluate(x_test, y_test, shift))
            
                #if current weights are the best, save them
                if (self.best_main_metric_score == None) or (self.metrics_per_epoch[self.main_metric_name][-1] > self.best_main_metric_score):
                    self.best_main_metric_score = self.metrics_per_epoch[self.main_metric_name][-1]
            else:
                self.metrics_per_epoch[self.main_metric_name].append(None)
            
            if save:
                self.save_model()
            
            self.epoch += 1
            self.print_state()
    
    
    def predict(self, images, shift=None):
        
        if shift == None:
            shift = self.input_side
        
        image_side = images.shape[1]
        
        pred = np.zeros([images.shape[0], image_side, image_side, 1])
        #count number of predictions for each pixel
        counter = np.zeros([images.shape[0], image_side, image_side, 1])
        
        for x0 in range(0, image_side, shift):
            for y0 in range(0, image_side, shift):
                #prevent going out of borders
                x = min(x0, image_side - self.input_side)
                y = min(y0, image_side - self.input_side)
                pred[:, x:x+self.input_side, y:y+self.input_side, :] += self.model.predict(images[:, x:x+self.input_side, y:y+self.input_side, :])
                counter[:, x:x+self.input_side, y:y+self.input_side, :] += 1
        
        return pred / counter
    
    
    def evaluate(self, images, y_true, shift=None):
             
        y_pred = self.predict(images, shift)
        scores = map(lambda true_pred: self.main_metric_fn(true_pred[0], true_pred[1]), zip(y_true, y_pred))
        scores = np.array(list(scores))
        return scores.mean()
    
    
    def transform_test_data(self, x_test, y_test, shift):
        
        s = self.input_side
        S = self.image_side
        out_x_test = [x_test[k,x:x+s,y:y+s,:] for x in range(0,S-s+1,shift) for y in range(0,S-s+1,shift) for k in range(x_test.shape[0])]
        out_y_test = [y_test[k,x:x+s,y:y+s,:] for x in range(0,S-s+1,shift) for y in range(0,S-s+1,shift) for k in range(y_test.shape[0])]
        return np.array(out_x_test), np.array(out_y_test)
    
    
    def get_batch(self, x_train, y_train, use_flip=True, use_rot90=True, rot_angle = None):
        
        ids = np.random.randint(0, x_train.shape[0], self.batch_size)
        x_train_selected = x_train[ids]
        y_train_selected = y_train[ids]

        if rot_angle != None:
            angles = np.random.uniform(-rot_angle, rot_angle, self.batch_size)
            x_train_selected = np.array(list(map(lambda img_angle: self.rotate_image(img_angle[0], img_angle[1]), zip(x_train_selected, angles))))
            y_train_selected = np.array(list(map(lambda img_angle: self.rotate_image(img_angle[0][:,:,0], img_angle[1]), zip(y_train_selected, angles))))
            y_train_selected = y_train_selected[..., np.newaxis]

        left_bounds = np.random.randint(0, self.image_side - self.input_side + 1, self.batch_size)
        upper_bounds = np.random.randint(0, self.image_side - self.input_side + 1, self.batch_size)
        x_batch = [img[x:x+self.input_side, y:y+self.input_side, :] for img, x, y in zip(x_train_selected, left_bounds, upper_bounds)]
        y_batch = [img[x:x+self.input_side, y:y+self.input_side, :] for img, x, y in zip(y_train_selected, left_bounds, upper_bounds)]
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        if use_flip:
            axis_to_flip = np.random.randint(0, 3, x_batch.shape[0]) #if 0, no flipping applied
            for k in range(1,3):
                x_batch[axis_to_flip == k] = np.flip(x_batch[axis_to_flip == k], axis=k)
                y_batch[axis_to_flip == k] = np.flip(y_batch[axis_to_flip == k], axis=k)

        if use_rot90:
            #times to rotate by 90 degrees counterclockwise
            times_to_rotate = np.random.randint(0, 4, x_batch.shape[0])
            for k in range(1,4):
                x_batch[times_to_rotate == k] = np.rot90(x_batch[times_to_rotate == k], k=k, axes=(1,2))
                y_batch[times_to_rotate == k] = np.rot90(y_batch[times_to_rotate == k], k=k, axes=(1,2))
        
        return x_batch, y_batch
    
    
    def rotate_image(self, image, angle):
        img = Image.fromarray((image * 255).astype(np.uint8))
        img = img.rotate(angle)
        img = np.array(img) / 255.
        return img
    
    
    def print_state(self):
        print('Epoch: {:04d}: '.format(self.epoch), end='')
        for metric in self.metrics_per_epoch.keys():
            if (len(self.metrics_per_epoch[metric]) > 0) and (self.metrics_per_epoch[metric][-1] != None):
                print('{}: {:.5f}, '.format(metric, self.metrics_per_epoch[metric][-1]), end='')
        print()
    
    
    def reg_score(self):
        
        if self.reg_name == None:
            return
        elif self.reg_name == 'l1':
            reg_fn = lambda x: np.sum(np.abs(x))
        elif self.reg_name == 'l2':
            reg_fn = lambda x: np.sum(np.square(x))
        else:
            raise Exception('Incorrect value of reg_name')
        
        res = 0.
        for layer in self.model.layers:
            if (layer.name.startswith('conv2d') or layer.name.startswith('conv2d_transpose')) and (not layer.name.endswith('last')):
                kernel_weights = layer.get_weights()[0]
                res += reg_fn(kernel_weights)
        return res * self.lambd
    

    def draw_metrics(self):
        
        fig, axes = plt.subplots(len(self.metrics_names), 2, sharex=True, figsize=(12, 6*len(self.metrics_names)))
        for i, metric in enumerate(self.metrics_names):

            axes[i,0].set_title('Train {} vs epochs'.format(metric))
            axes[i,0].set_xlabel("epochs", fontsize=14)
            axes[i,0].set_ylabel("loss", fontsize=14)
            axes[i,0].plot(self.metrics_per_epoch[metric+'_train'])

            axes[i,1].set_title('Test {} vs epochs'.format(metric))
            axes[i,1].set_xlabel("epochs", fontsize=14)
            axes[i,1].set_ylabel("loss", fontsize=14)
            axes[i,1].plot(self.metrics_per_epoch[metric+'_test'])
        
        plt.show()
    
    
    def save_model(self):

        #saving model's weights
        model_path = FILE_PATH + self.path + '/model'               
        self.model.save_weights(model_path)

        #saving history of model's metrics
        train_info_path = FILE_PATH + self.path + '/train_info.txt'
        f=open(train_info_path,'w')
        for metric in self.metrics_per_epoch.keys():
            f.write(str(self.metrics_per_epoch[metric]) + '\n')
        f.close()
    
    
    def load_model(self):

        #loading model's weights
        model_path = FILE_PATH + self.path + '/model'
        self.model.load_weights(model_path)

        #loading history of model's metrics
        train_info_path = FILE_PATH + self.path + '/train_info.txt'
        f = open(train_info_path,'r')
        train_info = f.read()
        train_info = train_info.split(sep='\n')
        for i, metric in enumerate(self.metrics_per_epoch.keys()):
            self.metrics_per_epoch[metric] = ast.literal_eval(train_info[i])
        f.close()
        self.epoch = len(self.metrics_per_epoch[self.main_metric_name])