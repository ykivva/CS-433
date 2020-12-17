from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, MaxPool2D, Dropout

INPUT_SIDE = 128


def get_Unet_model(params):
    num_blocks = params['num_blocks']
    input_side = params['input_side']
    activation_ = params['activation_']
    regularizer_ = params['regularizer_']
    starting_num_channels = params['starting_num_channels']
    optimizer = params['optimizer']
    loss = params['loss']
    metrics = params['metrics']

    tensors = {}
    tensors['inputs'] = Input(shape=(input_side, input_side, 3), name='input')

    for encoder_block in range(num_blocks):
        
        last_output = None
        dropout_rate = 0.2
        if encoder_block == 0:
            last_output = 'inputs'
        else:
            last_output = f'pool_{encoder_block-1}'
        conv_cur = f'conv2d_{encoder_block}'
        channels = starting_num_channels * (2**(encoder_block))

        tensors[conv_cur] = Conv2D(channels, (3,3), strides = (1,1), padding='same', activation=activation_, 
                                   kernel_regularizer=regularizer_, name=f'conv2d_{2*encoder_block}')(tensors[last_output])
        tensors[conv_cur] = BatchNormalization()(tensors[conv_cur])
        tensors[conv_cur] = Dropout(dropout_rate)(tensors[conv_cur])
        tensors[conv_cur] = Conv2D(channels, (3,3), strides = (1,1), padding='same', activation=activation_, kernel_regularizer=regularizer_, 
                                   name=f'conv2d_{2*encoder_block+1}')(tensors[conv_cur])
        tensors[conv_cur] = BatchNormalization()(tensors[conv_cur])
        if encoder_block+1 != num_blocks:
            pool_cur = f'pool_{encoder_block}'
            tensors[pool_cur] = MaxPool2D((2, 2), strides=(2,2))(tensors[conv_cur])
        
    for decoder_block in range(num_blocks-1):

        last_output = f'conv2d_{num_blocks+decoder_block-1}'
        dropout_rate = 0.2
        conv_cur = f'conv2d_{num_blocks+decoder_block}'
        conv_to_concat = f'conv2d_{num_blocks-decoder_block-2}'
        channels = starting_num_channels * (2**(num_blocks-decoder_block-2))
        num_conv2d_blocks = 2*(num_blocks+decoder_block)
        
        tensors[conv_cur] = Conv2DTranspose(channels, (3,3), strides=(2,2), padding='same', activation='linear', 
                                            kernel_regularizer=regularizer_, name=f'conv2d_transpose_{decoder_block}')(tensors[last_output])
        tensors[conv_cur] = Concatenate(axis=-1)([tensors[conv_cur], tensors[conv_to_concat]])
        tensors[conv_cur] = Conv2D(channels, (3,3), padding='same', activation=activation_, 
                                   kernel_regularizer=regularizer_, name=f'conv2d_{num_conv2d_blocks}')(tensors[conv_cur])
        tensors[conv_cur] = BatchNormalization()(tensors[conv_cur])
        tensors[conv_cur] = Dropout(dropout_rate)(tensors[conv_cur])
        tensors[conv_cur] = Conv2D(channels, (3,3), padding='same', activation=activation_, kernel_regularizer=regularizer_, 
                                   name=f'conv2d_{num_conv2d_blocks+1}')(tensors[conv_cur])
        tensors[conv_cur] = BatchNormalization()(tensors[conv_cur])
    #last ecoder block
    last_output = f'conv2d_{2*num_blocks-2}'
    tensors['outputs'] = Conv2D(1, (3,3), padding='same', activation='sigmoid', name='conv2d_last')(tensors[last_output])

    model = Model(inputs=tensors['inputs'], outputs=tensors['outputs'])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_model_4_blocks(optimizer, loss, metrics, input_side=INPUT_SIDE, base_activation='elu', 
              dropout_rate = 0.2):

    inputs = Input(shape=(INPUT_SIDE, INPUT_SIDE, 3))
    conv1 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D((2, 2), strides=(2,2))(conv1)
    
    conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D((2, 2), strides=(2,2))(conv2)
    
    conv3 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(dropout_rate)(conv3)
    conv3 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D((2, 2), strides=(2,2))(conv3)
    
    conv4 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(dropout_rate)(conv4)
    conv4 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv4)
    conv4 = BatchNormalization()(conv4)
    
    upsample5 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='linear')(conv4)
    upsample5 = Concatenate(axis=-1)([upsample5, conv3])
    conv5 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(dropout_rate)(conv5)
    conv5 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    upsample6 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='linear')(conv5)
    upsample6 = Concatenate(axis=-1)([upsample6, conv2])
    conv6 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
    conv6 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    upsample7 = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', activation='linear')(conv6)
    upsample7 = Concatenate(axis=-1)([upsample7, conv1])
    conv7 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(dropout_rate)(conv7)
    conv7 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    outputs = Conv2D(1, (3,3), strides=(1,1), padding='same', activation='sigmoid', name='conv2d_last')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model


def get_model_5_blocks(optimizer, loss, metrics, input_side=INPUT_SIDE, base_activation='elu', 
              dropout_rate = 0.2):

    inputs = Input(shape=(INPUT_SIDE, INPUT_SIDE, 3))
    conv1 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    conv1 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D((2, 2), strides=(2,2))(conv1)
    
    conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    conv2 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D((2, 2), strides=(2,2))(conv2)
    
    conv3 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(dropout_rate)(conv3)
    conv3 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D((2, 2), strides=(2,2))(conv3)
    
    conv4 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(dropout_rate)(conv4)
    conv4 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPool2D((2, 2), strides=(2,2))(conv4)
    
    conv5 = Conv2D(256, (3,3), strides=(1,1), padding='same', activation=base_activation)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(dropout_rate)(conv5)
    conv5 = Conv2D(256, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv5)
    conv5 = BatchNormalization()(conv5)
    
    upsample6 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='linear')(conv5)
    upsample6 = Concatenate(axis=-1)([upsample6, conv4])
    conv6 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
    conv6 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv6)
    conv6 = BatchNormalization()(conv6)
    
    upsample7 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='linear')(conv6)
    upsample7 = Concatenate(axis=-1)([upsample7, conv3])
    conv7 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(dropout_rate)(conv7)
    conv7 = Conv2D(64, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv7)
    conv7 = BatchNormalization()(conv7)
    
    upsample8 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='linear')(conv7)
    upsample8 = Concatenate(axis=-1)([upsample8, conv2])
    conv8 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(dropout_rate)(conv8)
    conv8 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv8)
    conv8 = BatchNormalization()(conv8)
    
    upsample9 = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', activation='linear')(conv8)
    upsample9 = Concatenate(axis=-1)([upsample9, conv1])
    conv9 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(upsample9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(dropout_rate)(conv9)
    conv9 = Conv2D(16, (3,3), strides=(1,1), padding='same', activation=base_activation)(conv9)
    conv9 = BatchNormalization()(conv9)
    
    outputs = Conv2D(1, (3,3), strides=(1,1), padding='same', activation='sigmoid', name='conv2d_last')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model