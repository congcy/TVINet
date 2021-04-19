from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.preprocessing.image import img_to_array, array_to_img

def get_unet(img_cols, img_rows, nchannels):
    
    inputs = Input((img_cols, img_rows, nchannels))
    
    '''
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(10, 4))(conv1)
    print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(10, 4))(conv2)
    print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(20, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print ("conv4 shape:",conv4.shape)
    conv4 = Conv2D(20, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    print ("conv4 shape:",conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)
    print ("pool4 shape:",pool4.shape)

    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print ("conv5 shape:",conv5.shape)
    conv5 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    print ("conv5 shape:",conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(20, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(drop5))
#    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = concatenate([drop4,up6], axis = 3)
    print ("merge6 shape:",merge6.shape)
    conv6 = Conv2D(20, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    print ("conv6 shape:",conv6.shape)
    conv6 = Conv2D(20, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print ("conv6 shape:",conv6.shape)

    up7 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv6))
#    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = concatenate([conv3,up7], axis = 3)
    print ("merge7 shape:",merge7.shape)
    conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    print ("conv7 shape:",conv7.shape)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print ("conv7 shape:",conv7.shape)

    up8 = Conv2D(12, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (10,4))(conv7))
#    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = concatenate([conv2,up8], axis = 3)
    print ("merge8 shape:",merge8.shape)
    conv8 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    print ("conv8 shape:",conv8.shape)
    conv8 = Conv2D(12, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print ("conv8 shape:",conv8.shape)
    
    up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (10,4))(conv8))
#    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = concatenate([conv1,up9], axis = 3)
    print ("merge9 shape:",merge9.shape)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    '''
    '''
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)
    print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)
    print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(216, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(216, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print ("conv4 shape:",conv4.shape)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    print ("conv4 shape:",conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)
    print ("pool4 shape:",pool4.shape)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print ("conv5 shape:",conv5.shape)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    print ("conv5 shape:",conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(drop5))
#    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = concatenate([drop4,up6], axis = 3)
    print ("merge6 shape:",merge6.shape)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    print ("conv6 shape:",conv6.shape)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print ("conv6 shape:",conv6.shape)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv6))
#    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = concatenate([conv3,up7], axis = 3)
    print ("merge7 shape:",merge7.shape)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    print ("conv7 shape:",conv7.shape)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print ("conv7 shape:",conv7.shape)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv7))
#    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = concatenate([conv2,up8], axis = 3)
    print ("merge8 shape:",merge8.shape)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    print ("conv8 shape:",conv8.shape)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print ("conv8 shape:",conv8.shape)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv8))
#    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = concatenate([conv1,up9], axis = 3)
    print ("merge9 shape:",merge9.shape)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    '''
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print ("conv1 shape:",conv1.shape)
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print ("conv1 shape:",conv1.shape)
    pool1 = MaxPooling2D(pool_size=(5, 5))(conv1)
    print ("pool1 shape:",pool1.shape)

    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print ("conv2 shape:",conv2.shape)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print ("conv2 shape:",conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 5))(conv2)
    print ("pool2 shape:",pool2.shape)

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print ("conv3 shape:",conv3.shape)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print ("conv3 shape:",conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 4))(conv3)
    print ("pool3 shape:",pool3.shape)

    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    print ("conv4 shape:",conv4.shape)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    print ("conv4 shape:",conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(1, 2))(drop4)
    print ("pool4 shape:",pool4.shape)

    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    print ("conv5 shape:",conv5.shape)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    print ("conv5 shape:",conv5.shape)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(drop5))
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    print ("conv6 shape:",conv6.shape)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print ("conv6 shape:",conv6.shape)

    up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(conv6))
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    print ("conv7 shape:",conv7.shape)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print ("conv7 shape:",conv7.shape)

    up8 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,5))(conv7))
    conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    print ("conv8 shape:",conv8.shape)
    conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print ("conv8 shape:",conv8.shape)
    
    up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (5,5))(conv8))
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    print ("conv9 shape:",conv9.shape)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['accuracy'])
#    model.compile(optimizer = SGD(lr=0.0001, decay=0.0, momentum=0.99, clipvalue=0.01), loss = 'mse', metrics = ['accuracy'])
#    model.compile(optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model









