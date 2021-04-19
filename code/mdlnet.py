'''
   Jackie Yuan
   2018.06
'''

from __future__ import print_function
from keras.callbacks import ModelCheckpoint
import sgydata
import myunet
import numpy as np

# input image dimensions
img_rows, img_cols = 1600, 200
img_rows2, img_cols2 = 200, 200
nchannels=3

# input data
#shot1
numsgy1,x_train1,y_train=sgydata.load_sgylist(sgylist='../config/train_marmo_pvel_syn_nt2000_ns_01_bk.txt', 
                                               floc='../config/train_marmo_label.txt',shuffle='false')
#shot2
numsgy2,x_train2,y_train=sgydata.load_sgylist(sgylist='../config/train_marmo_pvel_syn_nt2000_ns_03_bk.txt', 
                                               floc='../config/train_marmo_label.txt',shuffle='false')

#shot3
numsgy3,x_train3,y_train=sgydata.load_sgylist(sgylist='../config/train_marmo_pvel_syn_nt2000_ns_05_bk.txt', 
                                               floc='../config/train_marmo_label.txt',shuffle='false')

nums1,x_test1,y_test=sgydata.load_sgylist(sgylist='../config/test_marmo_pvel_syn_nt2000_ns_01_bk.txt', 
                                               floc='../config/test_marmo_label.txt')

nums2,x_test2,y_test=sgydata.load_sgylist(sgylist='../config/test_marmo_pvel_syn_nt2000_ns_03_bk.txt', 
                                               floc='../config/test_marmo_label.txt')

nums3,x_test3,y_test=sgydata_ycc.load_sgylist(sgylist='../config/test_marmo_pvel_syn_nt2000_ns_05_bk.txt', 
                                               floc='../config/test_marmo_label.txt')

# reshape training data
x_train1 = x_train1.reshape(x_train1.shape[0], img_cols, img_rows, 1)
x_train2 = x_train2.reshape(x_train2.shape[0], img_cols, img_rows, 1)
x_train3 = x_train3.reshape(x_train3.shape[0], img_cols, img_rows, 1)

y_train = y_train.reshape(y_train.shape[0], img_cols2, img_rows2, 1)

# reshape test data
x_test1 = x_test1.reshape(x_test1.shape[0], img_cols, img_rows, 1)
x_test2 = x_test2.reshape(x_test2.shape[0], img_cols, img_rows, 1)
x_test3 = x_test3.reshape(x_test3.shape[0], img_cols, img_rows, 1)

y_test = y_test.reshape(y_test.shape[0], img_cols2, img_rows2, 1)

# combine data
x_train = np.concatenate((x_train1,x_train2,x_train3),axis=3)
x_test = np.concatenate((x_test1,x_test2,x_test3),axis=3)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# design, train and test model
model = myunet.get_unet(img_cols, img_rows, nchannels)
print("got unet")

model_checkpoint = ModelCheckpoint('../results/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')
history_callback=model.fit(x_train, y_train, batch_size=2, nb_epoch=100, verbose=1,
                           validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])


# predict and output test data and image
str1='../results'
str3='/imgs_mask_test.npy'
str4="/loss_history_marmo_cnns.txt"
str5="/val_loss_history_marmo_cnns.txt"
str6="/mdlnet_marmo_cnns.h5";
print('predict test data')
imgs_mask_test = model.predict(x_test, batch_size=1, verbose=1)
np.save(str1+str3, imgs_mask_test)

# save model and img
print("array to image")
imgs = np.load(str1+str3)
for i in range(imgs.shape[0]):
    img = imgs[i]
    np.savetxt("../results/%d_gaussian_marmo_cnns.txt"%(i),img[...,0])
    
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]
numpy_loss_history = np.array(loss_history)
numpy_val_loss_history = np.array(val_loss_history)
np.savetxt(str1+str4, numpy_loss_history, delimiter=",")
np.savetxt(str1+str5, numpy_val_loss_history, delimiter=",")

print("save model")
model.save(str1+str6)  

