import tensorflow as tf
from tensorflow import keras

# from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Process, Queue
import pandas as pd
import cv2
import numpy as np
dataPath = 'F:/FlyAI/FlyAI_dataset/MRISegmentation/'
batchsize = 4
volSize = 256

def train(dataQueue):
    # train_datagen = ImageDataGenerator(featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     validation_split=0.1,
    #     rescale=1/255.0)
    # train_generator = train_datagen.flow_from_dataframe()
    base_model = keras.applications.VGG16(input_shape=(volSize,volSize,3), include_top=False)

    input = base_model.input
    x1 = base_model.get_layer("block1_conv2").output
    x1pool = base_model.get_layer("block1_pool").output
    x2 = base_model.get_layer("block2_conv2").output
    x2pool = base_model.get_layer("block2_pool").output
    x3 = base_model.get_layer("block3_conv3").output
    x3pool = base_model.get_layer("block3_pool").output
    x4 = base_model.get_layer("block4_conv3").output
    x4pool = base_model.get_layer("block4_pool").output
    x5 = base_model.get_layer("block5_conv3").output

    x6tmp = keras.layers.Concatenate()([x4pool, x5])
    x6tmp = keras.layers.Convolution2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2),padding='same')(x6tmp)
    x6tmp = keras.layers.Activation("relu")(x6tmp)
    x6 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same')(x6tmp))
    x6 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same')(x6))
    x6 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same')(x6))

    x7tmp = keras.layers.Concatenate()([x3pool, x6])
    x7tmp = keras.layers.Convolution2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2))(x7tmp)
    x7tmp = keras.layers.Activation("relu")(x7tmp)
    x7 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same')(x7tmp))
    x7 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same')(x7))
    x7 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same')(x7))

    x8tmp = keras.layers.Activation("relu")(keras.layers.Convolution2DTranspose(filters=128,kernel_size=(2,2),strides=(2,2),padding='same')(keras.layers.Concatenate()([x2pool,x7])))
    x8 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same')(x8tmp))
    x8 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same')(x8))

    x9tmp = keras.layers.Activation("relu")(keras.layers.Convolution2DTranspose(filters=64,kernel_size=(2,2),strides=(2,2),padding='same')(keras.layers.Concatenate()([x1pool,x8])))
    x9 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same')(x9tmp))
    x9 = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same')(x9))

    x10tmp = keras.layers.Activation("relu")(keras.layers.Conv2D(filters=2,kernel_size=(3,3),padding='same')(keras.layers.Concatenate()([input,x9])))
    x10sfm = keras.layers.Softmax()(keras.layers.Conv2D(filters=2,kernel_size=(1,1),padding='same')(x10tmp))

    model = tf.keras.Model(inputs=input, outputs=x10sfm)

    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
    model.load_weights('model')
    count = 0
    while count < 2000001:
        count += 1
        [imgMat, lblMat, imgKey, flag] = dataQueue.get()
        if flag == 1:
            model.fit(imgMat,lblMat)
        else:
            print(model.evaluate(imgMat,lblMat))
            model.save('model')

def prepareDataThread(dataQueue):
    trainInfo = pd.read_csv(dataPath + '/train/train.csv', sep=',', low_memory=False)

    trainDataList = [dataPath+'train/'+itm for itm in trainInfo['image_path']]
    trainLabelList = [dataPath+'train/'+itm for itm in trainInfo['label_path']]

    order = np.random.randint(0,len(trainDataList),1000000)
    imgMats = np.zeros(shape=[batchsize, volSize, volSize, 3], dtype=np.float32)
    lblMats = np.zeros(shape=[batchsize, volSize, volSize, 2], dtype=np.int32)
    bi = 0
    for ind in order:
        img = cv2.imread(trainDataList[ind])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        max_pix = np.max(img)
        img = img / max_pix

        img_mask = cv2.imread(trainLabelList[ind])
        label_1 = img_mask[:, :, 0] / 255.0
        if np.sum(label_1) <= 4:
            continue
        label_0 = 1 - label_1
        label = np.zeros((img_mask.shape[0], img_mask.shape[1], 2), dtype=np.int)
        label[:, :, 0] = label_0
        label[:, :, 1] = label_1

        # cv2.imshow('img',img)
        # cv2.imshow('label', label_1)
        # cv2.waitKey()

        imgMats[bi,:,:,:] = img
        lblMats[bi, :, :, :] = label
        if bi == batchsize-1:
            bi = 0
            dataQueue.put(tuple((imgMats, lblMats, trainDataList[ind], 1)))
        else:
            bi += 1

def prepareDataThreadVali(dataQueue):
    testInfo = pd.read_csv(dataPath+'/test/test.csv',sep=',',low_memory=False)

    testDataList = [dataPath+'test/'+itm for itm in testInfo['image_path']]
    testLabelList = [dataPath+'test/'+itm for itm in testInfo['label_path']]

    order = np.random.randint(0,len(testDataList),1000000)
    imgMats = np.zeros(shape=[batchsize, volSize, volSize, 3], dtype=np.float32)
    lblMats = np.zeros(shape=[batchsize, volSize, volSize, 2], dtype=np.int32)
    bi = 0
    for ind in order:
        img = cv2.imread(testDataList[ind])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        max_pix = np.max(img)
        img = img / max_pix

        img_mask = cv2.imread(testLabelList[ind])
        label_1 = img_mask[:, :, 0] / 255.0
        if np.sum(label_1) <= 4:
            continue
        label_0 = 1 - label_1
        label = np.zeros((img_mask.shape[0], img_mask.shape[1], 2), dtype=np.int)
        label[:, :, 0] = label_0
        label[:, :, 1] = label_1

        # cv2.imshow('img',img)
        # cv2.imshow('label', label_1)
        # cv2.waitKey()

        imgMats[bi,:,:,:] = img
        lblMats[bi, :, :, :] = label
        if bi == batchsize-1:
            bi = 0
            if np.random.random() < 0.01:
                dataQueue.put(tuple((imgMats, lblMats, testDataList[ind], 0)))
        else:
            bi += 1

if __name__ == '__main__':

    dataQueue = Queue(50)  # max 50 images in queue
    dataPreparation = [None] *2
    # print('params.params[ModelParams][nProc]: ', dataProdProcNum)
    # for proc in range(0, 1):
        # dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
    dataPreparation[0] = Process(target=prepareDataThread, args=(dataQueue,))
    dataPreparation[0].daemon = True
    dataPreparation[0].start()
    dataPreparation[1] = Process(target=prepareDataThreadVali, args=(dataQueue,))
    dataPreparation[1].daemon = True
    dataPreparation[1].start()
    # while True:
    #     tt=1
    train(dataQueue)
