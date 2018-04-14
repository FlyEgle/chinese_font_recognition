# coding=utf-8
"""
keras model for training hanzi
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Activation, Embedding
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, PReLU, Lambda
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

code_path = '/media/wislab/DataSet/jiang/FaceDataSet/'


# bn + prelu
def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


# build pipline model
def build_model(out_dims, input_shape=(128, 128, 1)):
    inputs_dim = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs_dim)
    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x_flat = Flatten()(x)

    fc1 = Dense(512)(x_flat)
    fc1 = bn_prelu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model


# build a model of softmax+0.01centerloss
def build_centerloss_model(out_dims, feat_dims, input_shape=(128, 128, 1), lambda_center=0.01):
    """
    isCenterloss
    """
    inputs_dim = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs_dim)
    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x_flat = Flatten()(x)

    fc1 = Dense(feat_dims)(x_flat)
    fc1 = bn_prelu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    base_model = Model(inputs=inputs_dim, outputs=fc2)

    # center loss
    lambda_c = lambda_center
    input_target = Input(shape=(1,))
    centers = Embedding(out_dims, feat_dims)(input_target)
    l2_losss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([fc1, centers])
    model_centers = Model(inputs=[base_model.input, input_target], outputs=[x, l2_losss])

    return model_centers


# build a resnet50 model from imagenet weights
def resnet50_100(feat_dims, out_dims):
    # resnett50 only have a input_shape=(128, 128, 3), if use resnet we must change
    # shape at least shape=(197, 197, 1)
    resnet_base_model = ResNet50(include_top=False, weights=None, input_shape=(128, 128, 1))

    # get output of original resnet50
    x = resnet_base_model.get_layer('avg_pool').output
    x = Flatten()(x)
    fc = Dense(feat_dims)(x)
    x = bn_prelu(fc)
    x = Dropout(0.5)(x)
    x = Dense(out_dims)(x)
    x = Activation("softmax")(x)

    # buid myself model
    input_shape = resnet_base_model.input
    output_shape = x

    resnet50_100_model = Model(inputs=input_shape, outputs=output_shape)

    return resnet50_100_model


# learning rate of epoch
def lrschedule(epoch):
    if epoch <= 40:
        return 0.1
    elif epoch <= 80:
        return 0.01
    else:
        return 0.001


# one-hot 2 label
def translate_onehot2label(one_hot):
    # length = num of images labels, nb_classes = classes of image
    length = one_hot.shape[0]
    nb_classes = one_hot.shape[1]

    labels = []
    for i in range(length):
        for j in range(nb_classes):
            if one_hot[i][j] == 1:
                labels.append(j)

    labels = np.array(labels).reshape((length, 1))

    return labels


# my generator for centerloss
def mygenerator(generator):
    """
    :param generator:
    :return: x: [x, y_value], y: [y, random_centers]
    """

    while True:
        data = next(generator)
        x, y = data[0], data[1]
        # not one-hot encoding
        y_value = translate_onehot2label(y)

        random_centers = np.random.randn(BATCH_SIZE, 1)

        data_x = [x, y_value]
        data_y = [y, random_centers]
        yield data_x, data_y


# training model
def model_train(model, loadweights, isCenterloss, lambda_center):
    lr = LearningRateScheduler(lrschedule)
    mdcheck = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', save_best_only=True)
    td = TensorBoard(log_dir=code_path + 'image_data/tensorboard_log/')

    if loadweights:
        if os.path.isfile(WEIGHTS_PATH):
            assert model.load_weights(WEIGHTS_PATH)
            print('model have load pre weights of hanzi image !!')
        else:
            print('model not load weights!!')
    else:
        print('not load weights model')

    # optimizer use sgd
    sgd = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    if not isCenterloss:
        # common cnn model
        print("model compile!!")
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print("model training!!")
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=32000 // BATCH_SIZE,
                                      epochs=max_Epochs,
                                      validation_data=val_generator,
                                      validation_steps=8000 // BATCH_SIZE,
                                      callbacks=[lr, mdcheck, td])
    else:

        # use mygenerator for centerloss generator shape
        train_generator_mygenerator = mygenerator(train_generator)
        val_generator_mygenerator = mygenerator(val_generator)
        centerloss_model = build_centerloss_model(100, 512)
        centerloss_model.compile(optimizer=sgd, loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                                 loss_weights=[1, lambda_center], metrics=['accuracy'])
        history = centerloss_model.fit_generator(train_generator_mygenerator,
                                                 steps_per_epoch=32000 // BATCH_SIZE,
                                                 epochs=max_Epochs,
                                                 validation_data=val_generator_mygenerator,
                                                 validation_steps=8000 // BATCH_SIZE
        callbacks = [lr, mdcheck, td])

        return history

    # draw and save loss pic and acc pic
    def draw_loss_acc(history):
        x_trick = [x + 1 for x in range(max_Epochs)]
        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']

        plt.style.use('ggplot')

        plt.figure(figsize=(10, 6))
        plt.title('model = %s, batch_size = %s' % ('losses', BATCH_SIZE))
        plt.plot(x_trick, loss, 'g-', label='loss')
        plt.plot(x_trick, val_loss, 'y-', label='val_loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.savefig(code_path + 'image_data/image/loss.png', format='png', dpi=300)

        plt.figure(figsize=(10, 6))
        plt.title('learninngRate = %s, batch_size = %s' % ('accuracy', BATCH_SIZE))
        plt.plot(x_trick, val_acc, 'y-', label='val_acc')
        plt.plot(x_trick, acc, 'b-', label='acc')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.show()
        plt.savefig(code_path + 'image_data/image/acc.png', format='png', dpi=300)

    # label for directory in disk
    def label_of_directory(directory):
        """
        sorted for label indices
        return a dict for {'classes', 'range(len(classes))'}
        """
        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)

        num_classes = len(classes)
        class_indices = dict(zip(classes, range(len(classes))))
        return class_indices

    # get key from value in dict
    def get_key_from_value(dict, index):
        for keys, values in dict.items():
            if values == index:
                return keys

    # geneartor list of image list in test
    def generator_list_of_imagepath(path):
        image_list = []
        for image in os.listdir(path):
            if not image == '.DS_Store':
                image_list.append(path + image)
        return image_list

    # read image and resize to gray
    def load_image(image):
        img = Image.open(image)
        img = img.resize((128, 128))
        img = np.array(img)
        img = img / 255
        img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)
        return img

    # get label of model predict test image top_1 preidct
    def get_label_predict_top1(image, model):
        """
        image = load_image(image), input image is a ndarray
        retturn best of label
        """
        predict_proprely = model.predict(image)
        predict_label = np.argmax(predict_proprely, axis=1)
        return predict_label

    # get label of model predict test image top_k predict
    def get_label_predict_top_k(image, model, top_k):
        """
        image = load_image(image), input image is a ndarray
        return top-5 of label
        """
        # array 2 list
        predict_proprely = model.predict(image)
        predict_list = list(predict_proprely[0])
        min_label = min(predict_list)
        label_k = []
        for i in range(top_k):
            label = np.argmax(predict_list)
            predict_list.remove(predict_list[label])
            predict_list.insert(label, min_label)
            label_k.append(label)
        return label_k

    # test image predict best label from model
    def test_image_predict_top1(model, test_image_path, directory):

        model.load_weights(WEIGHTS_PATH)
        image_list = generator_list_of_imagepath(test_image_path)

        predict_label = []
        class_indecs = label_of_directory(directory)
        for image in image_list:
            img = load_image(image)
            label_index = get_label_predict_top1(img, model)
            label = get_key_from_value(class_indecs, label_index)
            predict_label.append(label)

        return predict_label

    # test image predict top-5 label from model
    def test_image_predict_top_k(modle, test_image_path, directory, top_k):

        model.load_weights(WEIGHTS_PATH)
        image_list = generator_list_of_imagepath(test_image_path)

        predict_label = []
        class_indecs = label_of_directory(directory)
        for image in image_list:
            img = load_image(image)
            # return a list of label max->min
            label_index = get_label_predict_top_k(img, model, 5)
            label_value_dict = []
            for label in label_index:
                label_value = get_key_from_value(class_indecs, label)
                label_value_dict.append(str(label_value))

            predict_label.append(label_value_dict)

        return predict_label

    # translate list to str in label
    def tran_list2str(predict_list_label):
        new_label = []
        for row in range(len(predict_list_label)):
            str = ""
            for label in predict_list_label[row]:
                str += label
            new_label.append(str)
        return new_label

    # save filename , lable as csv
    def save_csv(test_image_path, predict_label):
        image_list = generator_list_of_imagepath(test_image_path)
        save_arr = np.empty((10000, 2), dtype=np.str)
        save_arr = pd.DataFrame(save_arr, columns=['filename', 'lable'])
        predict_label = tran_list2str(predict_label)
        for i in range(len(image_list)):
            filename = image_list[i].split('/')[-1]
            save_arr.values[i, 0] = filename
            save_arr.values[i, 1] = predict_label[i]
        save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
        print('submit_test.csv have been write, locate is :', os.getcwd())

    # main function
    if __name__ == "__main__":
        train_path = code_path + 'image_data/train_data/'
        val_path = code_path + 'image_data/val_data/'
        test_image_path = 'image_data/test1/'
        num_classes = 100
        BATCH_SIZE = 128
        WEIGHTS_PATH = 'best_weights_hanzi.hdf5'
        max_Epochs = 100

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(128, 128),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=(128, 128),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            class_mode='categorical'
        )

        simple_model = build_model(num_classes)
        print(simple_model.summary())

        print("=====start train image of epoch=====")

        model_history = model_train(simple_model, False)

        print("=====show acc and loss of train and val====")
        draw_loss_acc(model_history)

        print("=====test label=====")
        simple_model.load_weights(WEIGHTS_PATH)
        model = simple_model
        predict_label = test_image_predict_top_k(model, code_path + test_image_path, train_path, 5)

        print("=====csv save=====")
        save_csv(code_path + test_image_path, predict_label)

        print("====done!=====")










