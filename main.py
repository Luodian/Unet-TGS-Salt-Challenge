import gc
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dropout, Input, LeakyReLU, concatenate
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm

Train_Image_folder = './input/train/images/'
Train_Mask_folder = './input/train/masks/'
Test_Image_folder = './input/test/images/'
Train_Image_name = os.listdir(path=Train_Image_folder)
Test_Image_name = os.listdir(path=Test_Image_folder)
Train_Image_path = []
Train_Mask_path = []
Train_id = []
for i in Train_Image_name:
    path1 = Train_Image_folder + i
    path2 = Train_Mask_folder + i
    id1 = i.split(sep='.')[0]
    Train_Image_path.append(path1)
    Train_Mask_path.append(path2)
    Train_id.append(id1)

Test_Image_path = []
Test_id = []
for i in Test_Image_name:
    path = Test_Image_folder + i
    id2 = i.split(sep='.')[0]
    Test_Image_path.append(path)
    Test_id.append(id2)

df_Train_path = pd.DataFrame(
    {'id': Train_id, 'Train_Image_path': Train_Image_path, 'Train_Mask_path': Train_Mask_path})
df_Test_path = pd.DataFrame({'id': Test_id, 'Test_Image_path': Test_Image_path})

# 存储每一个图像的深度信息，这里包括了train和test
df_depths = pd.read_csv('./input/depths.csv')
df_sub = pd.read_csv('./input/sample_submission.csv')
df_Train_path = df_Train_path.merge(df_depths, on='id', how='left')
df_Test_path = df_Test_path.merge(df_depths, on='id', how='left')
df_Test_path = df_sub.merge(df_Test_path, on='id', how='left')
print(df_Train_path.shape, df_Test_path.shape)

print(df_Train_path.head())

df_Test_path.drop('rle_mask', axis=1, inplace=True)

df_Train_path["images"] = [np.array(load_img(path=idx, color_mode="grayscale")) / 255 for idx in
                           tqdm(df_Train_path.Train_Image_path)]
df_Train_path["masks"] = [np.array(load_img(path=idx, color_mode="grayscale")) / 255 for idx in
                          tqdm(df_Train_path.Train_Mask_path)]
df_Train_path["coverage"] = df_Train_path.masks.map(np.sum) / pow(101, 2)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


df_Train_path["coverage_class"] = df_Train_path.coverage.map(cov_to_class)

img_size_ori = 101
img_size_target = 128


# 101x101 -> 128x128
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


# 128x128 -> 101x101
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = \
    train_test_split(
        df_Train_path.id.values,
        np.array(df_Train_path.images.map(upsample).tolist()).reshape(-1, img_size_target,
                                                                      img_size_target,
                                                                      1),
        np.array(df_Train_path.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target,
                                                                     1),
        df_Train_path.coverage.values,
        df_Train_path.z.values,
        test_size=0.2, stratify=df_Train_path.coverage_class, random_state=123)

gc.collect()

print(ids_train.shape, ids_valid.shape)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(cov_train.shape, cov_test.shape)
print(depth_train.shape, depth_test.shape)


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


model = UNet((img_size_target, img_size_target, 1), start_ch=16, depth=5, batchnorm=True)
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])

# Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

gc.collect()

early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

epochs = 200
batch_size = 64

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr,
                               TensorBoard(log_dir='8_24')])

model = load_model("./keras.model", custom_objects={'mean_iou': mean_iou})

preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid = np.array([downsample(x) for x in y_valid])


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[
        0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


thresholds = np.linspace(0, 1, 50)
ious = np.array(
    [iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
# threshold_best = 0.638


print('Best Threshold: ', threshold_best)


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


x_test = np.array([upsample(np.array(load_img(path=idx, color_mode='grayscale'))) / 255 for idx in
                   tqdm(df_Test_path.Test_Image_path)]).reshape(-1, img_size_target, img_size_target, 1)

preds_test = model.predict(x_test)

pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in
             enumerate(tqdm(df_Test_path.id.values))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')

# applying crf method

from crf_sub import rle_decode
from crf_sub import rle_encode
from crf_sub import crf

test_path = './input/test/images/'

for i in tqdm(range(sub.shape[0])):
    if str(sub.loc[i, 'rle_mask']) != str(np.nan):
        decoded_mask = rle_decode(sub.loc[i, 'rle_mask'])
        orig_img = imread(test_path + sub.loc[i, 'id'] + '.png')
        crf_output = crf(orig_img, decoded_mask)
        sub.loc[i, 'rle_mask'] = rle_encode(crf_output)

sub.to_csv('crf_submission.csv', index=False)
