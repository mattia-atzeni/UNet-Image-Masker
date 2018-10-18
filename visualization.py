import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import params

input_size = params.input_size
batch_size = 1
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model

df_test = pd.read_csv('input/meta.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


model.load_weights(filepath='weights/best_weights.hdf5')

for start in tqdm(range(0, len(ids_test), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img_orig = cv2.imread('input/test/{}.jpg'.format(id))
        img = cv2.resize(img_orig, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for pred in preds:
        prob = cv2.resize(pred, (orig_width, orig_height))
        mask = prob > threshold
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img_orig)
        plt.subplot(132)
        plt.imshow(mask, cmap='winter')
        plt.subplot(133)
        plt.imshow(img_orig)
        plt.imshow(mask, alpha=0.5, cmap='winter')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()
