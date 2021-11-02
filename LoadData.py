import os
import cv2
import numpy as np
from sklearn.preprocessing import *


# scikit-learn 라이브러리 사용을 위해 데이터셋을 생성
def create_dataset(directory):
    files = os.listdir(directory)
    x = []
    y = []
    for file in files:
        attr_x = cv2.imread(directory + file)
        attr_x = attr_x.reshape(32, 32, 1)
        attr_y = int(file[0])

        x.append(attr_x)
        y.append(attr_y)

    x = np.array(x)

    y = np.array([[i] for i in y])
    enc = OneHotEncoder(categories='auto')
    enc.fit(y)
    y = enc.transform(y).toarray()

    return x, y


# train_dir = ''
# train_x, train_y = create_dataset(train_dir)

test_dir = 'data/'
test_x, test_y = create_dataset(test_dir)
