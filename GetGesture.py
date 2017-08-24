import cv2
import numpy as np
import time
import os
from random import shuffle

cap = cv2.VideoCapture(0)
train_data = np.load("training_data.npy")
train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")
image_size = 60


def Show(dir, gesture_name, save=False):
    i = 0
    j = 0
    init = True
    while cap.isOpened():
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        x, y, w, h = (120, 100, 175, 175)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        ROI = thresh[y:y+h,x:x+w]

        # cv2.imshow("thresh", thresh)
        cv2.imshow("image", frame)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(1)
        if j > 75 and save == True:
            cv2.imwrite(dir+gesture_name+'.{}.png'.format(i), ROI)
            print ("Saved Image {}".format(i))
            i += 1
        j += 1
        if cv2.waitKey(1) and 0xFF == ord('q') or i > 1000:
            break

    cap.release()
    cv2.destroyAllWindows


def label_image(image):
    label = image.split(".")[0]
    if label == "Hand":
        return [1,0,0,0]
    elif label == "Up":
        return [0,1,0,0]
    elif label == "Peace":
        return [0,0,1,0]
    elif label == "Weird":
        return [0,0,0,1]

def create_data(train_dir, image_size):
    training_data = []
    for image in os.listdir(train_dir):
        label = label_image(image)
        path = os.path.join(train_dir, image)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (image_size, image_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save("training_data.npy", training_data)
    return training_data

# create_data("C:\\Users\\Omar\\Documents\\FreshPython\\Gesture_Data\\All_Data\\", 60)


def check():
    for pair in train_data[:10]:
        cv2.imshow("image", pair[0])
        cv2.waitKey(0)
        if np.array_equal(pair[1], [1,0,0,0]):
            print ("Hand")
        elif np.array_equal(pair[1], [0,1,0,0]):
            print ("Up")
        elif np.array_equal(pair[1], [0,0,1,0]):
            print ("Peace")
        else:
            print ("Weird")

def split_data(train, test, image_size):
    print (len(train_data))
    tr_split = int(round(train*(len(train_data))))
    te_split = int(round(test*(len(train_data))))
    print (tr_split, te_split, (tr_split+te_split))
    train = train_data[:-te_split]
    test = train_data[-te_split:]
    print (len(train), len(test))
    test_x = np.array([i[0] for i in test]).reshape(-1, image_size, image_size, 1)
    test_y = [i[1] for i in test]
    train_x = np.array([i[0] for i in train]).reshape(-1, image_size, image_size, 1)
    train_y = [i[1] for i in train]
    np.save("test_x", test_x)
    np.save("test_y", test_y)
    np.save("train_x", train_x)
    np.save("train_y", train_y)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# check()
# Show("C:\\Users\\Omar\\Documents\\FreshPython\\Gesture_Data\\Weird\\", "Weird", save=True)
# split_data(0.8,0.2, 60)
