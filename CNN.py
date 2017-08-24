import tensorflow as tf
import time as time
import numpy as np
import GetGesture as Ges
import cv2
np.set_printoptions(threshold=np.inf)
cap = cv2.VideoCapture(0)



IMG_SIZE = Ges.image_size
Num_Classes = 4
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])
y = tf.placeholder(tf.float32, [None, Num_Classes])

# Network Architecture
# 3 Convolutional Layers, followed by 2 Fully Connected Layers

# Two Helper functions to create Weights and Biases of the correct dimensions for various layers of the network

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def create_biases(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# Helper functions to apply the convolution and max pooling processes

def Conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def Max_Pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Conv Layer 1
W_conv1 = create_weights([5, 5, 1, 64])
b_conv1 = create_weights([64])
h_conv1 = tf.nn.relu(Conv_2d(x, W_conv1) + b_conv1)
h_pool1 = Max_Pool(h_conv1)

#Conv Layer 2
W_conv2 = create_weights([5, 5, 64, 128])
b_conv2 = create_biases([128])
h_conv2 = tf.nn.relu(Conv_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = Max_Pool(h_conv2)

#Conv Layer 3
W_conv3 = create_weights([5, 5, 128, 128])
b_conv3 = create_biases([128])
h_conv3 = tf.nn.relu(Conv_2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = Max_Pool(h_conv3)

#Fully Connected Layer 1
current_size = int(round(IMG_SIZE / 8.0))
W_fc1 = create_weights([current_size*current_size*128, 4096])
b_fc1 = create_biases([4096])
h_pool3_flat = tf.reshape(h_pool3, [-1, current_size*current_size*128])
h_fc_1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob)

#Full Connected Layer 2
W_fc2 = create_weights([4096, Num_Classes])
b_fc2 = create_biases([Num_Classes])
prediction = tf.matmul(h_fc_1_drop, W_fc2) + b_fc2

Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(Cost)
Correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))
Accuracy = tf.reduce_mean(tf.cast(Correct_prediction, "float"))

init = tf.global_variables_initializer()
session = tf.Session()
saver = tf.train.Saver()

session.run(init)
start_time = time.time()
try:
    for iteration in range(500):
        batch_x, batch_y = Ges.next_batch(100, Ges.train_x, Ges.train_y)
        session.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
        if iteration % 100 == 0:
             print (" Accuracy on {}th Iteration: {} \n".format(iteration, session.run(Accuracy, feed_dict={x: Ges.test_x, y: Ges.test_y, keep_prob: 1})) + \
                  " Cost: {} \n".format(session.run(Cost, feed_dict={x: Ges.train_x, y: Ges.train_y, keep_prob: 1})) + \
                " Time elapsed: {} seconds \n".format(time.time() - start_time))

    print ("Final Model Accuracy: {}".format(session.run(Accuracy, feed_dict={x: Ges.test_x, y: Ges.test_y, keep_prob: 1})))
    print ("Total Time elapsed: {} minutes".format(round((time.time() - start_time) / 60.0)))
    print ("Saving model for future use")
    save_path = saver.save(session, '/tmp/OmarsModelC.ckpt')

except KeyboardInterrupt:
    print ("Training terminated prematurely")
    print ("Final Model Accuracy: {}".format(session.run(Accuracy, feed_dict={x: Ges.test_x, y: Ges.test_y, keep_prob: 1})))
    print ("Total Time elapsed: {} minutes".format(round((time.time() - start_time) / 60.0)))
    print ("Saving model for future use")
    save_path = saver.save(session, 'C:\\Users\\Omar\\Documents\\FreshPython\\TF_Model')

# with tf.Session() as session:
#     saver.restore(session, '/home/odiab/Documents/OmarsGestureModelC.ckpt')
#     print "Restored Model Accuracy: {}".format(session.run(Accuracy, feed_dict={x: Ges.test_x, y: Ges.test_y, keep_prob: 1}))
#     while (True):
#         ret, frame = cap.read()
#         MIN = np.array([0, 30, 60], np.uint8)
#         MAX = np.array([20, 150, 179], np.uint8)
#         HSVImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         filterImg = cv2.inRange(HSVImg, MIN, MAX)
#         cv2.imshow('frame', filterImg)
#         image = cv2.resize(filterImg, (60, 60))
#         input = np.reshape(image, (-1, 60, 60, 1))
#         predicted = session.run(prediction, feed_dict={x: input, keep_prob: 1})
#         output = np.argmax(predicted)
#         if output == 0:
#             print "Up"
#         if output == 1:
#             print "hand"
#         if output == 2:
#             print "Peace"
#
#         if cv2.waitKey(1) and 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
