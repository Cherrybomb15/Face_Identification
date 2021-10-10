import os
import cv2
import numpy as np
# import scipy.linalg

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Function to take file names and convert it to an array of images
def f(arr):
    return np.array(list(map(lambda y: cv2.cvtColor(cv2.imread(y), cv2.COLOR_BGR2HSV), arr)))


# Create arrays of all the image file names in each folder, then apply our function, f, to load each image file
testing_back = f(
    (np.vectorize(lambda x: 'testingData/background/' + x))(np.array(os.listdir('testingData/background'))))
testing_face = f(
    (np.vectorize(lambda x: 'testingData/face/' + x))(np.array(os.listdir('testingData/face'))))

training_back = f(
    (np.vectorize(lambda x: 'trainingData/background/' + x))(np.array(os.listdir('trainingData/background'))))
training_face = f(
    (np.vectorize(lambda x: 'trainingData/face/' + x))(np.array(os.listdir('trainingData/face'))))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Take the averages of the pixels for the face images
    mu1 = np.average(training_face, axis=0)

    # Reshape this matrix to an vector
    mu1 = mu1.reshape((np.array(mu1.shape)).prod(), 1)

    # Take the averages of the pixels for the background images
    mu0 = np.average(training_back, axis=0)

    # Reshape this matrix to an vector
    mu0 = mu0.reshape((np.array(mu0.shape)).prod(), 1)

    # Number of pixels we need to flatten the matrix to for a matrix
    pixels = np.array(training_face.shape)[1:].prod()

    # Reshaped versions of the training data to iterate through
    tf1 = training_face.reshape(training_face.shape[0], pixels, 1)
    tf0 = training_back.reshape(training_back.shape[0], pixels, 1)

    # Generate the vector for covariance matrix of faces
    sig1 = np.zeros(shape=(pixels, pixels), dtype=np.longdouble)
    for x in tf1:
        sig1 += (x - mu1) * np.transpose(x - mu1)
    sig1 = np.divide(sig1, tf1.shape[0])

    # Generate the vector for covariance matrix of backgrounds
    sig0 = np.zeros(shape=(pixels, pixels), dtype=np.longdouble)
    for x in tf0:
        sig0 += (x - mu0) * np.transpose(x - mu0)

    # Normalize the matrix
    sig0 = np.divide(sig0, tf0.shape[0])

    # Find the determinant of the covariant matrices by just looking at the diagonal elements
    det_sig0 = np.sum(np.log(sig0.diagonal())).item()
    det_sig1 = np.sum(np.log(sig1.diagonal())).item()

    #p, l, u = scipy.linalg.lu(sig0)
    #det_sig0 = np.sum(np.log(l.diagonal())).item() + np.sum(np.log(np.abs(u.diagonal()))).item()
    #print(u.diagonal())

    #p, l, u = scipy.linalg.lu(sig1)
    #det_sig1 = np.sum(np.log(l.diagonal())).item() + np.sum(np.log(np.abs(u.diagonal()))).item()

    #sig0_inv = np.linalg.inv(sig0)
    #sig1_inv = np.linalg.inv(sig1)

    # Find the inverse of the diagonal version of the matrices
    sig0_inv = np.linalg.inv(np.diagflat(sig0.diagonal()))
    sig1_inv = np.linalg.inv(np.diagflat(sig1.diagonal()))

    # Now that we have our average and covariance we can start guessing which image is a face and not a face
    def p_work(v):
        a = np.transpose(v - mu0)
        b = np.matmul(a, sig0_inv)
        c = v - mu0
        d = np.matmul(b, c)
        x = np.transpose(v - mu1)
        y = np.matmul(x, sig1_inv)
        z = v - mu1
        help = np.matmul(y, z)
        if det_sig0 + d.item() > det_sig1 + help.item():
            return 1
        else:
            return 0

    faces = 0
    for x in testing_face.reshape(testing_face.shape[0], pixels, 1):
        faces += p_work(x)

    backs = 0
    i = 1
    for x in testing_back.reshape(testing_back.shape[0], pixels, 1):
        if p_work(x) == 0:
            backs += 1

    # print(backs / testing_back.shape[0])

    precision = faces / (faces + (testing_back.shape[0] - backs))
    recall = faces / (faces + (testing_face.shape[0] - faces))
    fscore = (2 * precision * recall) / (precision + recall)
    print('precision:', precision, '\nrecall:', recall, '\nfscore: ', fscore)