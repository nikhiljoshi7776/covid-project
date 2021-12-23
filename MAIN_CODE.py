import tkinter
import tkinter.filedialog
from tkinter import messagebox 
import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
#from Sim_SV import Calc_Wt
import time
from tkinter import filedialog
import tkinter.messagebox
import cv2
from PyQt4 import QtCore, QtGui
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf, sys
import cv2
import time
from tkinter import filedialog
import tkinter.messagebox
import numpy as np
import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


root=tkinter.Tk()

def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('TRUE CLASS')
    plt.xlabel('PREDICTED CLASS')
    plt.tight_layout()
    
def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default
    

print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(bg3.jpg);\n"""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 180, 111, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
       
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################
        

        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 180, 131, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 220, 131, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(255, 128, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "ADMIN PANEL", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST ", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        #self.pushButton_5.setText(_translate("MainWindow", "ACCURACY", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()
         
    def show1(self):
        image_path= filedialog.askopenfilename(filetypes = (("BROWSE CHEST X-RAY IMAGE", "*.jpg"), ("All files", "*")))
        I=cv2.imread(image_path)
        #I=cv2.imread('TESTFULL.jpg')
        cv2.imshow('INPUT IMAGE',I);
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        # RESIZING
        img = cv2.resize(I,(512,512),3)
        cv2.imshow('RESIZED IMAGE',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # MEDIAN FILTERED
        img1 = cv2.medianBlur(img,5)
        cv2.imshow('MEDIAN IMAGE',img1)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # GRAY CONVERSION
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        cv2.imshow('GRAY IMAGE',gray)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # Open label file
        label_lines = [line.rstrip() for line
            in tf.gfile.GFile("retrained_labels.txt")]
        # CNN trained file
        with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')



        # TEST given input image
        with tf.Session() as sess:
             softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
             print(softmax_tensor)
             predictions = sess.run(softmax_tensor, 
             {'DecodeJpeg/contents:0': image_data})
             # Confidenece
             top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
             human_string = label_lines[0]
             score1 = predictions[0][0]
             print('PREDICTION----:\n',predictions)
             #print('SCORES--:\n',score1)
             CurID=np.argmax(predictions)
             print('PREDICTED CLASS INDEX\n',CurID)

        print('-----------------------------------------------------\n')
        print('----------------------RESULT-------------------------\n')
        if np.max(predictions)>=0.4:
            if CurID==1:
                print('CLASS: COVID--:\n')
            if CurID==0:
                print('CLASS: NORMAL--:\n')
        else:
            print('UNABLE TO PREDICT--:\n')




        

    def show2(self):
        import socket
        import time
        import cv2
        import os
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from keras.layers import Input, Dense
        from  keras import regularizers
        from  keras.models import Sequential, Model
        from  keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
        from  keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
        from keras.layers import Concatenate
        from keras.preprocessing.image import ImageDataGenerator
        from keras.optimizers import Adam, SGD
        import pickle

        # define parameters
        CLASS_NUM = 5
        BATCH_SIZE = 16
        EPOCH_STEPS = int(4323/BATCH_SIZE)
        IMAGE_SHAPE = (224, 224, 3)
        IMAGE_TRAIN = 'TRNMDL'
        MODEL_NAME = 'retrained_graph.pb'
        def inception(x, filters):
                path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
                path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)
                path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
                path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)
                return Concatenate(axis=-1)([path1,path2,path3,path4])
        def auxiliary(x, name=None):
                layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
                layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Flatten()(layer)
                layer = Dense(units=256, activation='relu')(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
                return layer

        layer_in = Input(shape=IMAGE_SHAPE)
        layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
        layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
        layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
        aux1  = auxiliary(layer, name='aux1')
        layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
        layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
        layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
        aux2  = auxiliary(layer, name='aux2')
        layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
        layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
        layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
        layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
        layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
        layer = Flatten()(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=256, activation='linear')(layer)
        main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
        model = Model(inputs=layer_in, outputs=[main, aux1, aux2])

        print(model.summary())
        file= open("TRNMDL.obj",'rb')
        cnf_matrix = pickle.load(file)
        file.close()

        plt.figure()
        plot_confusion_matrix(cnf_matrix[1:3,1:3], classes=['Normal ','Covid'], normalize=True,title='Proposed Method')
        plt.show()



if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())
    

