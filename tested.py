import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math
import os
from keras.models import load_model
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.
    cm: Confusion matrix (2D array)
    classes: List of class names
    normalize: Whether to normalize the matrix or not (default=False)
    title: Title of the plot (default='Confusion Matrix')
    cmap: Color map (default=plt.cm.Blues)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    model_path = "Model/keras_model.h5"
    model = load_model(model_path)
    print(model.summary())
    offset = 20
    imgSize = 300

    counter = 0
    text = ""
    word = ""
    count_same_frame = 0

    labels = ["A", "B", "Bestfriend", "Boy", "C", "D", "E", "F", "G", "Girl", "H", "Hello", "I", "i love you", "J", "K",
              "L", "M", "N", "O", "P", "Q", "R", "S", " ", "T", "Thanks", "U", "V", "W", "X", "Y", "Yes", "You", "Z"]
    predicted_labels = []
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        old_text = text

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']  ##bounding box

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            # np.unit8 from 0 to 255 *255 because the pixels are from o to 1
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            # to centre image weihght , according to which is bigger
            # height or width
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if imgCrop.size != 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                else:
                    continue
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if imgCrop.size != 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                else:
                    continue
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            print("index ", index)
            text = labels[index]
            if old_text == text:
                count_same_frame += 1
            else:
                count_same_frame = 0
            flag = True
            if count_same_frame > 8:
                word = word + text
                predicted_labels.append(text)
                count_same_frame = 0
            # predicted_label =  labels[index]  # Predicted label
            # predicted_labels.append(predicted_label)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

            cv2.putText(blackboard, "Predicted text- " + labels[index], (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 0))
            cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # cv2.imshow("Image", imgOutput)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # true_labels = np.argmax(labels, axis=1)
    labels_test = labels[:len(predicted_labels)]
    print("predicted_labels ", predicted_labels)
    print("labels_test ", labels_test)

    cm = confusion_matrix(labels_test, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    print("accuracy_score")
    print(accuracy_score(labels_test, predicted_labels))
    from sklearn.metrics import classification_report

    # Compute the accuracy score report
    report = classification_report(labels_test, predicted_labels)

    # Print the accuracy score report
    print("Accuracy Score Report:")
    print(report)
    # Plot the confusion matrix
    plot_confusion_matrix(cm, classes=labels_test, normalize=False, title='Confusion Matrix')

    # Show the plot
    plt.show()