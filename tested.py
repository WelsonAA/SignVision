import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
import math
import os
from keras.models import load_model
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/labels.txt")
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
          "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T", "Thanks", "U", "V", "W", "X", "Y", "Yes", "You", "Z"]
predicted_labels = []
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    old_text = text

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
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
    # true_labels = np.argmax(labels, axis=1)
labels_test = labels[:len(predicted_labels)]
print("predicted_labels ", predicted_labels)
print("labels_test ", labels_test)

cm = multilabel_confusion_matrix(labels_test, predicted_labels)
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

fig, axes = plt.subplots(nrows=len(labels_test), ncols=1, figsize=(5, 5 * len(labels_test)))

for i, ax in enumerate(axes):
    ax.matshow(cm[i], cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix for Label {labels_test[i]}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()