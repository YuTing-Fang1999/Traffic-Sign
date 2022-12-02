"""
Main program to run the detection
"""

from argparse import ArgumentParser
import cv2
import numpy as np
from model import MyModel

my_model = MyModel(n_layer=6, path="augment_x64_0_best.ckpt")

def main():

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        pred = my_model.pred(img)

        # 使用各種字體
        cv2.putText(img, pred, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('camera', img)
   
        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()