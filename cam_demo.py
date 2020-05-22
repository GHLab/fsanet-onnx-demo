import cv2
import numpy as np
import onnxruntime
from math import cos, sin
import datetime

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def main():
    ort_session = onnxruntime.InferenceSession("fsanet_epoch_95.onnx")

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    video_capture = cv2.VideoCapture(0)

    ad = 0.6

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        img_h, img_w, _ = np.shape(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for i, (x,y,w,h) in enumerate(faces):
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)

            img = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]

            img = cv2.resize(img, (64, 64))
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            start = datetime.datetime.now()

            ort_inputs = {ort_session.get_inputs()[0].name: img}
            pred_labels = ort_session.run(None, ort_inputs)

            finish = datetime.datetime.now()
            #print("fsa-net duration : ", int((finish - start).total_seconds() * 1000))

            pitch = pred_labels[0][0][0]
            yaw = pred_labels[0][0][1]
            roll = pred_labels[0][0][2]

            draw_axis(frame, yaw, pitch, roll)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()