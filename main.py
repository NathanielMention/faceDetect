import pathlib
import cv2

# import xml files for faces from cv2
cascade_path = pathlib.Path(cv2.__file__).parent.absolute(
) / "data/haarcascade_frontalface_default.xml"

# build classifier based on cascade_path file, finding faces in img data
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

while True:
    # setup camera frame
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
