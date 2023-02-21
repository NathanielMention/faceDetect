import pathlib
import cv2

# import xml files for faces from cv2
cascade_path = pathlib.Path(cv2.__file__).parent.absolute(
) / "data/haarcascade_frontalface_default.xml"
