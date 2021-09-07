import cv2
from ERT import ERT
import Utilis
import threading
import copy

class PredictThread(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    self._lock = threading.Lock()
    self._predict_image = None
    self._cur_landmark = None
    self._ert = ERT.load(model_full_path)
    print('PredictThread, load model ok')

  def commitFrame(self, grayFrame):
    self._lock.acquire()
    self._predict_image = copy.deepcopy(grayFrame)
    self._lock.release()

  def getLandmark(self):
    self._lock.acquire()
    cur_landmark = copy.deepcopy(self._cur_landmark)
    self._lock.release()
    return cur_landmark

  def run(self):
    print('==PredictThread begin==')

    while True:
      predict_image = None
      self._lock.acquire()
      predict_image = copy.deepcopy(self._predict_image)
      self._lock.release()
      cur_landmark = self._ert.predict(predict_image)

      self._lock.acquire()
      if cur_landmark is not None:
        self._cur_landmark = copy.deepcopy(cur_landmark)
      else:
        self._cur_landmark = None
      self._lock.release()


root = r"./"
model_full_path = root + r"lfpw/ert_model_good.json"
test_path = root + r"lfpw/testset_/"
test_video_path = test_path + r"face.mp4"


# 1. Begin predict thread to load model and begin predict loop
print('1. Begin predict thread to load model and begin predict loop')
pt = PredictThread()
pt.start()

# Begin to read video frame
print('2. Begin to read video frame..')
cap = cv2.VideoCapture(test_video_path)
while(cap.isOpened()):
  ret, frame = cap.read()
  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  pt.commitFrame(grayFrame)
  cv2.waitKey(25)
  cur_landmark = pt.getLandmark()
  if cur_landmark is not None:
    Utilis.drawLandmarks(cur_landmark, frame, 2, (0, 122, 255))
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
