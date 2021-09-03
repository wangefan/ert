import cv2
from ERT import ERT
import Utilis

root = r"./"
model_full_path = root + r"lfpw/ert_model.json"
test_path = root + r"lfpw/testset_/"
test_video_path = test_path + r"face.mp4"




#print('3. test data[2]')
#test_data2_image = cv2.imread(test_images_path, cv2.IMREAD_GRAYSCALE)


# 1.Load model & video
print('1. Load model & video..')
ert = ERT.load(model_full_path)
cap = cv2.VideoCapture(test_video_path)

# Begin to read video frame
print('2. Begin to read video frame..')
while(cap.isOpened()):
  ret, frame = cap.read()

  cv2.imshow('frame', frame)
  ert.predict(frame)
  cv2.waitKey(25)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
