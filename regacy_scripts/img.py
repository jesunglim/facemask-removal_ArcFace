# bitwise_and 연산으로 마스킹하기 (bitwise_masking.py)

import numpy as np
import cv2
import matplotlib.pylab as plt

#--① 이미지 읽기
img = cv2.imread('./14.jpg')

#--② 마스크 만들기
mask = cv2.rectangle(img, (0,65), (112,112), 0, -1)


#--④ 결과 출력
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()