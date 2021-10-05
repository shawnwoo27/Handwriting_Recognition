import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'pictures/Jung.jpg'

# 그레이 스케일로 변환
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
src = cv2.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2)))
cv2.imshow('gray', src)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 사진 데이터 이진화(binary)
ret, binary = cv2.threshold(src, 140, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('binary', binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 노이즈 제거
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
cv2.imshow('binary', binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 외곽선 검출
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이진화 이미지를 color 이미지로 복사
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# 초록색으로 외곽선 그리기
cv2.drawContours(color, contours, -1, (0, 255, 0), 6)
cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 검출한 외곽선에 사각형을 그려서 배열에 추가
boundingRectangle_arr = []
for i in range(len(contours)):
    bin_tmp = binary.copy()
    x, y, w, h = cv2.boundingRect(contours[i])
    boundingRectangle_arr.append([x, y, w, h])
    cv2.rectangle(color, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1)

print(len(boundingRectangle_arr))

cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

'''
for i in range(len(boundingRectangle_arr)):
    img[i] = cv2.resize(img[i], (32, 32))
    cv2.imwrite('data/'+str(i)+'.png', img[i])
    print(i)
'''

