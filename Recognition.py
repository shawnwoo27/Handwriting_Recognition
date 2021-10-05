import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'pictures/Jung.jpg'

# 그레이 스케일로 변환
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# src = cv2.resize(src, (int(src.shape[1]/2), int(src.shape[0]/2)))

# 그레이스케일링한 이미지를 통해, 가로축 256개의 히스토그램 생성
hist, bin_edges = np.histogram(src, bins=256)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(src, cmap='gray')
plt.title('original image', fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot(122), plt.plot(hist), plt.title('Histogram', fontsize=15)
plt.show()

cv2.imshow('gray', src)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 사진 데이터 이진화(binary)
# ret, binary = cv2.threshold(src, 140, 255, cv2.THRESH_BINARY_INV)
binary = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
"""
# otsu알고리즘을 통한 최적의 thresh값 찾기
ret, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(ret) # 154
"""
cv2.imshow('binary', binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 바를 정자 두껍게 만들기
repeat = 5
kern_size = 2
kern = np.ones((kern_size, kern_size), np.uint8)
binary = cv2.dilate(binary, kern, iterations=repeat)
cv2.imshow('binary', binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 노이즈 제거
for i in range(repeat + 1):
    binary = cv2.morphologyEx(binary,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size)),
                              iterations=i)

cv2.imshow('binary', binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 외곽선 검출
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이진화 이미지를 color 이미지로 복사
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# 초록색으로 외곽선 그리기
cv2.drawContours(color, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# 연산 데이터 저장용 변수 선언
bR_arr = []
digit_arr = []
count = 0

# 검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)):
    bin_tmp = binary.copy()
    x, y, w, h = cv2.boundingRect(contours[i])
    bR_arr.append([x, y, w, h])

# 나눈 이미지의 갯수 출력
print(len(bR_arr))

# 원치 않는 데이터 분류 및 이미지 생성
for x, y, w, h in bR_arr:
    tmp_y = bin_tmp[y-2:y+h+2, x-2:x+w+2].shape[0]
    tmp_x = bin_tmp[y-2:y+h+2, x-2:x+w+2].shape[1]

    if tmp_x and tmp_y > 10:
        cv2.rectangle(color, (x-3, y-3), (x+w+3, y+h+3), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y-2:y+h+2, x-2:x+w+2])

cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0, len(digit_arr)):
    cv2.imwrite('data/'+str(i)+'.png', digit_arr[i])
