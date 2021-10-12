import cv2
import numpy as np
import matplotlib.pyplot as plt

picNum = 1
filename = 'pictures/Jung_' + str(picNum) + '.jpg'

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

# kern_size = 2
# kern = np.ones((kern_size, kern_size), np.uint8)
# binary = cv2.dilate(binary, kern, iterations=2)
# cv2.imshow('binary', binary)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 모폴로지 opening 연산
# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
# cv2.imshow('binary', binary)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()

# 바를 정자 두껍게 만들기
repeat = 3
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
del_arr = []
digit_arr = []
count = 0

# 검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)):
    bin_tmp = binary.copy()
    x, y, w, h = cv2.boundingRect(contours[i])

    if w and h > 6:
        bR_arr.append([x, y, w, h])

# 외곽선 박스를 x+y 순으로 정렬
bR_arr = sorted(bR_arr, key=lambda num: num[0] + num[1], reverse=False)


def is_overlap(rect, pt):
    return rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3] or \
           rect[0] < pt[0]+pt[2] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3] or \
           rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1]+pt[3] < rect[1]+rect[3] or \
           rect[0] < pt[0]+pt[2] < rect[0]+rect[2] and rect[1] < pt[1]+pt[3] < rect[1]+rect[3]


for i in range(len(bR_arr)-1):
    # 좌상단 (x,y) = (bR_arr[i][0], bR_arr[i][1]),
    # 우하단 (x+w, y+h) = (bR_arr[i][0] + bR_arr[i][2], bR_arr[i][1] + bR_arr[i][3])

    if is_overlap(bR_arr[i], bR_arr[i+1]):
        # 비교한 두 사각형 영역이 겹치면
        area = bR_arr[i][2] * bR_arr[i][3]

        if area / 3 < (bR_arr[i][0] + bR_arr[i][2] - bR_arr[i + 1][0]) * (
                bR_arr[i][1] + bR_arr[i][3] - bR_arr[i + 1][1]):
            # 자신의 크기의 1/4보다, 겹친 영역이 크면

            if bR_arr[i][0] < bR_arr[i + 1][0]:
                small_x = bR_arr[i][0]
            else:
                small_x = bR_arr[i + 1][0]

            if bR_arr[i][1] < bR_arr[i + 1][1]:
                small_y = bR_arr[i][1]
            else:
                small_y = bR_arr[i + 1][1]

            if bR_arr[i][0] + bR_arr[i][2] > bR_arr[i + 1][0] + bR_arr[i + 1][2]:
                big_w = bR_arr[i][0] + bR_arr[i][2] - small_x
            else:
                big_w = bR_arr[i + 1][0] + bR_arr[i + 1][2] - small_x

            if bR_arr[i][1] + bR_arr[i][3] > bR_arr[i + 1][1] + bR_arr[i + 1][3]:
                big_h = bR_arr[i][1] + bR_arr[i][3] - small_y
            else:
                big_h = bR_arr[i + 1][1] + bR_arr[i + 1][3] - small_y

            # 두 영역을 합쳐 배열에 다시 넣어줌
            bR_arr[i] = [small_x, small_y, big_w, big_h]
            del_arr.append(i+1)
            print(i)
            print(bR_arr[i])


# 겹친 두 영역 중 한 영역 삭제
for i in range(len(del_arr)):
    del bR_arr[del_arr[i]-i]


# 원치 않는 데이터 분류 및 이미지 생성
for x, y, w, h in bR_arr:
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]

    if tmp_x and tmp_y > 10:
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])

# 나눈 이미지의 갯수 출력
print(len(digit_arr))

cv2.imshow('contours', color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0, len(digit_arr)):
    cv2.imwrite('data/' + str(picNum) + '_' + str(i) + '.png', digit_arr[i])
