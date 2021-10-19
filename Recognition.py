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
bR_arr = []     # 사각형 구획 저장
del_arr = []    # 삭제할 bR_arr 인덱스 저장
del_arr2 = []   # 삭제할 bR_arr 인덱스 저장
digit_arr = []  # 이미지 저장용
maskHorizontal_arr = []   # 마스크 이미지 인덱스 저장
maskVertical_arr = []   # 마스크 이미지 인덱스 저장
count = 0

# 검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)):
    bin_tmp = binary.copy()
    x, y, w, h = cv2.boundingRect(contours[i])

    if w and h > 6:
        bR_arr.append([x, y, w, h])

# 외곽선 박스를 x+y 순으로 정렬
bR_arr = sorted(bR_arr, key=lambda num: num[0] + num[1], reverse=False)


def is_overlap(rect, rect2):
    return rect[0] < rect2[0] < rect[0] + rect[2] and rect[1] < rect2[1] < rect[1] + rect[3] or \
           rect[0] < rect2[0] + rect2[2] < rect[0] + rect[2] and rect[1] < rect2[1] < rect[1] + rect[3] or \
           rect[0] < rect2[0] < rect[0] + rect[2] and rect[1] < rect2[1] + rect2[3] < rect[1] + rect[3] or \
           rect[0] < rect2[0] + rect2[2] < rect[0] + rect[2] and rect[1] < rect2[1] + rect2[3] < rect[1] + rect[3]


for i in range(len(bR_arr) - 1):
    # 좌상단 (x,y) = (bR_arr[i][0], bR_arr[i][1]),
    # 우하단 (x+w, y+h) = (bR_arr[i][0] + bR_arr[i][2], bR_arr[i][1] + bR_arr[i][3])
    rect_x = bR_arr[i][0]
    rect_y = bR_arr[i][1]
    rect_w = bR_arr[i][2]
    rect_h = bR_arr[i][3]

    rect2_x = bR_arr[i + 1][0]
    rect2_y = bR_arr[i + 1][1]
    rect2_w = bR_arr[i + 1][2]
    rect2_h = bR_arr[i + 1][3]

    if is_overlap(bR_arr[i], bR_arr[i + 1]):
        # 비교한 두 사각형 영역이 겹치면
        area = rect_w * rect_h

        if area / 3 < (rect_x + rect_w - rect2_x) * (rect_y + rect_h - rect2_y):
            # 자신의 크기의 1/3보다, 겹친 영역이 크면

            if rect_x < rect2_x:
                small_x = rect_x
            else:
                small_x = rect2_x

            if rect_y < rect2_y:
                small_y = rect_y
            else:
                small_y = rect2_y

            if rect_x + rect_w > rect2_x + rect2_w:
                big_w = rect_x + rect_w - small_x
            else:
                big_w = rect2_x + rect2_w - small_x

            if rect_y + rect_h > rect2_y + rect2_h:
                big_h = rect_y + rect_h - small_y
            else:
                big_h = rect2_y + rect2_h - small_y

            # 두 영역을 합쳐 배열에 다시 넣어줌
            bR_arr[i] = [small_x, small_y, big_w, big_h]
            del_arr.append(i + 1)
            # print(i)
            # print(bR_arr[i])

# 겹친 두 영역 중 한 영역 삭제
for i in range(len(del_arr)):
    del bR_arr[del_arr[i] - i]

# for i in range(len(bR_arr)-1, 0, -1):
#     # 좌상단 (x,y) = (bR_arr[i][0], bR_arr[i][1]),
#     # 우하단 (x+w, y+h) = (bR_arr[i][0] + bR_arr[i][2], bR_arr[i][1] + bR_arr[i][3])
#     rect_x = bR_arr[i][0]
#     rect_y = bR_arr[i][1]
#     rect_w = bR_arr[i][2]
#     rect_h = bR_arr[i][3]
#
#     rect2_x = bR_arr[i-1][0]
#     rect2_y = bR_arr[i-1][1]
#     rect2_w = bR_arr[i-1][2]
#     rect2_h = bR_arr[i-1][3]
#
#     if is_overlap(bR_arr[i], bR_arr[i-1]):
#         # 비교한 두 사각형 영역이 겹치면
#         area = rect_w * rect_h
#
#         if area / 3 < (rect_x + rect_w - rect2_x) * (rect_y + rect_h - rect2_y):
#             # 자신의 크기의 1/3보다, 겹친 영역이 크면
#
#             if rect_x < rect2_x:
#                 small_x = rect_x
#             else:
#                 small_x = rect2_x
#
#             if rect_y < rect2_y:
#                 small_y = rect_y
#             else:
#                 small_y = rect2_y
#
#             if rect_x + rect_w > rect2_x + rect2_w:
#                 big_w = rect_x + rect_w - small_x
#             else:
#                 big_w = rect2_x + rect2_w - small_x
#
#             if rect_y + rect_h > rect2_y + rect2_h:
#                 big_h = rect_y + rect_h - small_y
#             else:
#                 big_h = rect2_y + rect2_h - small_y
#
#             # 두 영역을 합쳐 배열에 다시 넣어줌
#             bR_arr[i] = [small_x, small_y, big_w, big_h]
#             del_arr2.append(i-1)
#             print(i)
#             print(bR_arr[i])
#
# # 겹친 두 영역 중 한 영역 삭제
# for i in range(len(del_arr2)):
#     del bR_arr[del_arr2[i]-i]

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

# 이미지 분할 저장
for i in range(0, len(digit_arr)):
    # digit_arr[i] = cv2.resize(digit_arr[i], (32, 32))
    cv2.imwrite('data/' + str(picNum) + '_' + str(i) + '.png', digit_arr[i])

# 이미지가 세로로만 길거나, 가로로만 긴 경우 해당 이미지의 인덱스 저장
for i in range(0, len(digit_arr)):
    img = cv2.imread('data/' + str(picNum) + '_' + str(i) + '.png')
    height, width, channel = img.shape

    if (width // height) > 2:
        maskHorizontal_arr.append(i)

    if (height // width) > 2:
        maskVertical_arr.append(i)

# 이미지 마스킹 후, 저장
for i in range(0, len(digit_arr)):
    for item in maskHorizontal_arr:
        if item == i:
            width = digit_arr[i].shape[1]
            height = digit_arr[i].shape[0]
            tmp = (width - height) / 2
            mask = np.zeros((width, width))
            mask[int(tmp):int(tmp) + height, 0:width] = digit_arr[i]
            digit_arr[i] = cv2.resize(mask, (32, 32))

    for item in maskVertical_arr:
        if item == i:
            width = digit_arr[i].shape[1]
            height = digit_arr[i].shape[0]
            tmp = (height - width) / 2
            mask = np.zeros((height, height))
            mask[0:height, int(tmp):int(tmp) + width] = digit_arr[i]
            digit_arr[i] = cv2.resize(mask, (32, 32))

    digit_arr[i] = cv2.resize(digit_arr[i], (32, 32))
    cv2.imwrite('data/' + str(picNum) + '_' + str(i) + '.png', digit_arr[i])
