import os

import numpy as np
import cv2
from naive.utils import get_bar_endpoints, pair_and_draw_armor_bars

def process_frame(frame):
    BINARY_THRESHOLD = 200  # 二值化阈值
    ASPECT_RATIO_THRESHOLD = 3.0  # 长宽比阈值
    VERTICALITY_THRESHOLD = 0.5  # 垂直度阈值
    AREA_THRESHOLD= 1000  # 面积阈值
    # 原图信息
    height, width = frame.shape[:2]
    original = frame.copy()

    # 灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化
    #TODO 加上颜色检测才准？
    _, thresh = cv2.threshold(blurred, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 腐蚀膨胀，去掉小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.erode(thresh, kernel, iterations=1)
    morph = cv2.dilate(morph, kernel, iterations=2)

    # 查找轮廓并画到一张图上
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = np.zeros_like(frame)
    cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 1)

    bar_data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 面积过滤
        if area < AREA_THRESHOLD:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w > h:
            w, h = h, w

        # 长宽比过滤
        aspect_ratio = h / (w + 1e-6)
        if aspect_ratio < ASPECT_RATIO_THRESHOLD:
            continue

        top_pts, bottom_pts, u_vec, v_vec = get_bar_endpoints(rect)

        # 垂直度过滤
        if abs(v_vec[1]) < VERTICALITY_THRESHOLD:
            continue
        bar_data.append((rect, top_pts, bottom_pts, u_vec, v_vec))

    # 两两配对计算装甲中心
    centers_img = frame.copy()
    centers = pair_and_draw_armor_bars(bar_data, width, centers_img)

    # 输出图像
    figure_width, figure_height = width // 3, height // 3

    def draw_figure(img):
        return cv2.resize(img, (figure_width, figure_height), interpolation=cv2.INTER_AREA)

    # 画出识别到的光条
    bars_img = np.zeros_like(frame)
    for rect, tpts, bpts, u_vec, v_vec in bar_data:
        box = cv2.boxPoints(rect).astype(int)
        cv2.polylines(bars_img, [box], True, (0, 0, 255), 2)

    # 转换图像
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    # 拼接
    figures = [
        draw_figure(original),
        draw_figure(gray_bgr),
        draw_figure(blurred_bgr),
        draw_figure(thresh_bgr),
        draw_figure(morph_bgr),
        draw_figure(contours_img),
        draw_figure(bars_img),
        draw_figure(centers_img)
    ]
    black = np.zeros_like(figures[0])
    figures.append(black)
    row1 = cv2.hconcat([figures[0], figures[1], figures[2]])
    row2 = cv2.hconcat([figures[3], figures[4], figures[5]])
    row3 = cv2.hconcat([figures[6], figures[7], figures[8]])
    collage = cv2.vconcat([row1, row2, row3])

    return centers_img, collage

if __name__ == '__main__':
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
    img = cv2.imread(f'{input_dir}/test1.jpg')
    _, collage = process_frame(img)
    cv2.imshow('Result', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()