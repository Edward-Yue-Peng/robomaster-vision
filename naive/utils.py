import cv2
import numpy as np

def get_bar_endpoints(rect):
    """
    计算旋转矩形的长边方向和垂直方向的单位向量，并返回长边方向上的顶点和底点。
    :param rect: tuple，矩形 ((cx, cy), (w, h), angle)
            cx, cy: 矩形中心坐标
            w, h: 矩形宽度和高度
            angle: 矩形旋转角度
    :return:
        top_pts: numpy.ndarray, shape (2, 2)
            沿长边方向投影值最大的两个点（顶部点）
        bottom_pts: numpy.ndarray, shape (2, 2)
            沿长边方向投影值最小的两个点（底部点）
        u: numpy.ndarray
            垂直于长边方向的单位向量（从左向右）
        v: numpy.ndarray
            沿长边方向的单位向量（从下向上）
    """

    # 获取矩形的四个角
    box = cv2.boxPoints(rect).astype(np.float32)

    # 找长边
    max_len = 0
    max_idx = 0
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        if length > max_len:
            max_len = length
            max_idx = i
    # 计算长边方向的单位向量 v 和垂直方向的单位向量 u
    pA = box[max_idx]
    pB = box[(max_idx + 1) % 4]
    v_vec = pB - pA
    v_norm = v_vec / (np.linalg.norm(v_vec) + 1e-6)  # 单位向量 v：长边方向
    # 垂直于 v 的单位向量 u
    u_norm = np.array([-v_norm[1], v_norm[0]], dtype=np.float32)

    # 计算角点在长边方向上的投影值，并找到投影值最大的两点和最小的两点
    projs_v = [pt.dot(v_norm) for pt in box]
    idx_sorted_v = np.argsort(projs_v)
    bottom_idxs = idx_sorted_v[:2]  # 投影最小两个点
    top_idxs = idx_sorted_v[-2:]  # 投影最大两个点

    top_pts = box[top_idxs]
    bottom_pts = box[bottom_idxs]

    return top_pts, bottom_pts, u_norm, v_norm

# POWERED BY GPT
def pair_and_draw_armor_bars(bar_data, w_img, centers_img):
    centers = []
    bar_data.sort(key=lambda x: x[0][0][0])
    for i in range(len(bar_data) - 1):
        rect1, top1, bot1, u1, v1 = bar_data[i]
        rect2, top2, bot2, u2, v2 = bar_data[i + 1]
        # 中心间距筛选
        cx1, cy1 = rect1[0]
        cx2, cy2 = rect2[0]
        #TODO 设为超参
        if abs(cx2 - cx1) > w_img * 0.7:
            continue

        # 取出两根光条的四个“顶角”：top1_pts(2)、top2_pts(2)、bot1_pts(2)、bot2_pts(2)
        # 先把[N×2] 数组合并，后面根据 u 方向决定左右
        top_all = np.vstack((top1, top2))
        bot_all = np.vstack((bot1, bot2))

        # 计算这对光条的“整体” u 向量（取两根光条 u 向量的平均，再归一化）
        # 理论上两根光条几乎平行，所以 u1≈u2。但为了稳健，先求平均：
        u_avg = (u1 + u2) / 2.0
        u_avg = u_avg / (np.linalg.norm(u_avg) + 1e-6)

        # 在 top_all 上做投影，投影值最小为 top_left，最大为 top_right
        projs_top_u = [pt.dot(u_avg) for pt in top_all]
        idxs_top = np.argsort(projs_top_u)
        top_left = top_all[idxs_top[0]]
        top_right = top_all[idxs_top[-1]]

        # 在 bot_all 上做投影，投影值最小为 bottom_left，最大为 bottom_right
        projs_bot_u = [pt.dot(u_avg) for pt in bot_all]
        idxs_bot = np.argsort(projs_bot_u)
        bottom_left = bot_all[idxs_bot[0]]
        bottom_right = bot_all[idxs_bot[-1]]

        # 四个装甲顶点：p1=top_left, p2=bottom_right, p3=top_right, p4=bottom_left
        p1 = np.array(top_left, dtype=np.float32)
        p2 = np.array(bottom_right, dtype=np.float32)
        p3 = np.array(top_right, dtype=np.float32)
        p4 = np.array(bottom_left, dtype=np.float32)

        # 画四边形轮廓（蓝色）
        cv2.line(centers_img, tuple(p1.astype(int)), tuple(p3.astype(int)), (255, 255, 0), 1)
        cv2.line(centers_img, tuple(p3.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 1)
        cv2.line(centers_img, tuple(p2.astype(int)), tuple(p4.astype(int)), (255, 255, 0), 1)
        cv2.line(centers_img, tuple(p4.astype(int)), tuple(p1.astype(int)), (255, 255, 0), 1)
        # 画对角线 (p1->p2) 和 (p3->p4)（绿色）
        cv2.line(centers_img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 1)
        cv2.line(centers_img, tuple(p3.astype(int)), tuple(p4.astype(int)), (255, 255, 0), 1)

        # 计算两条对角线交点
        A = np.array([
            [p2[0] - p1[0], -(p4[0] - p3[0])],
            [p2[1] - p1[1], -(p4[1] - p3[1])]
        ], dtype=np.float32)
        b = np.array([p3[0] - p1[0], p3[1] - p1[1]], dtype=np.float32)

        if abs(np.linalg.det(A)) < 1e-6:
            # 近似平行则用四顶点平均
            center = (
                (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0,
                (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
            )
        else:
            t, _ = np.linalg.solve(A, b)
            inter = p1 + t * (p2 - p1)
            center = (float(inter[0]), float(inter[1]))

        centers.append(center)
        # 绘制中心点
        cv2.circle(centers_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

    return centers