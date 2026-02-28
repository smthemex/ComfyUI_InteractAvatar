import math
import numpy as np
import matplotlib
import cv2


eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return  color
    #return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score, th=0.3):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 40 #4
    # 2 3 右肩  2 6 左肩 2 1 脖子 3 4 右臂 4-5 右肘

    limbSeq = [
        [2, 3],
        [2, 6], 
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10], 
        [10, 11],
        [2, 12], # 10
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17], 
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18]
        ]

    colors = [
        [150, 0, 255],
        [0, 78, 255],
        [0, 150, 255],
        [0, 215, 255],
        [70, 125, 180],
        [0, 255, 0],
        [125, 110, 255],
        [35, 140, 35],
        [170, 255, 0],
        [48, 150, 90], # 10
        [255, 170, 0], 
        [255, 85, 0],
        [255, 0, 0], 
        [205, 20, 140], 
        [140, 0, 140],
        [78, 0, 135],
        [70, 60, 140],
        [85, 0, 255],
        [255, 0, 255]
        ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < th or conf[1] < th:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))
            cv2.line(canvas, (int(Y[0]), int(X[0])), (int(Y[1]), int(X[1])), 
                    alpha_blend_color(colors[i], conf[0] * conf[1]), thickness=stickwidth)

    point_colors = {}
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        color = colors[i]
        for point_idx in limb:
            if point_idx not in point_colors:
                point_colors[point_idx] = color


    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            point_color = point_colors.get(i+1, colors[i])
            
            size = stickwidth 
            pts = np.array([
                [x, y - size],      # 上顶点
                [x + size, y],      # 右顶点
                [x, y + size],      # 下顶点
                [x - size, y]       # 左顶点
            ], np.int32)
            
            # 绘制填充菱形
            cv2.fillPoly(canvas, [pts], alpha_blend_color(point_color, conf))
    canvas = (canvas * 0.9).astype(np.uint8)
    # for i in range(18):
    #     for n in range(len(subset)):
    #         index = int(subset[n][i])
    #         if index == -1:
    #             continue
    #         x, y = candidate[index][0:2]
    #         conf = score[n][i]
    #         x = int(x * W)
    #         y = int(y * H)
    #         cv2.circle(canvas, (int(x), int(y)), stickwidth//2, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas
def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, C = canvas.shape
    stickwidth = 12 #4
    
    # 每根手指的连接顺序
    finger_connections = [
        [0, 1, 2, 3, 4],      # 拇指
        [0, 5, 6, 7, 8],      # 食指
        [0, 9, 10, 11, 12],   # 中指
        [0, 13, 14, 15, 16],  # 无名指
        [0, 17, 18, 19, 20]   # 小指
    ]
    
    left_hand_colors = [
        [235, 0, 120],    
        [155, 0, 230],
        [0, 0, 255],
        [0, 165, 255],
        [0, 255, 165]
    ]
    
    right_hand_colors = [
        [255, 255, 0],  
        [165, 255, 0],
        [0, 255, 0],
        [0, 255, 155],
        [0, 165, 165]
    ]

    for hand_idx, (peaks, scores) in enumerate(zip(all_hand_peaks, all_hand_scores)):
        # 根据手的索引来决定使用哪套颜色
        if hand_idx % 2 == 0:  # 偶数索引使用左手颜色
            current_colors = left_hand_colors
        else:  # 奇数索引使用右手颜色
            current_colors = right_hand_colors
        
        for finger_idx, finger_indices in enumerate(finger_connections):
            color = current_colors[finger_idx % len(current_colors)]
            
            # 绘制手指内部的连线
            for i in range(len(finger_indices) - 1):
                idx1 = finger_indices[i]
                idx2 = finger_indices[i + 1]
                
                if idx1 < len(peaks) and idx2 < len(peaks):
                    x1, y1 = peaks[idx1]
                    x2, y2 = peaks[idx2]
                    s1, s2 = scores[idx1], scores[idx2]
                    
                    x1 = int(x1 * W)
                    y1 = int(y1 * H)
                    x2 = int(x2 * W)
                    y2 = int(y2 * H)
                    
                    score = int(s1 * s2 * 255)
                    
                    if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                        cv2.line(canvas, (x1, y1), (x2, y2), 
                                 color, thickness=stickwidth)
            
            for i in finger_indices:
                if i < len(peaks):
                    x, y = peaks[i]
                    score = scores[i]
                    
                    x = int(x * W)
                    y = int(y * H)
                    score_val = int(score * 255)
                    
                    if x > eps and y > eps:
                        cv2.circle(canvas, (x, y), stickwidth//2, color, thickness=-1)
    
    return canvas
# def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
#     H, W, C = canvas.shape
#     stickwidth = 18 #4
#     edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
#              [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

#     for peaks, scores in zip(all_hand_peaks, all_hand_scores):

#         for ie, e in enumerate(edges):
#             x1, y1 = peaks[e[0]]
#             x2, y2 = peaks[e[1]]
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             x2 = int(x2 * W)
#             y2 = int(y2 * H)
#             score = int(scores[e[0]] * scores[e[1]] * 255)
#             if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
#                 cv2.line(canvas, (x1, y1), (x2, y2), 
#                          matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=stickwidth)

#         for i, keyponit in enumerate(peaks):
#             x, y = keyponit
#             x = int(x * W)
#             y = int(y * H)
#             score = int(scores[i] * 255)
#             if x > eps and y > eps:
#                 cv2.circle(canvas, (x, y), stickwidth//2, (0, 0, score), thickness=-1)
#     return canvas

def draw_facepose(canvas, all_lmks, all_scores, th=0.3):
    return canvas
    # for lmks, scores in zip(all_lmks, all_scores):
    #     for lmk, score in zip(lmks, scores):
    #         x, y = lmk
    #         x = int(x * W)
    #         y = int(y * H)
    #         if score < th:
    #             score = 0
    #         conf = int(score * 255)
    #         if x > eps and y > eps:
    #             cv2.circle(canvas, (x, y), stickwidth, (conf, conf, conf), thickness=-1)
    # return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])

    ########################################### draw hand pose #####################################################
    canvas = draw_handpose(canvas, hands, pose['hands_score'])

    ########################################### draw face pose #####################################################
    canvas = draw_facepose(canvas, faces, pose['faces_score'])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

# def draw_pose(pose, H, W, ref_w=1080):
#     """vis dwpose outputs

#     Args:
#         pose (List): DWposeDetector outputs in dwpose_detector.py
#         H (int): height
#         W (int): width
#         ref_w (int, optional) Defaults to 2160.

#     Returns:
#         np.ndarray: image pixel value in RGB mode
#     """
#     bodies = pose['bodies']
#     faces = pose['faces']
#     hands = pose['hands']
#     candidate = bodies['candidate']
#     subset = bodies['subset']

#     sz = min(H, W)
#     sr = (ref_w / sz) if sz != ref_w else 1

#     ########################################## create zero canvas ##################################################
#     canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

#     ########################################### draw body pose #####################################################
#     canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'], th=0.5)

#     ########################################### draw hand pose #####################################################
#     canvas = draw_handpose(canvas, hands, pose['hands_score'])

#     ########################################### draw face pose #####################################################
#     canvas = draw_facepose(canvas, faces, pose['faces_score'], th=0.5)

#     return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)