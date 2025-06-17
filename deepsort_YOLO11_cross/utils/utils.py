import cv2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def draw_cross_lines(frame, dir):
    # 画线
    c = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红黄绿蓝
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 8:  # 画有交点的停止线
            points = dir[dir_id]['stop']['points']
            cv2.line(frame, points[0], points[1], c[dir_id], 3)
            for key in dir[dir_id].keys():
                if key.startswith('Line'):
                    point_s = dir[dir_id][key]['L1']['points'][0]
                    point_e = dir[dir_id][key]['L1']['points'][1]
                    cv2.line(frame, point_s, point_e, c[dir_id], 3)

    # cv2.imwrite('output_image_track.jpg', frame)

    return frame