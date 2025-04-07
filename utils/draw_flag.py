import cv2
import numpy as np

def draw_line(img):
    start_points = []#开始点
    stop_points = []#结束点
    def draw(event, x, y, flags, param):
        img2 = img.copy()
        # 左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
          print([x, y])
          start_points.append([x, y])
          cv2.circle(img2, [x, y], 5, (0, 255, 0), -2)
          for i in range(len(stop_points)):
            cv2.line(img2, start_points[i], stop_points[i], (0, 0, 255), 3)
          cv2.imshow('image', img2)
        # 按住左键拖曳
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            for i in range(len(stop_points)):
                cv2.line(img2, start_points[i], stop_points[i], (0, 0, 255), 3)
            cv2.line(img2, start_points[-1], (x, y), (255, 0, 0), 3)
            cv2.imshow('image', img2)
        # 左键释放
        elif event == cv2.EVENT_LBUTTONUP:
            print([x, y])
            stop_points.append([x, y])
            for i in range(np.array(start_points).shape[0]):
                cv2.line(img2, start_points[i], stop_points[i], (0, 0, 255), 3)
            cv2.imshow('image', img2)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return start_points, stop_points

def draw_lane(img):
    #画车道线
    print('Please mark the lane!')
    lanes_x1y1, lanes_x2y2 = draw_line(img)
    for i in range(len(lanes_x1y1)):
        cv2.line(img, lanes_x1y1[i], lanes_x2y2[i], (0, 0, 255), 3)
    lane_xyxy = np.concatenate((lanes_x1y1, lanes_x2y2),axis=1)
    return lane_xyxy

def draw_speed_line(img):
    # 画开始线
    print('Please mark the speed line!')
    speed_line_x1y1, speed_line_x2y2 = draw_line(img)
    cv2.line(img, speed_line_x1y1[0], speed_line_x2y2[0], (0, 0, 255), 3)
    speed_line_xyxy = np.concatenate((speed_line_x1y1, speed_line_x2y2), axis=1)
    return speed_line_xyxy[0]

def draw_stop_line(img):
    # 画停止线
    print('Please mark the stop line!')
    stop_line_x1y1, stop_line_x2y2 = draw_line(img)
    cv2.line(img, stop_line_x1y1[0], stop_line_x2y2[0], (0, 0, 255), 3)
    stop_line_xyxy = np.concatenate((stop_line_x1y1, stop_line_x2y2), axis=1)
    return stop_line_xyxy[0]

if __name__ == '__main__':
    img = cv2.imread('../img/2.jpg')
    print(img)
    print(draw_lane(img))
    print(draw_speed_line(img))
    print(draw_stop_line(img))