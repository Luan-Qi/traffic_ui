import math
import os

from haversine import haversine, Unit
import cv2
import numpy as np
global point1, point2


class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """

    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape == (4, 2), "Need (4,2) input array"
        assert lonlat_array.shape == (4, 2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array), np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array), np.float32(pixel_array))

    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1, 2)
        assert pixel.shape[1] == 2, "Need (N,2) input array"
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
        lonlat = np.dot(self.M, pixel.T)

        return (lonlat[:2, :] / lonlat[2, :]).T

    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1, 2)
        assert lonlat.shape[1] == 2, "Need (N,2) input array"
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0], 1))], axis=1)
        pixel = np.dot(self.invM, lonlat.T)

        return (pixel[:2, :] / pixel[2, :]).T


class SpeedEstimate:
    def __init__(self):
        # 配置相机画面与地图的映射点，需要根据自己镜头和地图上的点重新配置
        quad_coords = {
            "lonlat": np.array([
                [30.221866, 120.287402],  # top left
                [30.221527, 120.287632],  # top right
                [30.222098, 120.285806],  # bottom left
                [30.221805, 120.285748]  # bottom right
            ]),
            "pixel": np.array([
                [196, 129],  # top left
                [337, 111],  # top right
                [12, 513],  # bottom left
                [530, 516]  # bottom right
            ])
        }

        self.pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    def pixel2lonlat(self,x, y):
        # 像素坐标转为经纬度
        return self.pm.pixel_to_lonlat((x, y))[0]

    def pixelDistance(self, pa_x, pa_y, pb_x, pb_y):
        # 相机画面两点在地图上实际的距离

        lonlat_a = self.pm.pixel_to_lonlat((pa_x, pa_y))
        lonlat_b = self.pm.pixel_to_lonlat((pb_x, pb_y))

        lonlat_a = tuple(lonlat_a[0])
        lonlat_b = tuple(lonlat_b[0])

        return haversine(lonlat_a, lonlat_b, unit='m')


def splicing_csvdata(list1, list2, list3, list4, list5):
    temp = []
    temp.append(list1)
    temp = temp + list2
    temp = temp + list3
    temp.append(list4)
    temp.append(list5)
    return temp


# 计算数组插值
def splicing_csvdata2(list1, list2):  # , list3, list4, list5
    temp = []
    temp.append(list1)
    temp = temp + list2
    # temp = temp + list3
    # temp = temp + list4
    # temp = temp + list5
    return temp


# 计算数组插值
def splicing_csvdata4(list1, list2, list3, list4, list5):
    temp = []
    temp.append(list1)
    temp = temp + list2
    temp = temp + list3
    temp = temp + list4
    temp = temp + list5


def splicing_csvdata5(list1, list2, list3, list4, list5):
    temp = []
    temp.append(list1)
    # temp = temp + list2
    # temp = temp + list3
    temp.append(list2)
    temp.append(list3)
    temp.append(list4)
    temp.append(list5)
    return temp


def splicing_csvdata7(list1, list2, list3, list4, list5, list6, list7):
    temp = []
    temp.append(list1)
    # temp = temp + list2
    # temp = temp + list3
    temp.append(list2)
    temp.append(list3)
    temp.append(list4)
    temp.append(list5)
    temp.append(list6)
    temp.append(list7)
    return temp

def splicing_csvdata(list1, list2, list3, list4, list5, list6, list7, list8, list9,list10,list11,list12,list13):
    temp = []
    temp.append(list1)
    temp.append(list2)
    temp.append(list3)
    temp.append(list4)
    temp.append(list5)
    temp.append(list6)
    temp.append(list7)
    temp.append(list8)
    temp.append(list9)
    temp.append(list10)
    temp.append(list11)
    temp.append(list12)
    temp.append(list13)
    return temp


def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def frames_to_timecode(framerate, frames):
    """
    视频 通过视频帧转换成时间
    :param framerate: 视频帧率
    :param frames: 当前视频帧数
    :return:时间（00:00:01:01）
    """
    return '{0:02d}:{1:02d}:{2:02d}:{3:02d}'.format(int(frames / (3600 * framerate)),
                                                    int(frames / (60 * framerate) % 60),
                                                    int(frames / framerate % 60),
                                                    int(frames % framerate))
    # return '{0:02d}:{1:02d}:{2:02d}'.format(int(frames / (3600 * framerate)),
    #                                         int(frames / (60 * framerate) % 60),
    #                                         int(frames / framerate % 60))


# global record1, record2
# record1 = []
# record2 = []

def lanemark(img):
  def on_mouse(event, x, y, flags, param):
    global  point1, point2
    img2 = img.copy()
    # 左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
      point1 = (x, y)
      record1.append(point1)
      cv2.circle(img2, point1, 10, (0, 255, 0), 2)
      for i in range(np.array(record2).shape[0]):
        cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)  # 画之前划过的线
      cv2.imshow('image', img2)
    # 按住左键拖曳
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
      for i in range(np.array(record2).shape[0]):
        cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)
      cv2.line(img2, point1, (x, y), (255, 0, 0), 2)  # 显示拖拽时上一次左键点击和点钱拖拽点间的连线
      cv2.imshow('image', img2)
    # 左键释放
    elif event == cv2.EVENT_LBUTTONUP:
      point2 = (x, y)
      record2.append(point2)
      for i in range(np.array(record1).shape[0]):
        cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)

      cv2.imshow('image', img2)
      # min_x = min(point1[0], point2[0])
      # min_y = min(point1[1], point2[1])
      # width = abs(point1[0] - point2[0])
      # height = abs(point1[1] - point2[1])
      # cut_img = img[min_y:min_y + height, min_x:min_x + width]
      #cv2.imwrite('lena3.jpg', cut_img)
    # 中键点击（撤销画的上一条线）
    elif event == cv2.EVENT_MBUTTONDOWN:
      record1.pop(np.array(record1).shape[0] - 1)
      record2.pop(np.array(record2).shape[0] - 1)
      for i in range(np.array(record1).shape[0]):
        cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)
      cv2.imshow('image', img2)


  WIDTH = cv2.CAP_PROP_FRAME_WIDTH
  HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

  record = []
  record1 = []
  record2 = []
  cv2.namedWindow('image')
  cv2.setMouseCallback('image', on_mouse)
  cv2.imshow('image', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  record.append(record1)
  record.append(record2)

  if record ==[[],[]]:
    record = ' '
  else:
      print('The Location has been marked successfully ')
      print('lane(location)', record)

  return record


# 获取直线起始，结束点
def get_line_se(img, line):
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = line
    left_y = int((-x * vy / vx) + y)
    right_y = int(((cols - x) * vy / vx) + y)
    start_point = (0, left_y)
    end_point = (right_y, cols - 1)
    return start_point, end_point

# 获取直线 与 点的垂足
def get_foot(start_point, end_point, point_a):
    start_x, start_y = start_point
    end_x, end_y = end_point
    pa_x, pa_y = point_a

    p_foot = [0, 0]
    if start_point[0] == end_point[0]:
        p_foot[0] = start_point[0]
        p_foot[1] = point_a[1]
        return p_foot

    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = int((b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b))
    p_foot[1] = int((a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b))

    return p_foot


# 计算点到直线距离
def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0]
    line_s_y = line[1]
    line_e_x = line[2]
    line_e_y = line[3]
    # 若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    # 若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    # 斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    # 截距
    b = line_s_y - k * line_s_x
    # 带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return int(dis)

def location2kb(location):
    start = location[0]
    end = location[1]
    K =[]
    B =[]
    for i in range(len(location[0])):
        x1 = start[i][0]
        y1 = start[i][1]
        x2 = end[i][0]
        y2 = end[i][1]
        k, b = get_kb(x1,y1,x2,y2)
        K.append(k)
        B.append(b)

    return [K, B]


def get_kb(x1,y1,x2,y2):
    if y1 == y2:
        k = 0
        b = y1   # y = b
        return k, b
    if x1 == x2:
        k = ''
        b = x1  # x = b
        return k, b

    # 斜率
    k = (y1 - y2) / (x1 - x2)
    # 截距
    b = y1 - k * x1

    return k, b


# 计算两直线交点
def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def get_line_cross_point(line1, line2):
    # x1y1x2y2
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    # print(x, y)
    return int(x), int(y)


# 计算所要求的所有交点，
def lane_cross_ew(speedline, lane):
    cross = []
    # temp_cross = [] * np.array(lane[0]).shape[0]
    for i in range(np.array(lane[0]).shape[0]):
        temp = []
        for j in range(np.array(speedline[0]).shape[0]):
            temp_speed = speedline[0][j] + speedline[1][j]
            temp_lane = lane[0][i] + lane[1][i]
            temp_cross = get_line_cross_point(temp_speed, temp_lane)
            temp.append(temp_cross)
        cross.append(temp)
    cross.sort(key = lambda cross : cross[0][1])   # 根据每个二维列表的第一个元素的第二个值进行升序排序。
    return cross

def lane_cross_ns(speedline, lane):
    cross = []
    # temp_cross = [] * np.array(lane[0]).shape[0]
    for i in range(np.array(lane[0]).shape[0]):
        temp = []
        for j in range(np.array(speedline[0]).shape[0]):
            temp_speed = speedline[0][j] + speedline[1][j]
            temp_lane = lane[0][i] + lane[1][i]
            temp_cross = get_line_cross_point(temp_speed, temp_lane)
            temp.append(temp_cross)
        cross.append(temp)
    cross.sort(key = lambda cross : cross[0][0])  # 根据每个二维列表的第一个元素的第一个值进行升序排序。
    return cross

def lane_cross(speedline, lane):
    cross = []
    # temp_cross = [] * np.array(lane[0]).shape[0]
    for i in range(np.array(lane[0]).shape[0]):
        temp = []
        for j in range(np.array(speedline[0]).shape[0]):
            temp_speed = speedline[0][j] + speedline[1][j]
            temp_lane = lane[0][i] + lane[1][i]
            temp_cross = get_line_cross_point(temp_speed, temp_lane)
            temp.append(temp_cross)
        cross.append(temp)
    cross.sort(key = lambda cross : cross[0][0])  # 根据每个二维列表的第一个元素的第一个值进行升序排序。
    return cross


# 返回所需的每车道中点值 两两一组，分别是每条车道线与speedline的 两个交点 [528, 537, 547, 560, 571, 602, 614, 628, 646, 663]
def midpoint(data):
    remid = [[] for _ in range(2)]
    for j in range(2):
        for i in range(np.array(data).shape[0] - 1):
            remid[j].append((data[i][j][1] + data[i + 1][j][1]) // 2)

    return remid


# 计算车道线在y_stable时的像素点值，并从小到大排列（从左到右）,还需输入车道线数
def calculate_linemark(data, kmeans_k, y_stable):
    x = []
    y = []
    x_lane = []
    for i in range(kmeans_k + 1):
        for j in range(2):
            x.append(data[j][i][0])
            y.append(data[j][i][1])
        fit = np.polyfit(y, x, 1)
        fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b
        x_stable = int(fit_fn(y_stable))
        x = []
        y = []
        x_lane.append(x_stable)

    x_lane.sort()
    return x_lane


def calculate_ylane(img):
    data = lanemark(img)
    datasize = np.array(data[0]).shape[0]
    print('datasize', datasize)

    while datasize != 1:
        print('Wrong!! Format error! \n')
        print('Please try again \n')
        data = lanemark(img)
        datasize = np.array(data[0]).shape[0]
        print('datasize', datasize)

    ylane = (data[0][0][1] + data[1][0][1]) // 2

    return ylane


def calculate_speedlane(img):
    data = lanemark(img)
    if data != ' ':
        datasize = np.array(data[0]).shape[0]
        # print('datasize', datasize)


        while datasize > 1:
            print('Wrong!! Format error! \n')
            print('Only one line is need!Please try again \n')
            data = lanemark(img)
            datasize = np.array(data[0]).shape[0]

        if datasize ==1:
            print('The speed lane has been marked successfully ')
            print('speedlane(speed_lane)', data)

    return data


def calculate_stoplane(img):
    stop = lanemark(img)
    datasize = np.array(stop[0]).shape[0]
    # print('datasize', datasize)

    while datasize != 1:
        print('Wrong!! Format error! \n')
        print('Please try again \n')
        stop = lanemark(img)
        datasize = np.array(stop[0]).shape[0]
    return stop


def roi_mask(img, vertices):  # img是输入的图像，vertices是兴趣区的四个点的坐标（三维的数组）
    mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count  # 如果 channel_count=3,则为(255,255,255)
    else:
        mask_color = 255
    for i in range(len(vertices)):
        a = vertices[i]
        vertice = np.array(a)
        vertice = vertice.reshape((-1, 1, 2))

        # 在原始图像上绘制多边形
        cv2.fillPoly(mask, [vertice], mask_color)  # 使用白色填充多边形，形成蒙板
    masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
    # cv2.imwrite('zone.jpg',  masked_img)
    # cv2.imshow('mask', masked_img)
    # cv2.waitKey(0)
    return masked_img


def arraydet(arr, fpstime, maxstopfps):
    arr = np.array(arr)
    det = []
    for i in range(arr.shape[0] - 1):
        if abs(fpstime[arr[i]] - fpstime[arr[i + 1]]) <= maxstopfps:
            det.append(abs(fpstime[arr[i]] - fpstime[arr[i + 1]]))
    det_num = np.mean(det)
    return det_num


def timeoccup(arr, fpstime, maxlookfps):
    arr = np.array(arr)
    det = 0
    for j in range(arr.shape[0] - 1):
        if abs(fpstime[arr[arr.shape[0] - 1]] - fpstime[arr[j]]) <= maxlookfps:
            det = det + fpstime[arr[j]]
    return det


def counter_vehicles(outputs, polygon_mask, counter_recording, counter, list_overlapping):
    for each_output in outputs:
        x1, y1, x2, y2, track_id, cls = each_output
        if track_id not in counter_recording:
            # print(polygon_mask[y1, x1][0])
            if polygon_mask[y1, x1][0] != 0:
                if track_id not in list_overlapping:
                    list_overlapping[track_id] = [polygon_mask[y1, x1][0]]
                else:
                    if list_overlapping[track_id][-1] != polygon_mask[y1, x1]:
                        list_overlapping[track_id].append(polygon_mask[y1, x1][0])
                        if len(list_overlapping[track_id]) == 2:
                            counter_index = [list_overlapping[track_id][0], list_overlapping[track_id][-1]]
                            counter[counter_index[0] - 1][counter_index[1] - 1][cls] += 1
                            counter_recording.append(track_id)
    for id in counter_recording:
        is_found = False
        for _, _, _, _, bbox_id, _ in outputs:
            if bbox_id == id:
                is_found = True
        if not is_found:
            counter_recording.remove(id)
            del list_overlapping[id]

    return counter_recording, counter, list_overlapping


# def counter_vehicles(outputs, counter_recording, counter, list_overlapping):
#     for each_output in outputs:
#         x1, y1, x2, y2, track_id, cls = each_output
#         if track_id not in counter_recording:
#             if track_id not in list_overlapping:
#                 list_overlapping[track_id] = [polygon_mask[y1, x1][0]]
#             else:
#                 if list_overlapping[track_id][-1] != polygon_mask[y1, x1]:
#                     list_overlapping[track_id].append(polygon_mask[y1, x1][0])
#                     if len(list_overlapping[track_id]) == 2:
#                         counter_index = [list_overlapping[track_id][0], list_overlapping[track_id][-1]]
#                         counter[counter_index[0] - 1][counter_index[1] - 1][cls] += 1
#                         counter_recording.append(track_id)
#     for id in counter_recording:
#         is_found = False
#         for _, _, _, _, bbox_id, _ in outputs:
#             if bbox_id == id:
#                 is_found = True
#         if not is_found:
#             counter_recording.remove(id)
#             del list_overlapping[id]
#
#     return counter_recording, counter, list_overlapping

def estimateSpeed_drawlines(location1, location2, scale, fps):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    if d_pixels == 0:
        v_x = 0
        v_y = 0
    elif location2[0] - location1[0] ==0:
        v_x = 0
        v_y = 3.6 * (location2[1] - location1[1]) * scale * fps
    elif location2[1] - location1[1] ==0:
        v_x = 3.6 * (location2[0] - location1[0]) * scale * fps
        v_y = 0
    else:
        # if (location2[0] - location1[0]) != 0:
        #     tan_aplusb = (location2[1] - location1[1]) / (location2[0] - location1[0])
        #     tan_a = lane_k
        #     tan_b = (tan_aplusb - tan_a) / (1 + tan_a*tan_aplusb)
        #     cos_b = 1 / math.sqrt(1+tan_b*tan_b)
        #     sin_b = tan_b*tan_b / math.sqrt(1+tan_b*tan_b)
        #
        #     # carWidht = 4
        #     # ppm = 150 / carWidht  # pixel width/car width  #  pix_wid small then h high
        #     # d_meters = d_pixels / ppm
        #     speed = 3.6 * d_pixels * scale * fps # km/h
        #     v_x = speed * cos_b
        #     v_y = speed * sin_b
        # else:
        #     # a + b =90°
        #     tan_a = lane_k
        #     cos_a = 1 / math.sqrt(1+tan_a*tan_a)
        #     v_x = 0
        #     v_y = 3.6 * d_pixels * scale * fps * cos_a
        tan_v = (location2[1] - location1[1]) / (location2[0] - location1[0])
        cos_v = (location2[0] - location1[0]) / d_pixels
        sin_v = (location2[1] - location1[1]) / d_pixels
        speed = 3.6 * d_pixels * scale * fps  # km/h
        v_x = speed * cos_v
        v_y = speed * sin_v
    return v_x, v_y




def estimateSpeed(location1, location2, scale, fps,dir_id,dir):
    lane_k = dir[dir_id]['Line0']['L1']['fit'][0]
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    if d_pixels == 0:
        v_x = 0
        v_y = 0
    else:
        if (location2[0] - location1[0]) != 0:
            tan_aplusb = (location2[1] - location1[1]) / (location2[0] - location1[0])
            tan_a = lane_k
            tan_b = (tan_aplusb - tan_a) / (1 + tan_a*tan_aplusb)
            cos_b = 1 / math.sqrt(1+tan_b*tan_b)
            sin_b = tan_b*tan_b / math.sqrt(1+tan_b*tan_b)

            # carWidht = 4
            # ppm = 150 / carWidht  # pixel width/car width  #  pix_wid small then h high
            # d_meters = d_pixels / ppm
            speed = 3.6 * d_pixels * scale * fps
            v_x = speed * cos_b
            v_y = speed * sin_b
        else:
            # a + b =90°
            tan_a = lane_k
            cos_a = 1 / math.sqrt(1+tan_a*tan_a)
            v_x = 0
            v_y = 3.6 * d_pixels * scale * fps * cos_a
    return v_x, v_y

def estimate_a(v1, v2, pos1, pos2, scale):
    x = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[0]+pos2[0])**2) # pixel
    x = x * scale
    a_x = (v2**2 - v1**2) / (2 * x * 3.6*3.6)
    return a_x


def draw_counter(im0, counter, names, direction):
    title_txt = 'Direction    '
    for i in names:
        title_txt = title_txt + str(i) + ' '
    # cv2.rectangle(im0, (0, 0), (110*len(names), 90*len(direction)), (255, 255, 255), thickness=-1)
    cv2.putText(im0, title_txt, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

    # blk = np.zeros(im0.shape, np.uint8)
    # cv2.rectangle(blk, (0, 0), (110*len(names), 90*len(direction)), (255, 0, 0), thickness=-1)
    # im0 = cv2.addWeighted(im0, 1.0, blk, 0.5, 1)
    for num, each_import in enumerate(counter):
        for num1, each_export in enumerate(each_import):
            direction_txt = '%s-%s ' % (direction[num], direction[num1])
            cv2.putText(im0, direction_txt, (10, ((num + 1) * 80 + num1 * 20) - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 0), 2)
            counter_txt = ''
            for num2, each_class in enumerate(each_export):
                counter_txt = counter_txt + str(each_class) + '   '
            cv2.putText(im0, counter_txt, (190, ((num + 1) * 80 + num1 * 20) - 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 0), 2)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, classes2, identities=None, last_ids_info={}, offset=(0, 0)):
    this_ids_info = last_ids_info
    for i, box in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        if id in this_ids_info:
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar

            color = compute_color_for_labels(int(classes2[i] * 100))
            label = '%d %s' % (id, cls_names[i])
            # if id in this_ids_info and this_ids_info[id]['speed'][0] != 0 and this_ids_info[id]['speed'][1] != 0 and cls_names[i] == 'car':
            if id in this_ids_info :
                if this_ids_info[id]['speed'][0] is not None:
                    # if this_ids_info[id]['speed'][0] != 0 and this_ids_info[id]['speed'][1] != 0:
                    vx = this_ids_info[id]['speed'][0]
                    vy = this_ids_info[id]['speed'][1]
                    v = math.sqrt(vx*vx + vy*vy)
                    speed = round(v, 1)
                    label = '%d %s %s km/h' % (id, cls_names[i], speed)
            # label = '%d %s' % (id, '')
            # label +='%'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0] # 2,2
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # 3
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 8)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)

    return img


if __name__ == '__main__':
  # img = cv2.imread('img/1.jpg')
  # lanemark(img)
  k,b = get_kb(10,100,10,200)

