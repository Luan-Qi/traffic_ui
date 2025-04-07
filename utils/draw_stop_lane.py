import cv2
import numpy as np
import itertools
import math

# from unet.lanes_fit import find_intersection




# 计算两点之间的距离
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 计算每个点到其他所有点中的最小距离
def min_distance_to_other_points(point, points):
    min_distance = float('inf')
    for p in points:
        if p[0] != point[0] and p[1] != point[1]:
            dist = distance(point, p)
            if dist < min_distance:
                min_distance = dist
    return min_distance

# 计算平均最小距离
def average_min_distance(points):
    min_distances = []
    for point in points:
        min_dist = min_distance_to_other_points(point, points)
        min_distances.append(min_dist)
    return sum(min_distances) / len(min_distances)

def draw_cross_lines(frame, dir):
    # 画线
    c = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红黄绿蓝
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 3:  # 画有交点的停止线
            points = dir[dir_id]['stop']['points']
            cv2.line(frame, points[0], points[1], c[dir_id], 5)
            for line_id in range(len(dir[dir_id]) - 4):
                points_l = dir[dir_id]['Line' + str(line_id)]['L1']['points']
                cv2.line(frame, points_l[0], points_l[1], c[dir_id], 5)

    # cv2.imwrite('output_image_track.jpg', frame)

    return frame


def draw_all_lines(frame, dir,last_frame_info,lights):
    # 画线
    c = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红黄绿蓝 BGR
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 4:  # 画有交点的停止线
            points = dir[dir_id]['stop']['points']
            cv2.line(frame, points[0], points[1], c[dir_id], 10)
            # cv2.imwrite('fit_a_line.jpg', image0)

            for line_id in range(len(dir[dir_id]) - 4):
                # 画实线
                points_s = dir[dir_id]['Line' + str(line_id)]['L1']['points']
                cv2.line(frame, points_s[0], points_s[1], c[dir_id], 10)
                # 画虚线
                if 'L2' in dir[dir_id]['Line' + str(line_id)]:
                    points_x = dir[dir_id]['Line' + str(line_id)]['L2']['points']
                    cv2.line(frame, points_x[0], points_x[1],
                             (c[dir_id][0] / 2, c[dir_id][1] / 2, c[dir_id][2] / 2), 5)

                # 画红绿灯
                if line_id > 0:
                    point1 = dir[dir_id]['Line' + str(line_id-1)]['L1']['points'][0]
                    point2 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0]
                    # if dir_id in lights:
                    #     if line_id in lights[dir_id]:
                    #         if 'light' in lights[dir_id][line_id]:
                    #             light = lights[dir_id][line_id]['light']
                    if dir_id in last_frame_info:
                        if line_id in last_frame_info[dir_id]:
                            if 'light' in last_frame_info[dir_id][line_id]:
                                light = last_frame_info[dir_id][line_id]['light'][-1]
                                color = None
                                if light == 'red':
                                    color = (0,0,255)
                                elif light =='green':
                                    color = (0,255,0)
                                if color is not None:
                                    cv2.line(frame, point1, point2, color, 5)


    # cv2.imwrite('output_image_track.jpg', frame)

    return frame


def draw_road_lines(frame, dir, last_frame_info):
    # 画线
    c = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红黄绿蓝
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 4:  # 画有交点的停止线
            points = dir[dir_id]['stop']['points']
            cv2.line(frame, points[0], points[1], c[dir_id], 5)
            # cv2.imwrite('fit_a_line.jpg', image0)

            for key in dir[dir_id]:
                if key.startswith("Line"):
                    # 画实线
                    points_s = dir[dir_id][key]['L1']['points']
                    cv2.line(frame, points_s[0], points_s[1], c[dir_id], 5)
                    # 画虚线
                    if 'L2' in dir[dir_id][key]:
                        points_x = dir[dir_id][key]['L2']['points']
                        cv2.line(frame, points_x[0], points_x[1],
                                 (c[dir_id][0] / 2, c[dir_id][1] / 2, c[dir_id][2] / 2), 5)

    # cv2.imwrite('l+d.jpg', frame)

    return frame




# def get_intersections(fits_s, fits_l, stop_numbers, d):
#     out = []
#     dirs_ = []
#     for index, fit_s in enumerate(fits_s):
#         def stop(x):
#             return fit_s[1] + fit_s[0] * x
#
#         intersection_points = []
#         dirs = []
#         for index_l, stop_number in enumerate(stop_numbers):
#             if stop_number == index:
#                 fit_l = fits_l[index_l]
#                 def lane(x):
#                     return fit_l[1] + fit_l[0] * x
#
#                 intersection_point = find_intersection(stop, lane)
#                 intersection_points.append(intersection_point)
#                 dir = d[index_l]
#                 dirs.append(dir)
#
#         out.append(intersection_points)
#         dirs_.append(dirs)
#
#     return out, dirs_


# 对路口
# def get_roi(intersections, dirs_, video_width, video_height):
#     # ranges = []
#     zone = []
#     zone1 = []
#     # d = []
#     for intersection_point, dirs in zip(intersections, dirs_):
#         x_min = int(min(intersection[0] for intersection in intersection_point))
#         x_max = int(max(intersection[0] for intersection in intersection_point))
#         y_min = int(min(intersection[1] for intersection in intersection_point))
#         y_max = int(max(intersection[1] for intersection in intersection_point))
#
#         if dirs[0] == 1:
#             x_max = video_width
#         if dirs[0] == 2:
#             y_max = video_height
#         if dirs[0] == 3:
#             x_min = 0
#         if dirs[0] == 4:
#             y_min = 0
#
#         # ranges.append((x_min,x_max,y_min,y_max))
#         # d.append(dirs[0])
#         zone.append([x_min,y_min])
#         zone.append([x_max,y_min])
#         zone1.append([x_max,y_max])
#         zone1.append([x_min,y_max])
#
#
#
#     zone += zone1[::-1]
#
#     return np.array(zone)



def get_roi(dir):
    ranges = []
    width_sum = 0
    for dir_id in range(len(dir)):
        range_dir = []
        width_sum = width_sum + dir[dir_id]['lane_width']
        x2 = dir[dir_id]['stop']['points'][0][0]
        y2 = dir[dir_id]['stop']['points'][0][1]
        xx2 = dir[dir_id]['stop']['points'][1][0]
        yy2 = dir[dir_id]['stop']['points'][1][1]
        for line_id in range(len(dir[dir_id]) - 6):
            if line_id == 0:
                x1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][0]
                y1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][1]
            if line_id == len(dir[dir_id]) - 7:
                xx1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][0]
                yy1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][1]

        x_min = min(x1,xx1,x2,xx2)
        x_max = max(x1,xx1,x2,xx2)
        y_min = min(y1,y2,yy1,yy2)
        y_max = max(y1,y2,yy1,yy2)

        range_dir.append((x_min,y_min))
        range_dir.append((x_max,y_min))
        range_dir.append((x_max,y_max))
        range_dir.append((x_min,y_max))
        ranges.append(range_dir)

    # ranges = np.array(ranges)
    mean_lane = width_sum / len(dir)  # 一个车道宽3m，对应的像素
    scale = 3 / mean_lane

    return ranges, scale


def get_position_id(x,y, dir,roi):
    dir_index =None
    for dir_id in range(len(roi)):
        vertice = roi[dir_id]
        x_min = vertice[0][0]
        y_min = vertice[0][1]
        x_max = vertice[2][0]
        y_max = vertice[2][1]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            dir_index = dir_id

            lane_index = None
            direction = dir[dir_index]['direction']
            line_number = len(dir[dir_index]) - 6
            p_list = []
            for line_id in range(line_number):
                fit = dir[dir_index]['Line' + str(line_id)]['L1']['fit']
                if direction == 1 or direction == 3:
                    def lane(x):
                        return fit[1] + fit[0] * x
                    y_on_line = int(lane(x))
                    p_list.append(y_on_line)
                if direction == 2 or direction == 4:
                    def lane(y):
                        return (y-fit[1]) / fit[0]
                    x_on_line = int(lane(y))
                    p_list.append(x_on_line)

            for lane_id in range(1,line_number):
                if direction == 1 or direction == 3:
                    if p_list[lane_id- 1] <= y < p_list[lane_id] or p_list[lane_id- 1] >= y > p_list[lane_id]:
                        lane_index = lane_id
                        break
                elif direction == 2 or direction == 4:
                    if p_list[lane_id- 1] <= x < p_list[lane_id] or p_list[lane_id- 1] >= x > p_list[lane_id]:
                        lane_index = lane_id
                        break
            if lane_index == None:
                if dir_id == len(dir) -1:
                    # print('this car in not in any lane')
                    dir_index = None
                    lane_index = None
                    direction = None
                else:
                    continue

    if dir_index == None:
        # print('this car in not in any direction or lane')
        dir_index = None
        lane_index = None
        direction = None

    return dir_index, lane_index, direction



def y2id(y,y_l):
    y_l.sort()  # 从上到下排列编号
    judge = []
    id = None
    for index,yy in enumerate(y_l):
        if y > yy:
            judge.append(1)
        else:
            judge.append(0)
            id = index
            break

    return id




def renew_k(ls,ds):
    l_d1 = []
    l_d2 = []
    l_d3 = []
    l_d4 = []
    for l,d in zip(ls, ds):
        if d == 1:
            l_d1.append(l)
        if d == 2:
            l_d2.append(l)
        if d == 3:
            l_d3.append(l)
        if d == 4:
            l_d4.append(l)

    k1 = getmean_k(l_d1)
    k2 = getmean_k(l_d2)
    k3 = getmean_k(l_d3)
    k4 = getmean_k(l_d4)

    for l,d in zip(ls, ds):
        if d == 1:
            l[0] = k1
        if d == 2:
            l[0] = k2
        if d == 3:
            l[0] = k3
        if d == 4:
            l[0] = k4

    return ls

def getmean_k(ls):
    # k = None
    if len(ls) > 0:
        k = 0
        for l in ls:
            k = k + l[0]
        k = k / len(ls)

        return k

