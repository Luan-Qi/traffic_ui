import math

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def fit_stop(x, coeffs):
    return coeffs[1] + coeffs[0]*x

def fit_stop_y2x(y, coeffs):
    return (y - coeffs[1]) / coeffs[0]

def fit_lane(x, coeffs):
    # return coeffs[2] + coeffs[1]*x + coeffs[0]*x*x
    return coeffs[1] + coeffs[0]*x

def fit_lane_y2x(y, coeffs):
    return (y - coeffs[1]) / coeffs[0]

def dis(x1, y1, x2, y2):  # 两点间
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def p2l_dis(x,y,fit):  # 点到直线的距离
    b = fit[1]
    k = fit[0]
    return abs(k * x - y + b) / math.sqrt(k ** 2 + 1)

# 定义求解两条曲线交点的函数
def find_intersection(curve1, curve2):
    def equations(vars):
        x, y = vars
        eq1 = curve1(x) - y
        eq2 = curve2(x) - y
        return [eq1, eq2]

    # 使用fsolve函数求解方程组
    intersection = fsolve(equations, (0, 0))
    return intersection

def lane_points_pre(points):
    # 初始最大值和最小值为第一个点的横坐标
    max_x = max(point[1] for point in points)
    min_x = min(point[1] for point in points)
    max_y = max(point[0] for point in points)
    min_y = min(point[0] for point in points)
    filter_points = []
    # 计算最大值和最小值之间的差值
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    value = diff_x / diff_y
    if 1/10 < value < 1:
        unique_y_values = set(y for y,x in points if list(y for y, x in points).count(y) > 1)
        for target_y in unique_y_values:
            # 找出y等于特定值的点
            target_x = [x for y,x in points if y == target_y]
            if len(target_x) > 0:
                x_min = min(target_x)
                x_max = max(target_x)
                x_mean = int((x_min + x_max) / 2)
                # fitted_points = [point for point in points if (x_mean - 1 <= point[0] <= x_mean + 1 and point[1] == target_y)]
                fitted_points = [point for point in points if (point[1] == x_mean and point[0] == target_y)]
                filter_points.extend(fitted_points)

    elif 1 <= value < 10:
        unique_x_values = set(x for y,x in points if list(x for y,x in points).count(x) > 1)
        for target_x in unique_x_values:
            target_y = [y for y,x in points if x == target_x]
            if len(target_y) > 0:
                y_min = min(target_y)
                y_max = max(target_y)
                y_mean = int((y_min + y_max) / 2)
                fitted_points = [point for point in points if
                                 (point[0] == y_mean and point[1] == target_x)]
                filter_points.extend(fitted_points)

    else: filter_points = points

    filter_points = np.array(filter_points)
    return filter_points


def points_pre(clustered_points):  #(输入坐标使图像像素yx坐标，该函数中使用x表示像素纵坐标)
    # 将点展开成一组点
    flat_points = []
    for points in clustered_points:
        flat_points.extend(points)

    filted_points = []
    fit_category = []
    seg_img = np.zeros((1530, 2720, 3), dtype=np.uint8)
    for points in clustered_points:
        # 初始最大值和最小值为第一个点的横坐标
        max_x = max(point[0] for point in points)
        min_x = min(point[0] for point in points)
        max_y = max(point[1] for point in points)
        min_y = min(point[1] for point in points)
        filter_points = []
        # 计算最大值和最小值之间的差值
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        stop_wid = min(diff_x, diff_y)
        if stop_wid >= 3:
            if diff_x <= diff_y:
                unique_y_values = set(y for x, y in points if list(y for x, y in points).count(y) > 1)
                # # 纵坐标相同的值对应的横坐标集合
                # x_values = [x for x, y in points if y in unique_y_values]
                for target_y in unique_y_values:
                    # 找出纵坐标等于特定值的点
                    target_x = [x for x, y in points if y == target_y]
                    if len(target_x) > 0:
                        x_min = min(target_x)
                        x_max = max(target_x)
                        x_mean = int((x_min + x_max) / 2)
                        fitted_points = [point for point in points if
                                         (x_mean - 1 <= point[0] <= x_mean + 1 and point[1] == target_y)]
                        filter_points.extend(fitted_points)
                fit_category.append(0)

            if diff_y < diff_x:
                unique_x_values = set(x for x, y in points if list(x for x, y in points).count(x) > 1)
                for target_x in unique_x_values:
                    # 找出纵坐标等于特定值的点
                    target_y = [y for x, y in points if x == target_x]
                    if len(target_y) > 0:
                        y_min = min(target_y)
                        y_max = max(target_y)
                        y_mean = int((y_min + y_max) / 2)
                        fitted_points = [point for point in points if
                                         (y_mean - 1 <= point[1] <= y_mean + 1 and point[0] == target_x)]
                        filter_points.extend(fitted_points)
                fit_category.append(1)  # 这条线是竖线

            # if diff_x <= diff_y:
            #     # 统计横坐标出现的次数
            #     y_values = [point[0] for point in points]
            #     counter = Counter(y_values)
            #     # 找出出现次数最多的横坐标值
            #     mode_y = counter.most_common(1)[0][0]
            #     # 指定横坐标的范围
            #     y_min = mode_y-1
            #     y_max = mode_y+1
            #     # 去掉不在范围内的点
            #     filter_points = [point for point in points if y_min <= point[0] <= y_max]
            # if diff_y < diff_x:
            #     # 统计横坐标出现的次数
            #     x_values = [point[1] for point in points]
            #     counter = Counter(x_values)
            #     # 找出出现次数最多的坐标值
            #     mode_x = counter.most_common(1)[0][0]
            #     # 指定横坐标的范围
            #     x_min = mode_x-1
            #     x_max = mode_x+1
            #     # 去掉不在范围内的点
            #     filter_points = [point for point in points if x_min <= point[1] <= x_max]
            #     fit_category.append(1)

            points = filter_points
        #     print('filtered')
        # else:
        #     print('no need filter')
        filted_points.append(points)


        # for point in points:
        #     seg_img[point[0],point[1]] = (0,255,0)
        #     # mm = seg_img[point[0], point[1]]
        #
        # cv2.imwrite('select_stop.jpg',seg_img)


    return filted_points,fit_category



def stop_line_cluster_fit(image_s, image0):
    image = image0.copy()
    c = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    gray_image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
    # 找到车道线像素点
    lane_pixels = np.argwhere(gray_image_s > 50)  # 这里假设车道线为白色部分，阈值可根据实际情况调整

    # 使用DBSCAN聚类将像素点聚类
    dbscan = DBSCAN(eps=50, min_samples=30)
    labels = dbscan.fit_predict(lane_pixels)
    # 提取每个类别的像素点
    clustered_points = [lane_pixels[labels == i] for i in range(max(labels) + 1)]  # 坐标为yx

    # # 绘制聚类结果
    # x = [point[0] for point in lane_pixels]
    # y = [point[1] for point in lane_pixels]
    # labels_d = dbscan.labels_
    # plt.scatter(y, x, c=labels_d)
    # plt.title('Data Points Clustering')
    # plt.show()

    filted_points,fit_category = points_pre(clustered_points)   # 坐标yx
    # filted_points = clustered_points


    flat_filted = []
    clustered_points_new = []
    for points in filted_points:
        flat_filted.extend(points)
        clustered_points_new.append(points)

    i = 0
    dir = {}
    cross_size_x_min = 1000000
    cross_size_x_max = 0
    cross_size_y_min = 1000000
    cross_size_y_max = 0
    for points,category in zip(clustered_points_new,fit_category):
        points = np.array(points)
        if len(points) < 400:  # 至少需要3个点才能拟合二次多项式
            continue
        fit = np.polyfit(points[:, 1], points[:, 0], 1)

        if category == 0:
            max_x = max(point[1] for point in points)
            min_x = min(point[1] for point in points)
            y1 = int(fit_stop(max_x, fit))
            y2 = int(fit_stop(min_x, fit))
            x_mean = (max_x + min_x) / 2
            y_mean = (y1 + y2) / 2
            cv2.line(image0, (max_x, y1), (min_x, y2), c[i], 5)
            # cv2.imwrite('fit_a_line.jpg', image)
            points = ((min_x, y2), (max_x, y1))
            # print('hengxian')
        else:
            max_y = max(point[0] for point in points)
            min_y = min(point[0] for point in points)
            x1 = int(fit_stop_y2x(max_y, fit))
            x2 = int(fit_stop_y2x(min_y, fit))
            x_mean = (x1 + x2) / 2
            y_mean = (min_y + max_y) / 2
            cv2.line(image0, (x1, max_y), (x2, min_y), c[i], 5)
            # cv2.imwrite('fit_a_line.jpg', image)
            points = ((x2, min_y), (x1, max_y))
            # print('shuxian')

        k_ver = - 1 / fit[0]
        b_ver = y_mean - k_ver * x_mean
        dir[i] = {'category': category,'stop': {'fit': fit, 'points': points}, 'zhongdian': (x_mean, y_mean),'ver': (k_ver, b_ver)}
        cross_size_x_min = min(points[0][0],points[1][0], cross_size_x_min)
        cross_size_x_max = max(points[0][0],points[1][0], cross_size_x_max)
        cross_size_y_min = min(points[0][1],points[1][1], cross_size_y_min)
        cross_size_y_max = max(points[0][1],points[1][1], cross_size_y_max)
        i = i+1

    cross_size_x = abs(cross_size_x_max - cross_size_x_min)
    cross_size_y = abs(cross_size_y_max - cross_size_y_min)

    # cv2.imwrite('image_stop.jpg', image0)
    return dir,[cross_size_x,cross_size_y]


def scale_coordinates(coordinates, scale_factor):
    scaled_coordinates = []
    for coord in coordinates:
        x = int(coord[0] * scale_factor)
        y = int(coord[1] / scale_factor)
        scaled_coordinates.append((x, y))
    return np.array(scaled_coordinates)


def dir_cluster_fit(mode, image,image0,dir):
    image_C = image0
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 找到车道线像素点
    lane_pixels = np.argwhere(gray_image > 50)

    lane_pixels_for_stop = {}
    for dir_id in range(len(dir)):
        lane_pixels_for_stop[dir_id] = []
    for point in lane_pixels:
        min_dis1 = 10000
        min_dis2 = 10000
        for dir_id in range(len(dir)):
            distance1 = p2l_dis(point[1], point[0], dir[dir_id]['ver'])
            distance2 = dis(point[1], point[0], dir[dir_id]['zhongdian'][0], dir[dir_id]['zhongdian'][1])
            if distance1 < min_dis1 :
                min_dis1 = distance1
                point_belong_stop1 = dir_id
            if distance2 < min_dis2 :
                min_dis2 = distance2
                point_belong_stop2 = dir_id
        if point_belong_stop1 == point_belong_stop2:
            point_belong_stop = point_belong_stop1
            # category = dir[point_belong_stop]['category']
            # if min_dis1 < 300 and category == 1:
            if min_dis1 < 300:
                lane_pixels_for_stop[point_belong_stop].append((point[1], point[0]))
            # elif min_dis1 < 200 and category == 0:
            #     lane_pixels_for_stop[point_belong_stop].append((point[1], point[0]))



    for dir_id, lane_pixels in lane_pixels_for_stop.items():
        if len(lane_pixels) < 30:
            continue
        category = dir[dir_id]['category']
        # if category == 0:
        #     lane_pixels = scale_coordinates(lane_pixels, 30)
        # else:
        #     lane_pixels = scale_coordinates(lane_pixels, 1/30)

        # # 使用DBSCAN聚类将像素点按dir聚类，                          可改进：判断set（labels）的长度与dir的长度是否相等
        # dbscan = DBSCAN(eps=100, min_samples=10)
        # labels = dbscan.fit_predict(lane_pixels)
        #
        # # 提取每个类别的像素点
        # clustered_dirs = [lane_pixels[labels == i] for i in range(max(labels) + 1)]
        #
        # # 绘制聚类结果
        # x = [point[0] for point in lane_pixels]
        # y = [point[1] for point in lane_pixels]
        # plt.scatter(x, y, c=labels)
        # plt.title('Dir Points Clustering' + str(mode))
        # plt.show()

        dir = lane_fit(mode, dir_id, lane_pixels, category, image_C, dir)

    if mode == 1:
        dir = kbs_select(dir, image_C, mode)
        # frame0 = draw_road_lines(image_C, dir, {})
        # cv2.imwrite('output_image.jpg', frame0)
        dir = buchong(dir, image_C)
        # frame0 = draw_road_lines(image_C, dir, {})
        # cv2.imwrite('output_image_b.jpg', frame0)
    else:
        dir = xu_select(dir, image_C)


    return dir


def buchong(dir,image):
    sum_lane = 0
    count_dir_num = 0
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 5:
            direction = dir[dir_id]['direction']
            x2 = dir[dir_id]['stop']['points'][0][0]
            y2 = dir[dir_id]['stop']['points'][0][1]
            xx2 = dir[dir_id]['stop']['points'][1][0]
            yy2 = dir[dir_id]['stop']['points'][1][1]
            stop_lenth = dis(x2,y2,xx2,yy2)
            for line_id in range(len(dir[dir_id])-5):
                if line_id == 0:
                    x1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][0]
                    y1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][1]
                    jiaodian2start = dis(x1,y1,x2,y2)
                if line_id == len(dir[dir_id]) - 6:
                    xx1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][0]
                    yy1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][1]
                    jiaodian2end = dis(xx1, yy1, xx2, yy2)

            each_lane = (stop_lenth - jiaodian2start - jiaodian2end) / (len(dir[dir_id]) - 6)
            if jiaodian2start > each_lane - 10:
                # dir[dir_id]['Line' + str(len(dir[dir_id])-5)] = dir[dir_id]['Line' + str(len(dir[dir_id])-6)]
                for i in range(len(dir[dir_id]) - 5):
                    # now = 'Line' + str(len(dir[dir_id])-5 - i)
                    dir[dir_id]['Line' + str(len(dir[dir_id])-5 - i)] = dir[dir_id]['Line' + str(len(dir[dir_id])-6 - i)]
                del dir[dir_id]['Line0']
                dir[dir_id]['Line0'] = {'type':'single_line'}
                dir[dir_id]['Line0']['L1'] = {'lane_type':'shi'}
                fit_b = y2 - dir[dir_id]['Line1']['L1']['fit'][0] * x2
                dir[dir_id]['Line0']['L1']['fit'] = np.array((dir[dir_id]['Line1']['L1']['fit'][0], fit_b))
                if direction == 1:
                    xs = x2
                    ys = y2
                    xe = image.shape[1] - 1
                    ye = dir[dir_id]['Line0']['L1']['fit'][0] * xe + fit_b
                elif direction == 2:
                    ys = y2
                    xs = x2
                    ye = 0
                    xe = - fit_b / dir[dir_id]['Line0']['L1']['fit'][0]
                elif direction == 3:
                    xs = y2
                    ys = x2
                    ye = 0
                    xe = fit_b
                else:
                    xs = x2
                    ys = y2
                    ye =  image.shape[0] - 1
                    xe = (ye - fit_b) / dir[dir_id]['Line0']['L1']['fit'][0]
                points = ((int(xs),int(ys)),(int(xe),int(ye)))
                dir[dir_id]['Line0']['L1']['points'] = points

            if jiaodian2end > each_lane:
                dir[dir_id]['Line' + str(len(dir[dir_id]) - 5)] = {'type': 'single_line'}
                dir[dir_id]['Line' + str(len(dir[dir_id]) - 6)]['L1'] = {'lane_type': 'shi'}
                fit_b = yy2 - dir[dir_id]['Line0']['L1']['fit'][0] * xx2
                dir[dir_id]['Line' + str(len(dir[dir_id]) - 6)]['L1']['fit'] = np.array((dir[dir_id]['Line1']['L1']['fit'][0], fit_b))

                if direction == 1:
                    xs = xx2
                    ys = yy2
                    xe = image.shape[1] - 1
                    ye = dir[dir_id]['Line0']['L1']['fit'][0] * xe + fit_b
                elif direction == 2:
                    ys = yy2
                    xs = xx2
                    ye = 0
                    xe = - fit_b / dir[dir_id]['Line0']['L1']['fit'][0]
                elif direction == 3:
                    xs = xx2
                    ys = yy2
                    ye = fit_b
                    xe = 0
                else:
                    xs = xx2
                    ys = yy2
                    ye =  image.shape[0] - 1
                    xe = (ye - fit_b) / dir[dir_id]['Line0']['L1']['fit'][0]
                points = ((int(xs),int(ys)),(int(xe),int(ye)))
                dir[dir_id]['Line' + str(len(dir[dir_id])-6)]['L1']['points'] = points

            dir[dir_id]['lane_width'] = each_lane
            sum_lane = sum_lane + each_lane
            count_dir_num = count_dir_num + 1
        else:
            print('只画了一条线，稍后补充')

    if count_dir_num < len(dir):
        mean_lane = sum_lane / count_dir_num
        for dir_id in range(len(dir)):
            if len(dir[dir_id]) == 4:
                direction = dir[dir_id]['direction']
                x2 = dir[dir_id]['stop']['points'][0][0]
                y2 = dir[dir_id]['stop']['points'][0][1]
                xx2 = dir[dir_id]['stop']['points'][1][0]
                yy2 = dir[dir_id]['stop']['points'][1][1]
                x1 = dir[dir_id]['Line0']['L1']['points'][0][0]
                y1 = dir[dir_id]['Line0']['L1']['points'][0][1]
                jiaodian2start = dis(x1, y1, x2, y2)
                jiaodian2end = dis(x1, y1, xx2, yy2)

                if jiaodian2start > mean_lane:
                    dir[dir_id]['Line1'] = dir[dir_id]['Line0']
                    del dir[dir_id]['Line0']
                    dir[dir_id]['Line0'] = {'type': 'single_line'}
                    dir[dir_id]['Line0']['L1'] = {'lane_type': 'shi'}
                    fit_b = y2 - dir[dir_id]['Line1']['L1']['fit'][0] * x2
                    dir[dir_id]['Line0']['L1']['fit'] = np.array((dir[dir_id]['Line1']['L1']['fit'][0], fit_b))
                    if direction == 1:
                        xs = x2
                        ys = y2
                        xe = image.shape[1] - 1
                        ye = dir[dir_id]['Line0']['L1']['fit'][0] * xe + fit_b
                    elif direction == 2:
                        ys = y2
                        xs = x2
                        ye = 0
                        xe = - fit_b / dir[dir_id]['Line0']['L1']['fit'][0]
                    elif direction == 3:
                        xs = y2
                        ys = x2
                        ye = 0
                        xe = fit_b
                    else:
                        xs = x2
                        ys = y2
                        ye = image.shape[0] - 1
                        xe = (ye - fit_b) / dir[dir_id]['Line0']['L1']['fit'][0]
                    points = ((int(xs), int(ys)), (int(xe), int(ye)))
                    dir[dir_id]['Line0']['L1']['points'] = points

                if jiaodian2end > mean_lane:
                    dir[dir_id]['Line' + str(len(dir[dir_id]) - 5)] = {'type': 'single_line'}
                    dir[dir_id]['Line' + str(len(dir[dir_id]) - 6)]['L1'] = {'lane_type': 'shi'}
                    fit_b = yy2 - dir[dir_id]['Line0']['L1']['fit'][0] * xx2
                    dir[dir_id]['Line' + str(len(dir[dir_id]) - 6)]['L1']['fit'] = np.array(
                        (dir[dir_id]['Line1']['L1']['fit'][0], fit_b))

                    if direction == 1:
                        xs = xx2
                        ys = yy2
                        xe = image.shape[1] - 1
                        ye = dir[dir_id]['Line0']['L1']['fit'][0] * xe + fit_b
                    elif direction == 2:
                        ys = yy2
                        xs = xx2
                        ye = 0
                        xe = - fit_b / dir[dir_id]['Line0']['L1']['fit'][0]
                    elif direction == 3:
                        xs = xx2
                        ys = yy2
                        ye = fit_b
                        xe = 0
                    else:
                        xs = xx2
                        ys = yy2
                        ye = image.shape[0] - 1
                        xe = (ye - fit_b) / dir[dir_id]['Line0']['L1']['fit'][0]
                    points = ((int(xs), int(ys)), (int(xe), int(ye)))
                    dir[dir_id]['Line' + str(len(dir[dir_id]) - 6)]['L1']['points'] = points

                dir[dir_id]['lane_width'] = mean_lane

    return dir


def get_out_lane_num(image,image0,dir):
    out = {}
    for dir_id in range(len(dir)):
        direction = dir[dir_id]['direction']
        out[direction] = 1
    image_C = image0
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 找到车道线像素点
    lane_pixels = np.argwhere(gray_image > 50)

    # 使用DBSCAN聚类将像素点按dir聚类，                          可改进：判断set（labels）的长度与dir的长度是否相等
    dbscan = DBSCAN(eps=200, min_samples=50)
    labels = dbscan.fit_predict(lane_pixels)

    # 提取每个类别的像素点
    clustered_dirs = [lane_pixels[labels == i] for i in range(max(labels) + 1)]

    # # 绘制聚类结果
    # x = [point[0] for point in lane_pixels]
    # y = [point[1] for point in lane_pixels]
    # plt.scatter(y, x, c=labels)
    # plt.title('Dir_out Points Clustering')
    # plt.show()

    # 对于某个dir的点
    for dir_points in clustered_dirs:
        if len(dir_points) < 30:
            continue

        # 确定该dir_points对应的dir_id（stop的序号）
        max_x = max(point[1] for point in dir_points)
        min_x = min(point[1] for point in dir_points)
        max_y = max(point[0] for point in dir_points)
        min_y = min(point[0] for point in dir_points)
        x1 = (max_x + min_x) / 2
        y1 = (max_y + min_y) / 2

        min_dir = image0.shape[0]
        for dir_id in range(len(dir)):
            x2 = dir[dir_id]['stop']['points'][0][0]
            y2 = dir[dir_id]['stop']['points'][0][1]
            which_dir = dis(x1,y1,x2,y2)
            if which_dir < min_dir:
                min_dir = which_dir
                dir_index = dir_id

        category = dir[dir_index]['category']
        out = get_num(dir_index, dir_points, category, image_C, dir,out)

    return out

def lane_fit(mode, dir_index, dir_points, category, image,dir):
    max_x = max(point[0] for point in dir_points)
    min_x = min(point[0] for point in dir_points)
    max_y = max(point[1] for point in dir_points)
    min_y = min(point[1] for point in dir_points)
    if category == 0:
        scale = 5
        min_lane_length = (max_x - min_x) / 8
        dir_points_scale = scale_coordinates(dir_points, scale)  # 路段
        dbscan_d = DBSCAN(eps=10, min_samples=5)
        labels_d = dbscan_d.fit_predict(dir_points_scale)
        label_num = set(labels_d)
    else:
        scale = 1/50
        min_lane_length = (max_y - min_y) / 4
        dir_points_scale = scale_coordinates(dir_points, scale)  # 路段
        dbscan_d = DBSCAN(eps=50, min_samples=100)
        labels_d = dbscan_d.fit_predict(dir_points_scale)
        label_num = set(labels_d)



    # # 绘制聚类结果
    # plt.clf()
    # x = [point[0] for point in dir_points]
    # y = [point[1] for point in dir_points]
    # plt.scatter(x, y, c=labels_d)
    # plt.title('Dir' + str(dir_index) + 'Points Clustering' + str(mode))
    # plt.show()

    # 提取每个类别的像素点
    dir_points = np.array(dir_points)
    clustered_points = [dir_points[labels_d == i] for i in range(max(labels_d) + 1)]

    for points_l in clustered_points:
        points_l = list(points_l)
        if len(points_l) > 300:
            points_l = lane_points_pre(points_l)
            fit_l = np.polyfit(points_l[:, 0], points_l[:, 1], 1)
            def lane(x):
                return fit_l[1] + fit_l[0] * x

            max_x_l = max(point[0] for point in points_l)
            min_x_l = min(point[0] for point in points_l)
            max_y_l = max(point[1] for point in points_l)
            min_y_l = min(point[1] for point in points_l)
            dis_line = dis(max_x_l, max_y_l, min_x_l, min_y_l)

            # if dis_line >= min_lane_length:
            if dis_line >= 180:
                fit, stop_line = dir[dir_index]['stop']['fit'],dir[dir_index]['stop']['points']
                def stop(x):
                    return fit[1] + fit[0] * x

                tan_jiajiao = abs((fit_l[0] - fit[0]) / (1 + fit[0] * fit_l[0]))
                if tan_jiajiao > 0:
                    # 求解两条线的交点
                    intersection_point = find_intersection(stop, lane)
                    if 'lines' not in dir[dir_index]:
                        dir[dir_index]['lines'] = {'fits': [], 'end': [], 'start': [],'weight':[]}

                    if category == 0:
                        x1 = min(np.array(stop_line)[0][0], np.array(stop_line)[1][0])
                        x2 = max(np.array(stop_line)[0][0], np.array(stop_line)[1][0])
                        if intersection_point[0] >= (x1 - 3) and intersection_point[0] <= (x2 + 3):
                        # if intersection_point[0] >= (x1 - 200) and intersection_point[0] <= (x2 + 200):
                            min_y_l_draw = min(intersection_point[1], min_y_l)
                            max_y_l_draw = max(intersection_point[1], max_y_l)
                            dir[dir_index]['lines']['fits'].append(fit_l)
                            if mode == 1:
                                dir[dir_index]['lines']['end'].append(int(max_y_l_draw))
                                dir[dir_index]['lines']['start'].append(int(min_y_l_draw))
                                dir[dir_index]['lines']['weight'].append(len(points_l))
                                y = (min_y_l_draw + max_y_l_draw) / 2
                                # if fit[0] * (fit[0] * (y - fit_l[1]) / fit_l[0] - y + fit[1]) > 0:
                                if fit[0] * (y - fit_l[1]) / fit_l[0] - y + fit[1] > 0:
                                    if 'direction' not in dir[dir_index]:
                                        dir[dir_index]['direction'] = 2 # “山”
                                        dir[dir_index]['stop']['points'] = ((np.array(stop_line)[1][0],np.array(stop_line)[1][1]),(np.array(stop_line)[0][0],np.array(stop_line)[0][1]))
                                else:
                                    if 'direction' not in dir[dir_index]:
                                        dir[dir_index]['direction'] = 4 # 倒“山”
                            else:
                                min_dis = p2l_dis(min_y_l,lane(min_y_l),fit)
                                max_dis = p2l_dis(max_y_l,lane(max_y_l),fit)
                                if min_dis < max_dis:
                                    start_y = min_y_l
                                else:
                                    start_y = max_y_l
                                dir[dir_index]['lines']['start'].append(int(start_y))

                            # # 画的lane
                            # x_range_l = np.arange(min_x_l, max_x_l)
                            # y_range_l = fit_lane(x_range_l, fit_l)
                            # draw_points_l = np.column_stack((x_range_l, y_range_l)).astype(np.int32)
                            #
                            # if mode == 1:
                            #     # 在RGB图像上绘制车道线
                            #     for i in range(len(draw_points_l) - 1):
                            #         cv2.line(image, tuple(draw_points_l[i]), tuple(draw_points_l[i + 1]),
                            #                  (255, 0, 0), 5)
                            #     cv2.imwrite('image_lane_shi.jpg', image)
                            # else:
                            #     # 在RGB图像上绘制车道线
                            #     for i in range(len(draw_points_l) - 1):
                            #         cv2.line(image, tuple(draw_points_l[i]), tuple(draw_points_l[i + 1]),
                            #                  (0, 255, 0), 5)
                            #     cv2.imwrite('image_lane_xu.jpg', image)


                    elif category == 1:
                        y1 = min(stop_line[0][1], stop_line[1][1])
                        y2 = max(stop_line[0][1], stop_line[1][1])
                        if intersection_point[1] >= y1 - 10 and intersection_point[1] <= y2 + 15:
                            min_x_l_draw = min(intersection_point[0], min_x_l)
                            max_x_l_draw = max(intersection_point[0], max_x_l)
                            dir[dir_index]['lines']['fits'].append(fit_l)
                            if mode == 1:
                                dir[dir_index]['lines']['end'].append(int(max_x_l_draw))
                                dir[dir_index]['lines']['start'].append(int(min_x_l_draw))
                                dir[dir_index]['lines']['weight'].append(len(points_l))
                                x = (min_x_l_draw + max_x_l_draw) / 2
                                if fit[0] * (fit[0] * x - lane(x) + fit[1]) > 0:
                                    if 'direction' not in dir[dir_index]:
                                        dir[dir_index]['direction'] = 1 # “E”
                                        dir[dir_index]['stop']['points'] = (
                                        (np.array(stop_line)[1][0], np.array(stop_line)[1][1]),
                                        (np.array(stop_line)[0][0], np.array(stop_line)[0][1]))
                                else:
                                    if 'direction' not in dir[dir_index]:
                                        dir[dir_index]['direction'] = 3  # 反“E”
                            else:
                                min_dis = p2l_dis(min_x_l,lane(min_x_l),fit)
                                max_dis = p2l_dis(max_x_l,lane(max_x_l),fit)
                                if min_dis < max_dis:
                                    start_x = min_x_l
                                else:
                                    start_x = max_x_l
                                dir[dir_index]['lines']['start'].append(int(start_x))

                            # # 画的lane
                            # x_range_l = np.arange(min_x_l, max_x_l)
                            # y_range_l = fit_lane(x_range_l, fit_l)
                            # draw_points_l = np.column_stack((x_range_l, y_range_l)).astype(np.int32)
                            #
                            # if mode == 1:
                            #     for i in range(len(draw_points_l) - 1):
                            #         cv2.line(image, tuple(draw_points_l[i]), tuple(draw_points_l[i + 1]), (255, 0, 0), 5)
                            #     cv2.imwrite('image_lane_shi.jpg', image)
                            # else:
                            #     for i in range(len(draw_points_l) - 1):
                            #         cv2.line(image, tuple(draw_points_l[i]), tuple(draw_points_l[i + 1]), (0, 255, 0), 5)
                            #     cv2.imwrite('image_lane_xu.jpg', image)


    return dir

def xu_select(dir, image0):
    for dir_id in range(len(dir)):
        dir[dir_id].pop('category')
        direction = dir[dir_id]['direction']

        for line_id in range(len(dir[dir_id])):
            if 0 < line_id < len(dir[dir_id]) - 5:
                point1 = dir[dir_id]['Line' + str(line_id)]['L1']['fit']
                min_dis = image0.shape[0]
                for xu_id in range(len(dir[dir_id]['lines']['fits'])):
                    point2 = dir[dir_id]['lines']['fits'][xu_id]
                    dis = euclidean_distance(point1, point2)
                    if dis < min_dis:
                        min_dis = dis
                        index = xu_id

                fit_s = dir[dir_id]['Line' + str(line_id)]['L1']['fit']

                def lane1(x):
                    return fit_s[1] + fit_s[0] * x

                fit_x = dir[dir_id]['lines']['fits'][index]


                if 0 < line_id < len(dir[dir_id]) - 4:
                    point1 = dir[dir_id]['Line' + str(line_id)]['L1']['fit']
                    min_dis = image0.shape[0]
                    for xu_id in range(len(dir[dir_id]['lines']['fits'])):
                        point2 = dir[dir_id]['lines']['fits'][xu_id]
                        dis = euclidean_distance(point1, point2)
                        if dis < min_dis:
                            min_dis = dis
                            index = xu_id

                    fit_s = dir[dir_id]['Line' + str(line_id)]['L1']['fit']
                    def lane1(x):
                        return fit_s[1] + fit_s[0] * x

                    fit_x = dir[dir_id]['lines']['fits'][index]

                    if direction == 1:
                        x1 = dir[dir_id]['lines']['start'][index]
                        y1 = int(lane1(x1))
                        x2 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][0]
                        y2 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][1]
                        dir[dir_id]['Line' + str(line_id)]['L1']['points'] = (dir[dir_id]['Line' + str(line_id)]['L1']['points'][0],(x1,y1))
                    elif direction == 2:
                        x1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][0]
                        y1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][1]
                        y2 = dir[dir_id]['lines']['start'][index]
                        x2 = int(fit_lane_y2x(y2, fit_x))
                        dir[dir_id]['Line' + str(line_id)]['L1']['points'] = ((x2,y2),dir[dir_id]['Line' + str(line_id)]['L1']['points'][1])
                    elif direction == 3:
                        x1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][0]
                        y1 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][0][1]
                        x2 = dir[dir_id]['lines']['start'][index]
                        y2 = int(lane1(x1))
                        dir[dir_id]['Line' + str(line_id)]['L1']['points'] = (
                        (x2,y2), dir[dir_id]['Line' + str(line_id)]['L1']['points'][1])
                    else:
                        y1 = dir[dir_id]['lines']['start'][index]
                        x1 = int(fit_lane_y2x(y1, fit_x))
                        x2 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][0]
                        y2 = dir[dir_id]['Line' + str(line_id)]['L1']['points'][1][1]
                        dir[dir_id]['Line' + str(line_id)]['L1']['points'] = (dir[dir_id]['Line' + str(line_id)]['L1']['points'][0],(x1,y1))


                    points = ((x1, y1), (x2, y2))
                    lane_type = 'xu'
                    dir[dir_id]['Line' + str(line_id)]['L2'] = {'lane_type': lane_type,'fit': dir[dir_id]['lines']['fits'][index],'points': points}

    return dir


def get_num(dir_index, dir_points, category, image,dir,out):
    direction = dir[dir_index]['direction']
    if category == 0:
        scale = 1 / 20
    else:
        scale = 20
    while True:
        dir_points_scale = scale_coordinates(dir_points, scale)  # 路段
        dbscan_d = DBSCAN(eps=80, min_samples=100)
        labels_d = dbscan_d.fit_predict(dir_points_scale)
        label_num = set(labels_d)

        # 绘制聚类结果
        x = [point[0] for point in dir_points]
        y = [point[1] for point in dir_points]
        plt.scatter(y, x, c=labels_d)
        plt.title('Dir' + str(dir_index) + 'Points Clustering')
        plt.show()

        max_num,min_num =0,len(labels_d)
        for label in label_num:
            number = list(labels_d).count(label)
            max_num = max(number,max_num)
            min_num = min(number,min_num)
        if max_num <= 20*min_num or len(label_num) < 4:
            break
        if category == 0:
            scale = scale / 2
        else:
            scale = scale * 2

    # 提取每个类别的像素点
    clustered_points = [dir_points[labels_d == i] for i in range(max(labels_d) + 1)]
    # # 绘制聚类结果
    # x = [point[0] for point in dir_points]
    # y = [point[1] for point in dir_points]
    # plt.scatter(y, x, c=labels_d)
    # plt.title('Dir' + str(dir_index) + ' Out Points Clustering, scale=' + str(scale))
    # plt.show()

    intersection_points = []
    for points_l in clustered_points:
        if len(points_l) > 10:
            points_l = lane_points_pre(points_l)
            fit_l = np.polyfit(points_l[:, 1], points_l[:, 0], 1)
            def lane(x):
                return fit_l[1] + fit_l[0] * x

            max_x_l = max(point[1] for point in points_l)
            min_x_l = min(point[1] for point in points_l)
            max_y_l = max(point[0] for point in points_l)
            min_y_l = min(point[0] for point in points_l)
            dis_line = dis(max_x_l, max_y_l, min_x_l, min_y_l)

            if dis_line >= 10:
                fit, stop_line = dir[dir_index]['stop']['fit'],dir[dir_index]['stop']['points']
                def stop(x):
                    return fit[1] + fit[0] * x

                tan_jiajiao = abs((fit_l[0] - fit[0]) / (1 + fit[0] * fit_l[0]))
                if tan_jiajiao > 10:
                    # 求解两条线的交点
                    intersection_point = find_intersection(stop, lane)

                    if category == 0:
                        x1 = min(np.array(stop_line)[0][0], np.array(stop_line)[1][0])
                        x2 = max(np.array(stop_line)[0][0], np.array(stop_line)[1][0])
                        if intersection_point[0] <= (x1 - 3) or intersection_point[0] >= (x2 + 3):
                            out[direction] += 1
                            intersection_points.append(intersection_point)

                    elif category == 1:
                        y1 = min(stop_line[0][1], stop_line[1][1])
                        y2 = max(stop_line[0][1], stop_line[1][1])
                        if intersection_point[1] <= y1 - 15 or intersection_point[1] >= y2 + 15:
                            out[direction] += 1
                            intersection_points.append(intersection_point)


                    # 画lane
                    x_range_l = np.arange(min_x_l, max_x_l)
                    y_range_l = fit_lane(x_range_l, fit_l)
                    draw_points_l = np.column_stack((x_range_l, y_range_l)).astype(np.int32)
                    for i in range(len(draw_points_l) - 1):
                        cv2.line(image, tuple(draw_points_l[i]), tuple(draw_points_l[i + 1]), (255, 0, 0), 5)
                    cv2.imwrite('image_lane_re.jpg', image)

    lane_mean = dir[dir_index]['lane_width']
    if len(intersection_points) > 1:
        for id,point1 in enumerate(intersection_points):
            count = 0
            for i in range(len(intersection_points)):
                if i != id:
                    point2 = intersection_points[i]
                    dis_inter = dis(point1[0],point1[1],point2[0],point2[1])
                    if dis_inter <= 3 * lane_mean/4:
                        count += 1
            if count >=2:
                out[direction] -= 1

    return out

def check(a1,a2,a3,a4):
    a = [0,0,0,0]
    if len(a1)>0:
        a[0] = 1
    if len(a2)>0:
        a[1] = 1
    if len(a3)>0:
        a[2] = 1
    if len(a4)>0:
        a[3] = 1

    return a

def hebing(a1,a2,a3,a4):
    a = check(a1,a2,a3,a4)
    aa = [a1,a2,a3,a4]
    out = []
    for i in range(4):
        if a[i] ==1:
            out.extend(aa[i])

    return out


def kb_select(direction, dd, mode):
    fits_l, end_l, start_l, weights = direction['lines']['fits'], direction['lines']['end'], direction['lines']['start'],direction['lines']['weight']

    if len(fits_l) > 0:
        kbs_old = []
        kbs = []
        k_mean = 0
        weight_sum = 0
        if dd == 1 or dd == 3:
            Y1 = []
            for fit_,weight,s,e in zip(fits_l,weights,start_l,end_l):
                judge_v = (s+e)/2
                kbs_old.append(np.array(fit_))
                y1 = fit_[0]*judge_v + fit_[1]
                Y1.append(y1)
                k_mean = k_mean + fit_[0] * weight
                weight_sum = weight_sum + weight
            k_mean = k_mean / weight_sum
            for y1 in Y1:
                b2 = y1 - k_mean*judge_v
                kbs.append((k_mean,b2))
        else:
            X1 = []
            for fit_, weight, s, e in zip(fits_l, weights, start_l, end_l):
                judge_v = (s + e) / 2
                kbs_old.append(np.array(fit_))
                x1 = (judge_v - fit_[1]) / fit_[0]
                X1.append(x1)
                k_mean = k_mean + fit_[0]* weight
                weight_sum = weight_sum + weight
            k_mean = k_mean / weight_sum
            for x1 in X1:
                b2 = judge_v - k_mean*x1
                kbs.append((k_mean,b2))

        # kbs = select_points(kbs)
        kbs = np.array(kbs)
        # kbs_new = np.array(kbs)


        # # 使用DBSCAN聚类将像素点聚类
        # ddis = 40
        # while True:
        #     dbscan_kb = DBSCAN(eps=ddis, min_samples=1)
        #     labels = dbscan_kb.fit_predict(kbs_new)
        #     num_unique_elements = len(set(labels))
        #     if num_unique_elements == 5:
        #         break
        #     ddis = ddis - 2
        # 使用DBSCAN聚类将像素点聚类
        ddis = 20
        dbscan_kb = DBSCAN(eps=ddis, min_samples=1)
        labels = dbscan_kb.fit_predict(kbs)


        # 提取每个类别的像素点
        clustered_kbs = [kbs[labels == i] for i in range(max(labels) + 1)]
        # clustered_kbs = [kbs_new[labels == i] for i in range(max(labels) + 1)]

        # # 绘制聚类结果
        # k = [point[0] for point in kbs]
        # b = [point[1] for point in kbs]
        # labels_d = dbscan_kb.labels_
        # plt.scatter(k, b, c=labels_d)
        # plt.title('kb Points Clustering' + str(mode))
        # plt.show()


        b_all = [point[1] for point in kbs]
        index_all = {value: index for index, value in enumerate(b_all)}
        new_kbs = []
        if len(kbs) > len(clustered_kbs):
            for points in clustered_kbs:
                # 聚类后点数多于一个时，选择粗略最长的进行保留
                if len(points) > 1:
                    indx = 0
                    max_dis = 0
                    for i in range(len(points)):
                        b_0 = points[i][1]
                        dis = end_l[i] - start_l[i]
                        if dis > max_dis:
                            indx = i
                            max_dis = dis
                    points = points[indx]
                #     print('kb filtered')
                else:
                    points = points[0]
                #     print('kb no need filter')
                new_kbs.append(points)
        else:
            for points in clustered_kbs:
                points = points[0]
                new_kbs.append(points)
        clustered_kbs = new_kbs

        if dd == 1 or dd == 3:
            b = [point[1] for point in clustered_kbs]
            if dd == 3:
                set_b = sorted(b)   # 从上到下
                index_l = {value: index for index, value in enumerate(set_b)}
            else:
                b.sort(reverse=True)
                index_l = {value: index for index, value in enumerate(b)}
        else:
            x0 = [-(point[1] / point[0]) for point in clustered_kbs]
            index_xb = {x: point[1] for x, point in zip(x0,clustered_kbs)}
            if dd == 4:
                set_x0 = sorted(x0)
                index_l = {index_xb[value]: index for index, value in enumerate(set_x0)}
            else:
                x0.sort(reverse=True)
                index_l = {index_xb[value]: index for index, value in enumerate(x0)}

        for kb in clustered_kbs:
            name = index_l[kb[1]]
            index = index_all[kb[1]]
            if mode == 1:
                type = 'single_line'
                lane_type = 'shi'
                direction['Line' + str(name)] = {'type': type,
                                                 'L1': {'lane_type': lane_type, 'fit': kbs[index], 'end': end_l[index],
                                                        'start': start_l[index]}}


    direction.pop('lines')
    return direction


def find_max_min_index(lst):
    max_val = max(lst)
    min_val = min(lst)
    max_index = lst.index(max_val)
    min_index = lst.index(min_val)
    return max_index, min_index



def kbs_select(dir, image,mode):
    for dir_id in range(len(dir)):
        direction = dir[dir_id]['direction']
        # if direction == 1 or direction == 3:
        #     mid = image.shape[1] / 2
        # else:
        #     mid = image.shape[0] / 2
        # dir[dir_id] = kb_select(dir[dir_id], direction, mid, mode)
        dir[dir_id] = kb_select(dir[dir_id], direction, mode)


    for dir_id in range(len(dir)):
        direction = dir[dir_id]['direction']
        for line_id in range(len(dir[dir_id]) - 5):
            fit_s = dir[dir_id]['Line' + str(line_id)]['L1']['fit']

            if direction == 1:
                x1 = dir[dir_id]['Line' + str(line_id)]['L1']['start']
                x2 = image.shape[1] - 1
                y1 = int(fit_lane(x1, fit_s))
                y2 = int(fit_lane(x2, fit_s))
            elif direction == 2:
                y1 = dir[dir_id]['Line' + str(line_id)]['L1']['end']
                y2 = 0
                x1 = int(fit_lane_y2x(y1, fit_s))
                x2 = int(fit_lane_y2x(y2, fit_s))
            elif direction == 3:
                x1 = dir[dir_id]['Line' + str(line_id)]['L1']['end']
                x2 = 0
                y1 = int(fit_lane(x1, fit_s))
                y2 = int(fit_lane(x2, fit_s))
            else:
                y1 = dir[dir_id]['Line' + str(line_id)]['L1']['start']
                y2 = image.shape[0] - 1
                x1 = int(fit_lane_y2x(y1, fit_s))
                x2 = int(fit_lane_y2x(y2, fit_s))

            points = ((x1, y1), (x2, y2))
            dir[dir_id]['Line' + str(line_id)]['L1'].pop('start')
            dir[dir_id]['Line' + str(line_id)]['L1'].pop('end')
            dir[dir_id]['Line' + str(line_id)]['L1']['points'] = points

    return dir


# 计算两点之间的欧氏距离
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 计算每个点到其他所有点的距离
def select_points(points):
    Mean = []
    for point in points:
        mean = 0
        for p in points:
            if p[0] != point[0] and p[1] != point[1]:
                dist = euclidean_distance(point, p)
                mean = mean + dist
        mean = mean / len(points)
        Mean.append(mean)
    max_val = max(Mean)
    min_val = min(Mean)
    if max_val >= 3 * min_val / 2:
        max_index = Mean.index(max_val)
        new_list = []
        for index, value in enumerate(points):
            if index != max_index:
                new_list.append(value)

        points = new_list

    return points



def find_max_in_2d_array(arr):
    max_val = arr[0][0]
    for row in arr:
        for val in row:
            if val > max_val:
                max_val = val
    return max_val

def input_out_lane_num():
    out = {
        1: 4,
        3: 4,
        4: 2
    }
    return out


def fit_lanes(image_s, image, image_l):
    image0 = image.copy()
    dir, size = stop_line_cluster_fit(image_s, image0)
    dir = dir_cluster_fit(1, image_l, image0, dir)
    # dir = dir_cluster_fit(2,image_x,image0,dir)
    # out_num = get_out_lane_num(image_x,image0,dir)
    out_num = input_out_lane_num()

    return dir, out_num, size


if __name__ == '__main__':
    # 绘制拟合线段在输出图片上
    image0 = cv2.imread('img/13_0.jpg')

    image_s = cv2.imread('../img_s.jpg')
    image_l = cv2.imread('../img_l.jpg')
    image_x = cv2.imread('../img_x.jpg')

    dir = fit_lanes(image_s,image0,image_l,image_x)

    # 画线
    c = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # 红黄绿蓝
    for dir_id in range(len(dir)):
        if len(dir[dir_id]) > 2:  # 画有交点的停止线
            points = dir[dir_id]['stop']['points']
            cv2.line(image0, points[0], points[1], c[dir_id], 5)

            for line_id in range(len(dir[dir_id]) - 5):
                # 画实线
                points_s = dir[dir_id]['Line' + str(line_id)]['L1']['points']
                cv2.line(image0, points_s[0], points_s[1], c[dir_id], 5)
                #     # 画虚线
                # if 'L2' in dir[dir_id]['Line' + str(line_id)]:
                #     points_x = dir[dir_id]['Line' + str(line_id)]['L2']['points']
                #     cv2.line(image_final, points_x[0], points_x[1], (c[dir_id][0] / 2,c[dir_id][1] / 2, c[dir_id][2] / 2), 5)

    cv2.imwrite('output_image.jpg', image0)