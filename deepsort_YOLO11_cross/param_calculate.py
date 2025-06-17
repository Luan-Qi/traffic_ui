import math
import numpy as np
import cv2

'''
判断目标位于哪一个车道线 - sort_lane
 各车道线开始坐标 start_points
 各车道线结束坐标 stop_points
待完善：异常判断
1、不属于任何车道
2、斜率垂直时，为nan,不能进行数值计算
param  - lanes_xyxy 车道线的起始结束坐标 (起始x1,起始y1,结束x2,结束y2)
       - tracks 所有跟踪对象
output 每个检测框位于的车道标号
'''
def sort_lane(lanes_xyxy, tracks):
    k = []
    output = []
    # if xyxy.size == 0:
    #     return output
    #计算各车道线斜率
    for i in range(len(lanes_xyxy)):
        k.append((lanes_xyxy[i][3] - lanes_xyxy[i][1] ) / (lanes_xyxy[i][2] - lanes_xyxy[i][0]));
    # 判断位于哪个车道
    for track in tracks:
        cx_of_lane = []
        # print(f'track.xywh[0])
        for i in range(len(lanes_xyxy)):
            cx_of_lane.append(lanes_xyxy[i][0] + (track.xywh[1] - lanes_xyxy[i][1]) / k[i])
        for j in range(len(cx_of_lane)-1):
            if track.xywh[0] >= cx_of_lane[j] and track.xywh[0] < cx_of_lane[j+1]:
                output.append(j)
                track.lane_label = j
                break
    return output
'''
按照帧数统计
例如：车道1时间占有率 = 车道1有目标出现的帧数/总帧数
- param: sum_frame--总帧数
         lane_frame--车道1-n有目标出现的帧数
'''
def time_occupancy_of_lane(sum_frame, lane_frame):
    time_occupancy = []
    for i in range(len(lane_frame)):
        time_occupancy.append(lane_frame[i] / sum_frame)
    return time_occupancy

'''
交点计算 - get_line_cross_point
- 用来计算两条线之间的交点
- 算法：
    a=y0-y1,b=x1-x0,c=x0y1-x1y0
    F0(x,y) = a0x + b0y + c0 = 0, F(x,y) = a1x + b1y + c1 = 0 
    那么为了计算两条直线的交点应该联立
    a0x+b0y+c0 = 0
    a1x+b1y+c1 = 0
    可得 D = a0*b1 - a1*b0(D为0时，表示两直线平行)
    交点(x,y)为
    - x=(b1*c1 - b1*c0)/D
    - y=(a1*c0 - ao*c1)/D
'''
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
    return [int(x), int(y)]

def distance_2d(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def roi_mask(img, vertices):  # img是输入的图像，vertices是兴趣区的四个点的坐标（三维的数组）
    mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
    # 灰度图和RGB图
    if len(img.shape) > 2:
        channel_num = img.shape[2]
        mask_color = (255,) * channel_num  # 如果 channel_count=3,则为(255,255,255)
    else:
        mask_color = 255
    # print(mask.shape)
    # print(mask_color)
    # print(vertices)
    cv2.fillPoly(mask, vertices, mask_color)  # 使用白色填充多边形，形成蒙板
    return mask

# def roi_max_mask(img, dir):  # img是输入的图像，vertices是兴趣区的四个点的坐标（三维的数组）
#     mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
#     # 灰度图和RGB图
#     if len(img.shape) > 2:
#         channel_num = img.shape[2]
#         mask_color = (255,) * channel_num  # 如果 channel_count=3,则为(255,255,255)
#     else:
#         mask_color = 255
#
#     map = {}
#     map['shui_ping'] = []
#     map['shu_zhi'] = []
#     for i in range(len(dir)):
#         line_keys = [k for k in dir[i].keys() if k.startswith('Line')]
#         if line_keys:
#             max_index_key = max(line_keys, key=lambda x: int(x[4:]))
#             min_index_key = min(line_keys, key=lambda x: int(x[4:]))
#             fit_r = dir[i][max_index_key]['L1']['fit']
#             fit_l = dir[i][min_index_key]['L1']['fit']
#         d = dir[i]['direction']
#
#         if d % 2 == 1:
#             if(len(map['shui_ping']) == 0):
#                 fit = fit_r
#                 map['shui_ping'].append((0,fit[1]))
#                 map['shui_ping'].append((img.shape[1],fit[0] * img.shape[0] + fit[1]))
#                 fit = fit_l
#                 map['shui_ping'].append((img.shape[1], fit[0] * img.shape[0] + fit[1]))
#                 map['shui_ping'].append((0, fit[1]))
#             else:
#                 map['shui_ping'] = map['shui_ping'][:-2]
#                 fit = fit_r
#                 map['shui_ping'].append((img.shape[1], fit[0] * img.shape[0] + fit[1]))
#                 map['shui_ping'].append((0, fit[1]))
#         else:
#             if (len(map['shu_zhi']) == 0):
#                 fit = fit_r
#                 map['shu_zhi'].append((-fit[1] / fit[0], 0))
#                 map['shu_zhi'].append(((img.shape[0]-fit[1]) / fit[0],img.shape[1]))
#                 fit = fit_l
#                 map['shu_zhi'].append(((img.shape[0] - fit[1]) / fit[0], img.shape[1]))
#                 map['shu_zhi'].append((-fit[1] / fit[0], 0))
#             else:
#                 map['shu_zhi'] = map['shu_zhi'][:-2]
#                 fit = fit_r
#                 map['shu_zhi'].append(((img.shape[0] - fit[1]) / fit[0], img.shape[1]))
#                 map['shu_zhi'].append((-fit[1] / fit[0], 0))
#
#     final_v = []
#     final_v.append([np.array(map['shui_ping'], dtype=np.int32)])
#     final_v.append([np.array(map['shu_zhi'], dtype=np.int32)])
#     # cv2.fillPoly(mask, final_v, mask_color)  # 使用白色填充多边形，形成蒙板
#     cv2.fillPoly(mask, [np.array(map['shui_ping'], dtype=np.int32)], mask_color)  # 使用白色填充多边形，形成蒙板
#     cv2.fillPoly(mask, [np.array(map['shu_zhi'], dtype=np.int32)], mask_color)  # 使用白色填充多边形，形成蒙板
#     masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
#     return masked_img

def roi_max_mask(img, dir):  # img是输入的图像，vertices是兴趣区的四个点的坐标（三维的数组）
    mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
    # 灰度图和RGB图
    if len(img.shape) > 2:
        channel_num = img.shape[2]
        mask_color = (255,) * channel_num  # 如果 channel_count=3,则为(255,255,255)
    else:
        mask_color = 255

    map = {}
    map['shui_ping'] = []
    map['shu_zhi'] = []
    for i in range(len(dir)):
        # masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
        # cv2.imwrite("roi.jpg", masked_img)

        line_keys = [k for k in dir[i].keys() if k.startswith('Line')]
        if line_keys:
            max_index_key = max(line_keys, key=lambda x: int(x[4:]))
            min_index_key = min(line_keys, key=lambda x: int(x[4:]))
            fit_r = dir[i][max_index_key]['L1']['fit']
            fit_l = dir[i][min_index_key]['L1']['fit']
        d = dir[i]['direction']

        if d % 2 == 1:
            fit = fit_r
            map['shui_ping'].append((0,fit[1]))
            map['shui_ping'].append((img.shape[1],fit[0] * img.shape[1] + fit[1]))
            fit = fit_l
            map['shui_ping'].append((img.shape[1], fit[0] * img.shape[1] + fit[1]))
            map['shui_ping'].append((0, fit[1]))

            cv2.fillPoly(mask, [np.array(map['shui_ping'], dtype=np.int32)], mask_color)  # 使用白色填充多边形，形成蒙板
        else:
            fit = fit_r
            map['shu_zhi'].append((-fit[1] / fit[0], 0))
            map['shu_zhi'].append(((img.shape[0]-fit[1]) / fit[0], img.shape[0]))
            fit = fit_l
            map['shu_zhi'].append(((img.shape[0] - fit[1]) / fit[0], img.shape[0]))
            map['shu_zhi'].append((-fit[1] / fit[0], 0))

            cv2.fillPoly(mask, [np.array(map['shu_zhi'], dtype=np.int32)], mask_color)  # 使用白色填充多边形，形成蒙板


    masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
    return masked_img


def find_extremum_points(arr1, arr2):
    # 合并两组点
    combined_points = np.vstack((arr1, arr2))

    # 找出 x 的最小值和最大值
    min_x = np.min(combined_points[:, 0])
    max_x = np.max(combined_points[:, 0])

    # 找出 y 的最小值和最大值
    min_y = np.min(combined_points[:, 1])
    max_y = np.max(combined_points[:, 1])

    # 构建由极值组成的四个点的数组
    extremum_points = np.array([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y]
    ])
    return extremum_points

'''
车流量统计 - traffic_volumn_count
param :
- traffic_volumn 每股车道的车流量
- tracks 所有的跟踪目标
'''
def traffic_volume_count(dir):
    for i in range(len(dir)):
        for lane_id, cars in dir[i]['cars'].items():
            dir[i]['traffic_volume'][lane_id] = len(cars)


def get_lane_id(outputs,dir):
    for output in outputs:
        bbox_xyxy = output[:4]
        for dir_id in range(len(dir)):
            dist2line,point = distance_to_line(dir[dir_id]['stop']['points'],bbox_xyxy)
            if(dist2line< 10):
                for i in range(len(dir[dir_id]['cars'])):
                    if(isInRange(dir[dir_id]['crosspoint'][i],dir[dir_id]['crosspoint'][i+1],point)):
                        dir[dir_id]['cars'][i].add(output[-1])
    return dir

def get_cur_lane_id(xy,dir):
    for dir_id in range(len(dir)):
        d = dir[dir_id]['direction']

        if d % 2 == 1:  # shuiping
            key = 'Line0'
            fit_l = dir[dir_id][key]['L1']['fit']
            lastyrange = fit_l[0] * xy[0] + fit_l[1]
            for i in range(1,dir[dir_id]['lane_num']+1):
                key = 'Line' + str(i)
                fit_l = dir[dir_id][key]['L1']['fit']
                nowyrange = fit_l[0] * xy[0] + fit_l[1]
                if (lastyrange <= xy[1] < nowyrange or lastyrange > xy[1] >= nowyrange):
                    return dir_id,i
                lastyrange = nowyrange

        else:
            key = 'Line0'
            fit_l = dir[dir_id][key]['L1']['fit']
            lastxrange = (xy[1] - fit_l[1]) / fit_l[0]
            for i in range(1, dir[dir_id]['lane_num'] + 1):
                key = 'Line' + str(i)
                fit_l = dir[dir_id][key]['L1']['fit']
                nowxrange = (xy[1] - fit_l[1]) / fit_l[0]
                if (lastxrange <= xy[0] < nowxrange or lastxrange > xy[0] >= nowxrange):
                    return dir_id, i
                lastxrange = nowxrange

    return None,None


def isInRange(left,right,point):
    if(left[0] <= point[0] < right[0] or left[1] <= point[1] < right[1]):
        return True
    if(left[0] > point[0] >= right[0] or left[1] > point[1] >= right[1]):
        return True
    return False


# def get_velocity(outputs,track):


def distance_to_line(lane_xyxy, point):
    x1 = lane_xyxy[0][0]
    y1 = lane_xyxy[0][1]
    x2 = lane_xyxy[1][0]
    y2 = lane_xyxy[1][1]
    x0 = (point[0]+point[2])/2
    y0 = (point[1]+point[3])/2
    # 计算直线Ax + By + C = 0的系数
    A = y2 - y1
    B = -(x2 - x1)
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    # 计算点到直线的距离
    distance = abs(A * x0 + B * y0 + C) / math.sqrt(A ** 2 + B ** 2)
    # 计算垂足坐标
    if x2 != x1 and y2 != y1:
        k1 = (y2 - y1) / (x2 - x1)
        k2 = -1 / k1
        x_Q = (k1 * x1 - k2 * x0 + y0 - y1) / (k1 - k2)
        y_Q = k2 * (x_Q - x0) + y0
    elif x2 == x1:
        x_Q = x1
        y_Q = y0
    else:
        y_Q = y1
        x_Q = x0
    return distance, (x_Q, y_Q)


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


if __name__ == '__main__':
    img = cv2.imread('../img/test.jpg')
    print(img)
    # lane_xyxy = draw_lane(img)
    # speed_line_xyxy = draw_speed_line(img)
    # stop_line_xyxy = draw_stop_line(img)
    # print(lane_xyxy)
    # print(speed_line_xyxy)
    # print(stop_line_xyxy)
    # (x, y) = get_line_cross_point(lane_xyxy[0], speed_line_xyxy)
    # cv2.circle(img, [int(x), int(y)], 5, (0, 255, 0), -2)
    roi_mask(img, np.array([[[444, 294],[613,291],[1145, 701],[215, 691]]], np.int32))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    # print(f'该目标位于车道 {sort_lane(start_points, stop_points, (900,485))}')

    cv2.waitKey(0)