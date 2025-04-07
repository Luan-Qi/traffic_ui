# import torch
import pandas as pd

import detect_with_api_revise
import cv2
import csv
# import dlib
# import time
# import threading
import numpy as np
import datetime
# from PIL import Image
# from utils.main_utils import (lanemark, calculate_speedlane, lane_cross, roi_mask, get_foot, frames_to_timecode,
#                         get_point_line_distance, splicing_csvdata2)
from utils.main_utils import *
# from yolo_track2 import YOLO
# from yolov7.utils.datasets import LoadImages, LoadStreams
# from yolov7.utils.general import (
#     check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)  # , plot_one_box
# from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
# import matplotlib.pyplot as plt
from utils.visdrone_lane_volume import detect

from utils.draw_stop_lane import draw_road_lines, get_roi, get_position_id,draw_all_lines



class Track_Block(object):
# 初始化
    # 模型加载 yolo deepsort
    # 文件加载 视频文件
    # 划线操作
    def __init__(self, filepath, row_num):
        # 文件加载
        self.video_path = filepath
        self.video = cv2.VideoCapture(filepath)
        # 加载视频
        self.flag, img = self.video.read()
        self.fps = round(self.video.get(cv2.CAP_PROP_FPS))  # cv2.CAP_PROP_FRAME_COUNT
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 保存路径根地址
        self.root = 'output/'
        if self.flag:
            cur = datetime.datetime.now()  # 获取当前系统时间
            # 路口0
            if data['location0'][row_num] != ' ':
                self.location = eval(data['location0'][row_num])
                self.kb = location2kb(self.location)
                self.speed_lane = eval(data['speedline0'][row_num])
                roi_str = data['roi0'][row_num]
                self.roi_vtx = np.fromstring(roi_str.strip('[]'), sep=' ').reshape(1,4,2)
                self.roi_vtx = np.int32(self.roi_vtx)
                self.lanecross = eval(data['lanecross0'][row_num])
                self.k = np.int32(data['lanenum0'][row_num])
                self.y_calculate = []
                self.carnumlane = [0 for _ in range(self.k)]
                self.lane = [[] for _ in range(self.k)]  # car id in lane
                for i in range(self.k + 1):
                    self.y_calculate.append(self.lanecross[i][0][1])  # yi01
                self.y_calculate = self.y_calculate[::-1]
            else:
                self.roi_vtx = np.array([[]])
                self.speed_lane = 0
            # 路口1
            if data['location1'][row_num] != ' ':
                self.location1 = eval(data['location1'][row_num])
                self.kb1 = location2kb(self.location1)
                self.speed_lane1 = eval(data['speedline1'][row_num])
                roi_str = data['roi1'][row_num]
                self.roi_vtx1 = np.fromstring(roi_str.strip('[]'), sep=' ').reshape(1,4,2)
                self.roi_vtx1 = np.int32(self.roi_vtx1)
                self.lanecross1 = eval(data['lanecross1'][row_num])
                self.k1 = np.int32(data['lanenum1'][row_num])

                self.y_calculate1 = []
                self.carnumlane1 = [0 for _ in range(self.k1)]
                self.lane1 = [[] for _ in range(self.k1)]
                for i in range(self.k1 + 1):
                    self.y_calculate1.append(self.lanecross1[i][0][1])  # xi00
                self.y_calculate = self.y_calculate[::-1]
            else:
                self.roi_vtx1 = np.array([[]])
                self.speed_lane1 = 0

            # 创建csv文件
            self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur,'%Y-%m-%d %H.%M.%S.csv')
            f = open(self.csvfile, 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(
                ["vehicle_id", "time", "vehicle_type", "dir_id", "lane_id", "position_x(m)", "position_y(m)",
                 "width(m)", "height(m)", 'xVelocity(km/h)', 'yVelocity(km/h)', 'xAccelerate(m/s^2)'])
            f.close()
            sum = 0
            for i,p in enumerate(self.lanecross):
                if i>0:
                    p_left = np.array(self.lanecross[i-1])
                    p_right = np.array(self.lanecross[i])
                    lane_width = np.linalg.norm(p_left - p_right)
                    sum = sum + lane_width
            for i,p in enumerate(self.lanecross1):
                if i>0:
                    p_left = np.array(self.lanecross1[i-1])
                    p_right = np.array(self.lanecross1[i])
                    lane_width = np.linalg.norm(p_left - p_right)
                    sum = sum + lane_width
            lane_width_mean = sum/(len(self.lanecross) + len(self.lanecross1) -2)
            self.scale = 3.75/lane_width_mean # 混合车道平均宽度3.75m

            # self.y_numcount = ((self.speed_lane[0][0][0] + self.speed_lane[1][0][0]) // 2)
            # self.y_calculate_stop = self.y_calculate  # calculate_linemark(self.regular_location, self.k, self.y_numcountstop)

#帧处理
    def process(self):
        # 模型加载
        self.detection = detect_with_api_revise.detectapi(weights='weights/best407_att.pt')  # YOLO() weights='models/yolov7.pt' epoch_299.pt weights/best113.pt
        classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus', 'e-bike']
        self.pts = {}

        print('开始读取视频')
        frame_index = 0
        cap = cv2.VideoCapture(self.video_path)

        while (cap.isOpened()):
            ret, frame = self.video.read()
            print("index:", frame_index)
            if not ret and type(frame) == type(None):
                if 30 > frame_index > 10:
                    print('frame' + str(frame_index) + '出现了问题跳过')
                    frame_index = frame_index + 1
                    continue
                elif frame_index > 30:
                    print("处理完成")
                    cv2.destroyAllWindows()
                    cap.release()
                    vid_writer.release()
                    break


            Image_b = frame.copy()
            v = []
            if len(self.roi_vtx[0]) > 0:
                v.append(self.roi_vtx)
            if len(self.roi_vtx1[0]) > 0:
                v.append(self.roi_vtx1)
            image = roi_mask(Image_b, v)  # 经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
            # cv2.imwrite('roi.jpg',image)


            this_frame_info = {}
            this_frame_track = {}
            for dir_id in range(2):
                this_frame_info[dir_id] = {}
                for lane_id in range(1, 5):
                    this_frame_info[dir_id][lane_id] = {}


            outputs, list_name = self.detection.detect([image])

            if outputs is not None:
                for index, datainoutput in enumerate(outputs):
                    [x1, y1, x2, y2, track_id, cls] = datainoutput
                    counpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    if self.speed_lane != 0:
                        # 计算垂足
                        # foot_x, foot_y = get_foot(self.speed_lane[0][0], self.speed_lane[1][0], counpoint)
                        # 计算x1+x2/2，属于各kb间的哪一个
                        K = self.kb[0]
                        B = self.kb[1]
                        Y = []
                        for ii in range(self.k +1):
                            if K[ii] != '':
                                y_ii = int(K[ii] * counpoint[0] + B[ii])
                                Y.append(y_ii)
                        Y.sort(reverse=True)

                        for j in range(self.k):
                            if ( counpoint[1] <= Y[j] and counpoint[1] > Y[j+1]):
                            # if (foot_y <= self.y_calculate[j] and foot_y >= self.y_calculate[j + 1]):  # y_calculate降序排列   # >=    <=
                                bbox_xyxy = outputs[:, :4]  # [x1,y1,x2,y2]
                                identities = outputs[:, -2]  # [i]
                                classes2 = outputs[:, -1]  # [cls]
                                Image_b = detect_with_api_revise.detectapi.draw_boxes(Image_b, bbox_xyxy, classes2, [classes_names[i] for i in classes2], identities)  # names
                                self.lane[j].append(i)
                                dir_id = 0
                                lane_id = j+1

                    if self.speed_lane1 != 0:
                        # 计算垂足
                        # foot_x, foot_y = get_foot(self.speed_lane1[0][0], self.speed_lane1[1][0], counpoint)
                        K1 = self.kb1[0]
                        B1 = self.kb1[1]
                        Y1 = []
                        for ii in range(self.k1 + 1):
                            if K1[ii] != '':
                                y_ii = int(K1[ii] * counpoint[0] + B1[ii])
                                Y1.append(y_ii)
                        Y1.sort()

                        for j in range(self.k1):
                            if (Y1[j] <= counpoint[1] < Y1[j+1]):
                                # if (foot_y <= self.y_calculate[j] and foot_y >= self.y_calculate[j + 1]):  # y_calculate降序排列   # >=    <=
                                bbox_xyxy = outputs[:, :4]  # [x1,y1,x2,y2]
                                identities = outputs[:, -2]  # [i]
                                classes2 = outputs[:, -1]  # [cls]
                                Image_b = detect_with_api_revise.detectapi.draw_boxes(Image_b, bbox_xyxy, classes2,[classes_names[i] for i in classes2], identities)  # names
                                self.lane1[j].append(i)
                                dir_id = 1
                                lane_id = j+1


                    if dir_id is not None and lane_id is not None:
                        this_frame_track[track_id] = {'dir_id': dir_id, 'lane_id': lane_id}
                        this_frame_info[dir_id][lane_id][track_id] = {'last_pos': counpoint, 'speed': (None, None)}
                        # 轨迹记录
                        if track_id not in self.pts:
                            self.pts[track_id] = []
                        self.pts[track_id].append(counpoint)

                        # 画车辆行驶轨迹
                        thickness = 3
                        cv2.circle(Image_b, counpoint, 1, [255, 255, 255], thickness)
                        if len(self.pts[track_id]) < 30:
                            draw_coords = self.pts[track_id]
                        else:
                            draw_coords = self.pts[track_id][-30:]
                        if len(draw_coords) > 1:
                            for j in range(1, len(draw_coords)):
                                if draw_coords[j - 1] is None or draw_coords[j] is None:
                                    continue
                                cv2.line(Image_b, (draw_coords[j - 1]), (draw_coords[j]), [0, 255, 255], thickness)
                            # cv2.imwrite('draw.jpg',Image_b)
                            # print('draw one car')

            if outputs is not None and len(outputs):
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                classes2 = outputs[:, -1]
                update_frame_info = {}
                for dir_id in range(2):
                    update_frame_info[dir_id] = {}
                    for lane_id in range(1,5):
                        update_frame_info[dir_id][lane_id] = {}
                for key, val in this_frame_track.items():
                    dir_id = this_frame_track[key]['dir_id']
                    lane_id = this_frame_track[key]['lane_id']
                    if key in last_frame_info[dir_id][lane_id]:
                        # 更新
                        # 本帧位置
                        this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                        # 上帧位置
                        last_frame_pos = last_frame_info[dir_id][lane_id][key]['last_pos']
                        # 上帧速度
                        last_frame_vx = last_frame_info[dir_id][lane_id][key]['speed'][0]
                        last_frame_vy = last_frame_info[dir_id][lane_id][key]['speed'][1]
                        # 计算距离
                        # if dir_id == 0:
                        #     self.speed_lane[0][1]-
                        #     dir_k = K
                        # elif dir_id == 1:
                        #     dir_k = K1
                        speed_x, speed_y = estimateSpeed_drawlines(this_frame_pos, last_frame_pos, self.scale, self.fps)  # km/h
                        this_frame_track[key]['speed'] = (speed_x, speed_y)
                        if last_frame_vx is not None:
                            a_x = estimate_a(speed_x, last_frame_vx, this_frame_pos, last_frame_pos, self.scale)  # m/s^2
                        else:
                            a_x = None
                        if last_frame_vy is not None:
                            a_y = estimate_a(speed_y, last_frame_vy, this_frame_pos, last_frame_pos, self.scale)
                        else:
                            a_y = None
                        update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,
                                                                   'speed': (speed_x, speed_y),
                                                                   'accelerate': a_x}


                    else:
                        # 插入 # 本帧位置
                        this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                        update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,
                                                                   'speed': (None, None), 'accelerate': None}
                        this_frame_track[key]['speed'] = (None, None)

                last_frame_info = update_frame_info
                draw_boxes(frame, bbox_xyxy, [classes_names[i] for i in classes2], classes2, identities, this_frame_track)
            else:
                last_frame_info = this_frame_info

            if self.speed_lane != 0:
                cv2.line(Image_b, self.speed_lane[0][0], self.speed_lane[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                for i in range(self.k + 1):
                    cv2.line(Image_b, (self.location[0][i][0], self.location[0][i][1]),
                             (self.location[1][i][0], self.location[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            if self.speed_lane1 != 0:
                cv2.line(Image_b, self.speed_lane1[0][0], self.speed_lane1[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                for i in range(self.k1 + 1):
                    cv2.line(Image_b, (self.location1[0][i][0], self.location1[0][i][1]),
                             (self.location1[1][i][0], self.location1[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度

            # # 清理lane中的重复元素,然后统计流量
            # if self.speed_lane != 0:
            #     for ii in range(self.k):
            #         if self.lane[ii] != []:
            #             self.lane[ii] = list(set(self.lane[ii]))
            #             self.carnumlane[ii] = len(self.lane[ii])
            # if self.speed_lane1 != 0:
            #     for ii in range(self.k1):
            #         if self.lane1[ii] != []:
            #             self.lane1[ii] = list(set(self.lane1[ii]))
            #             self.carnumlane[ii] = len(self.lane1[ii])

            # CSV写
            if outputs is not None and len(outputs):
                for j, output in enumerate(outputs):
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_right = output[2]
                    bbox_bottom = output[3]
                    track_id = output[-2]
                    cls_id = output[-1]
                    if track_id in this_frame_track:
                        dir_id = this_frame_track[track_id]['dir_id']
                        lane_id = this_frame_track[track_id]['lane_id']

                        vx = last_frame_info[dir_id][lane_id][track_id]['speed'][0]
                        vy = last_frame_info[dir_id][lane_id][track_id]['speed'][1]
                        ax = last_frame_info[dir_id][lane_id][track_id]['accelerate']
                        if vx is not None:
                            vx = round(vx, 4)
                            vy = round(last_frame_info[dir_id][lane_id][track_id]['speed'][1], 4)
                        with open(self.csvfile, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                splicing_csvdata(track_id,frames_to_timecode(self.fps, frame_index),  cls_id,
                                                 dir_id, lane_id,
                                                 round(self.scale * (bbox_left + bbox_right) / 2, 4),
                                                 round(self.scale * (bbox_top + bbox_bottom) / 2, 4),
                                                 round(self.scale * abs(bbox_right - bbox_left), 1),
                                                 round(self.scale * abs(bbox_bottom - bbox_top), 1),
                                                 vx,
                                                 vy,
                                                 ax
                                                 ))

                f.close()


            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            # cv2.imwrite('output/pic/frame' + str(frame_index) + '.jpg', Image_b)
            vid_writer.write(Image_b)
            frame_index = frame_index + 1



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filePath = 'source/20240913高架'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    data_file = os.path.join(filePath + '/data_load_n.csv')
    data = pd.read_csv(data_file, encoding='gbk')  # encoding="utf-8"会报错
    # '视频名称,视频时长,路口数,（roi_vtx,speedlane）
    for root, dirs, files in os.walk(filePath):
        files.sort()
        for file in files:
            if os.path.splitext(file)[1] == '.mp4' or os.path.splitext(file)[1] == '.MP4':
                file_name = os.path.splitext(file)[0]
                for i in range(len(data)):
                    if str(data['视频名称'][i]) == file_name:
                        source = os.path.join(filePath, file)
                        print('开始初始化视频')

                        #路口追踪器实例化
                        road = Track_Block(source, i)
                        vid_writer = cv2.VideoWriter('output/vid/' + file_name + '_output.mp4', fourcc, road.fps,(road.width, road.height))
                        while True:
                            #帧处理
                            road_out_img = road.process()
                            if road_out_img is None:
                                    break


