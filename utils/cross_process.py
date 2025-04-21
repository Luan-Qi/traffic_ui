# -*- coding: utf-8 -*-
import csv

import cv2
import pandas as pd
import numpy as np
import os
import datetime

from PIL import Image
from moviepy import *
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
import detect_with_api_revise
from unet.cross import fit_lanes, p2l_dis
from unet.predict import predict_road_pixel

from utils.draw_stop_lane import draw_road_lines, get_roi, get_position_id
from utils.main_utils import *
from utils.save_xml import write_crosses


class Hand_Draw_Cross(object):
    def __init__(self, filepath):
        # 文件加载
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        # 划线操作
        flag, img = self.video.read()
        if flag:
            # 路口0  从东向西行驶
            print('Please mark the lane!')
            self.location = lanemark(img)  # .lanemark
            self.roi_vtx = np.array([[self.location[0][0], self.location[1][0], self.location[1][len(self.location[0]) - 1],
                                      self.location[0][len(self.location[0]) - 1]]])
            print('The Location has been marked successfully ')
            print('lane(self.location)', self.location)
            # 标记路口0测流量线（1条）
            print('Please mark the speed lane!')
            self.speed_lane = calculate_speedlane(img)
            print('The speed lane has been marked successfully ')
            print('speedlane(self.speed_lane)', self.speed_lane)

            # 路口1
            self.location1 = lanemark(img)
            self.speed_lane1 = calculate_speedlane(img)
            self.roi_vtx1 = np.array(
                [[self.location1[0][0], self.location1[1][0], self.location1[1][len(self.location1[0]) - 1],
                  self.location1[0][len(self.location1[0]) - 1]]])

            # 路口2
            self.location2 = lanemark(img)
            self.speed_lane2 = calculate_speedlane(img)
            self.roi_vtx2 = np.array(
                [[self.location2[0][0], self.location2[1][0], self.location2[1][len(self.location2[0]) - 1],
                  self.location2[0][len(self.location2[0]) - 1]]])

            # 路口3
            self.location3 = lanemark(img)
            self.speed_lane3 = calculate_speedlane(img)
            self.roi_vtx3 = np.array(
                [[self.location3[0][0], self.location3[1][0], self.location3[1][len(self.location3[0]) - 1],
                  self.location3[0][len(self.location3[0]) - 1]]])

            self.y_numcount = ((self.speed_lane[0][0][0] + self.speed_lane[1][0][0]) // 2)
            # self.lanecross = lane_cross(self.speed_lane, self.location)
            # self.lanecross1 = lane_cross(self.speed_lane1, self.location1)
            # self.lanecross2 = lane_cross(self.speed_lane2, self.location2)
            # self.lanecross3 = lane_cross(self.speed_lane3, self.location3)
            self.k = np.array(self.location[0]).shape[0] - 1  # 车道数量（非车道线数量）
            self.k1 = np.array(self.location1[0]).shape[0] - 1  # 车道数量（非车道线数量）
            self.k2 = np.array(self.location2[0]).shape[0] - 1  # 车道数量（非车道线数量）
            self.k3 = np.array(self.location3[0]).shape[0] - 1  # 车道数量（非车道线数量）
            self.kb = location2kb(self.location)
            self.kb1 = location2kb(self.location1)
            self.kb2 = location2kb(self.location2)
            self.kb3 = location2kb(self.location3)

            # # 计算交点
            # self.y_calculate = []
            # for i in range(self.k + 1):
            #     self.y_calculate.append(self.lanecross[i][0][1])  # yi01
            # self.x_calculate = []
            # for i in range(self.k1 + 1):
            #     self.x_calculate.append(self.lanecross1[i][0][0])  # xi00
            # self.y_calculate1 = []
            # for i in range(self.k2 + 1):
            #     self.y_calculate1.append(self.lanecross2[i][0][1])  # yi01
            # self.x_calculate1 = []
            # for i in range(self.k3 + 1):
            #     self.x_calculate1.append(self.lanecross3[i][0][0])  # xi00
            # self.y_calculate_stop = self.y_calculate  # calculate_linemark(self.regular_location, self.k, self.y_numcountstop)

            self.carnumlane = [0 for _ in range(self.k)]
            self.carnumlane1 = [0 for _ in range(self.k1)]
            self.carnumlane2 = [0 for _ in range(self.k2)]
            self.carnumlane3 = [0 for _ in range(self.k3)]

            self.lane = [[] for _ in range(self.k)]  # car id in lane
            self.lane1 = [[] for _ in range(self.k1)]
            self.lane2 = [[] for _ in range(self.k2)]
            self.lane3 = [[] for _ in range(self.k3)]
            self.stop = [[] for _ in range(self.k)]

            # image = roi_mask(img, [self.roi_vtx, self.roi_vtx2])
            # cv2.imwrite("roi_image.jpg",image)

    def get_ready(self, car_track_save, car_num_save,vid_save):
        self.car_track_save = car_track_save
        self.car_num_save = car_num_save
        self.vid_save = vid_save
        self.flag, img = self.video.read()
        self.fps = round(self.video.get(cv2.CAP_PROP_FPS))  # cv2.CAP_PROP_FRAME_COUNT
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 保存路径根地址
        file_name = os.path.basename(self.filepath)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.root = './output/'
        if self.vid_save:
            self.vid_save_path = 'output/vid/' + file_name + '_output.mp4'
            self.vid_writer = cv2.VideoWriter('output/vid/' + file_name + '_output.mp4', fourcc, self.fps, (self.width, self.height))
        if self.flag:
            cur = datetime.datetime.now()  # 获取当前系统时间
            # 创建csv文件
            if self.car_track_save:
                self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S.csv')
                f = open(self.csvfile, 'a', newline='')
                writer = csv.writer(f)
                writer.writerow(
                    ["vehicle_id", "time", "vehicle_type", "dir_id", "lane_id", "position_x(m)", "position_y(m)",
                     "width(m)", "height(m)", 'xVelocity(km/h)', 'yVelocity(km/h)', 'xAccelerate(m/s^2)', 'yAccelerate(m/s^2)'])
                f.close()

            # 车流量统计csv
            if self.car_num_save:
                self.csvfile0 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                            '%Y-%m-%d %H.%M.%S') + ' west.csv'
                f = open(self.csvfile0, 'a', newline='')
                writer = csv.writer(f)
                writer.writerow(["时间", "车道1车流量", "车道2车流量", "车道3车流量", "车道4车流量", "车道5车流量"])
                f.close()

                self.csvfile1 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                             '%Y-%m-%d') + ' north.csv'
                f1 = open(self.csvfile1, 'a', newline='')
                writer1 = csv.writer(f1)
                writer1.writerow(["时间", "车道1车流量", "车道2车流量", "车道3车流量", "车道4车流量", "车道5车流量"])
                f1.close()

                self.csvfile2 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                             '%Y-%m-%d') + ' east.csv'
                f2 = open(self.csvfile2, 'a', newline='')
                writer2 = csv.writer(f2)
                writer2.writerow(["时间", "车道1车流量", "车道2车流量", "车道3车流量", "车道4车流量", "车道5车流量"])
                f2.close()

                self.csvfile3 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                             '%Y-%m-%d') + ' south.csv'
                f3 = open(self.csvfile3, 'a', newline='')
                writer3 = csv.writer(f3)
                writer3.writerow(["时间", "车道1车流量", "车道2车流量", "车道3车流量", "车道4车流量", "车道5车流量"])
                f3.close()

            sum = 0
            # for i, p in enumerate(self.lanecross):
            #     if i > 0:
            #         p_left = np.array(self.lanecross[i - 1])
            #         p_right = np.array(self.lanecross[i])
            #         lane_width = np.linalg.norm(p_left - p_right)
            #         sum = sum + lane_width
            # for i, p in enumerate(self.lanecross2):
            #     if i > 0:
            #         p_left = np.array(self.lanecross2[i - 1])
            #         p_right = np.array(self.lanecross2[i])
            #         lane_width = np.linalg.norm(p_left - p_right)
            #         sum = sum + lane_width
            # lane_width_mean = sum/(len(self.lanecross) + len(self.lanecross1) -2)
            lane_width_mean = 34
            # self.scale = 3.75/lane_width_mean # 混合车道平均宽度3.75m
            self.scale = 3.5 / lane_width_mean  # 混合车道平均宽度3.5m

    def process(self):
        # 模型加载
        self.detection = detect_with_api_revise.detectapi(weights='weights/best407_att.pt')  # YOLO() weights='models/yolov7.pt' epoch_299.pt weights/best113.pt
        classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus', 'e-bike']
        self.pts = {}

        print('开始读取视频')
        frame_index = 0
        self.cap = cv2.VideoCapture(self.filepath)

        while (self.cap.isOpened()):
            ret, frame = self.video.read()
            print("index:", frame_index)
            yield f"正在处理第{frame_index}帧\n"
            if not ret and type(frame) == type(None):
                if 30 > frame_index > 10:
                    print('frame' + str(frame_index) + '出现了问题跳过')
                    frame_index = frame_index + 1
                    continue
                elif frame_index > 30:
                    print("处理完成")
                    file_name = os.path.basename(self.filepath)
                    cur = datetime.datetime.now()  # 获取当前系统时间
                    outstr = ""
                    if self.vid_save:
                        outstr += f"运行结束，结果保存在{self.vid_save_path}\n"
                    if self.car_track_save:
                        self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                '%Y-%m-%d %H.%M.%S.csv')
                        outstr += f"运行结束，结果保存在{self.csvfile}\n"
                    # 车流量统计csv
                    if self.car_num_save:
                        self.csvfile0 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                 '%Y-%m-%d %H.%M.%S') + ' west.csv'
                        self.csvfile1 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                 '%Y-%m-%d') + ' north.csv'
                        self.csvfile2 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                 '%Y-%m-%d') + ' east.csv'
                        self.csvfile3 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                 '%Y-%m-%d') + ' south.csv'
                        outstr += f"运行结束，结果保存在{self.csvfile0}\n"
                        outstr += f"运行结束，结果保存在{self.csvfile1}\n"
                        outstr += f"运行结束，结果保存在{self.csvfile2}\n"
                        outstr += f"运行结束，结果保存在{self.csvfile3}\n"

                    yield outstr
                    cv2.destroyAllWindows()
                    self.cap.release()
                    if self.vid_save:
                        self.vid_writer.release()
                    break


            Image_b = frame.copy()
            v = []
            if len(self.roi_vtx[0]) > 0:
                v.append(self.roi_vtx)
            if len(self.roi_vtx2[0]) > 0:
                v.append(self.roi_vtx2)
            image = roi_mask(Image_b, v)  # 经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
            # cv2.imwrite('roi.jpg',image)
            # image = frame


            this_frame_info = {}
            this_frame_track = {}
            for dir_id in range(4):
                this_frame_info[dir_id] = {}
                for lane_id in range(1, 7):
                    this_frame_info[dir_id][lane_id] = {}


            outputs, list_name = self.detection.detect([image])

            if outputs is not None:
                for index, datainoutput in enumerate(outputs):
                    [x1, y1, x2, y2, track_id, cls] = datainoutput
                    counpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    # 获取所在路段、车道id
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
                                self.lane[j].append(track_id)
                                dir_id = 0
                                lane_id = j+1

                    if self.speed_lane1 != 0:
                            K1 = self.kb1[0]
                            B1 = self.kb1[1]
                            X1 = []
                            for ii in range(self.k1 + 1):
                                if K1[ii] != '':
                                    x_ii = int((counpoint[1] - B1[ii]) / K1[ii])
                                else:
                                    x_ii = B1[ii]
                                X1.append(x_ii)
                            X1.sort()

                            for j in range(self.k1):
                                if (counpoint[0] >= X1[j] and counpoint[0] < X1[j + 1]):
                                # if (foot_x >= self.x_calculate[j] and foot_x <= self.x_calculate[j + 1]):
                                    self.lane1[j].append(track_id)
                                    dir_id = 1
                                    lane_id = j + 1

                    if self.speed_lane2 != 0:
                        # 计算垂足
                        # foot_x, foot_y = get_foot(self.speed_lane2[0][0], self.speed_lane2[1][0], counpoint)
                        K2 = self.kb2[0]
                        B2 = self.kb2[1]
                        Y2 = []
                        for ii in range(self.k2 + 1):
                            if K2[ii] != '':
                                y_ii = int(K2[ii] * counpoint[0] + B2[ii])
                                Y2.append(y_ii)
                        Y2.sort()

                        for j in range(self.k2):
                            if (Y2[j] <= counpoint[1] < Y2[j+1]):
                                # if (foot_y <= self.y_calculate[j] and foot_y >= self.y_calculate[j + 1]):  # y_calculate降序排列   # >=    <=
                                self.lane2[j].append(track_id)
                                dir_id = 2
                                lane_id = j+1

                    if self.speed_lane3 != 0:
                            # 计算垂足
                            # foot_x, foot_y = get_foot(self.speed_lane3[0][0], self.speed_lane3[1][0], counpoint)
                            K3 = self.kb3[0]
                            B3 = self.kb3[1]
                            X3 = []
                            for ii in range(self.k3 + 1):
                                if K3[ii] != '':
                                    x_ii = int((counpoint[1] - B3[ii]) / K3[ii])
                                else:
                                    x_ii = B3[ii]
                                X3.append(x_ii)
                            X3.sort()

                            for j in range(self.k3):
                                # if (foot_x >= self.x_calculate1[j] and foot_x <= self.x_calculate1[j + 1]):
                                if (counpoint[0] >= X3[j] and counpoint[0] < X3[j + 1]):
                                    self.lane3[j].append(track_id)
                                    dir_id = 3
                                    lane_id = j + 1

                    if dir_id is not None and lane_id is not None:
                        this_frame_track[track_id] = {'dir_id': dir_id, 'lane_id': lane_id}
                        this_frame_info[dir_id][lane_id][track_id] = {'last_pos': counpoint, 'speed': (None, None), 'accelerate': (None, None)}
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

            if outputs is not None and len(outputs) and self.car_track_save:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                classes2 = outputs[:, -1]
                update_frame_info = {}
                for dir_id in range(2):
                    update_frame_info[dir_id] = {}
                    for lane_id in range(1,7):
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
                                                                   'accelerate': (a_x, a_y)}

                    else:
                        # 插入 # 本帧位置
                        this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                        update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos, 'speed': (None, None), 'accelerate': (None, None)}
                        this_frame_track[key]['speed'] = (None, None)

                last_frame_info = update_frame_info
                draw_boxes(frame, bbox_xyxy, [classes_names[i] for i in classes2], classes2, identities, this_frame_track)
            else:
                last_frame_info = this_frame_info

            if outputs is not None and len(outputs) and self.car_num_save:
                for ii in range(self.k):
                    if self.lane[ii] != []:
                        self.lane[ii] = list(set(self.lane[ii]))

                for ii in range(self.k1):
                    if self.lane1[ii] != []:
                        self.lane1[ii] = list(set(self.lane1[ii]))

                for ii in range(self.k2):
                    if self.lane2[ii] != []:
                        self.lane2[ii] = list(set(self.lane2[ii]))

                for ii in range(self.k3):
                    if self.lane3[ii] != []:
                        self.lane3[ii] = list(set(self.lane3[ii]))

                    # 车头时距计算+车道流量计算
                    # 路口0（west）
                for i in range(self.k):
                    if self.lane[i] != []:
                        sum = 0
                        for j in self.lane[i]:
                            sum = sum + int(1)
                            self.carnumlane[i] = sum

                    # 路口1（north）
                for i in range(self.k1):
                    if self.lane1[i] != []:
                        sum1 = 0
                        for j in self.lane1[i]:
                            sum1 = sum1 + int(1)
                            self.carnumlane1[i] = sum1
                    # 路口2（east）
                for i in range(self.k2):
                    if self.lane2[i] != []:
                        sum2 = 0
                        for j in self.lane2[i]:
                            sum2 = sum2 + int(1)
                            self.carnumlane2[i] = sum2
                    # 路口3（south）
                for i in range(self.k3):
                    if self.lane3[i] != []:
                        sum3 = 0
                        for j in self.lane3[i]:
                            sum3 = sum3 + int(1)
                            self.carnumlane3[i] = sum3


            if self.speed_lane != 0:
                cv2.line(Image_b, self.speed_lane[0][0], self.speed_lane[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                for i in range(self.k + 1):
                    cv2.line(Image_b, (self.location[0][i][0], self.location[0][i][1]),
                             (self.location[1][i][0], self.location[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            if self.speed_lane1 != 0:
                cv2.line(frame, self.speed_lane1[0][0], self.speed_lane1[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                for i in range(self.k1 + 1):
                    cv2.line(frame, (self.location1[0][i][0], self.location1[0][i][1]),
                             (self.location1[1][i][0], self.location1[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            if self.speed_lane2 != 0:
                for i in range(self.k2 + 1):
                    cv2.line(frame, (self.location2[0][i][0], self.location2[0][i][1]),
                             (self.location2[1][i][0], self.location2[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            if self.speed_lane3 != 0:
                cv2.line(frame, self.speed_lane3[0][0], self.speed_lane3[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                for i in range(self.k3 + 1):
                    cv2.line(frame, (self.location3[0][i][0], self.location3[0][i][1]),
                             (self.location3[1][i][0], self.location3[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度

            # CSV写
            # 流量
            if outputs is not None and len(outputs) and self.car_num_save:
                if frame_index % self.fps == 0:
                    f = open(self.csvfile0, 'a', newline='')
                    writer = csv.writer(f)
                    writer.writerow(
                        splicing_csvdata2(frames_to_timecode(self.fps, frame_index + 2), self.carnumlane))
                    # , fpstimeheadway, timeoccupation, Saturation
                    f.close()

                    f1 = open(self.csvfile1, 'a', newline='')
                    writer1 = csv.writer(f1)
                    writer1.writerow(
                        splicing_csvdata2(frames_to_timecode(self.fps, frame_index + 2), self.carnumlane1))
                    f1.close()

                    f2 = open(self.csvfile2, 'a', newline='')
                    writer2 = csv.writer(f2)
                    writer2.writerow(
                        splicing_csvdata2(frames_to_timecode(self.fps, frame_index + 2), self.carnumlane2))
                    f2.close()

                    f3 = open(self.csvfile3, 'a', newline='')
                    writer3 = csv.writer(f3)
                    writer3.writerow(
                        splicing_csvdata2(frames_to_timecode(self.fps, frame_index + 2), self.carnumlane3))
                    f3.close()

            # 轨迹
            if outputs is not None and len(outputs) and self.car_track_save:
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

                        vx_write = None
                        vy_write = None
                        ax_write = None
                        ay_write = None

                        if last_frame_info[dir_id][lane_id][track_id]['speed'][0] is not None:
                            vx = round(last_frame_info[dir_id][lane_id][track_id]['speed'][0], 4)
                            vy = round(last_frame_info[dir_id][lane_id][track_id]['speed'][1], 4)
                            if dir_id == 0:
                                vx_write = vx
                                vy_write = vy
                            if dir_id == 1:
                                vx_write = vy
                                vy_write = -vx
                            if dir_id == 2:
                                vx_write = -vx
                                vy_write = -vy
                            if dir_id == 3:
                                vx_write = -vy
                                vy_write = vx
                            if last_frame_info[dir_id][lane_id][track_id]['accelerate'][0] is not None:
                                ax = round(last_frame_info[dir_id][lane_id][track_id]['accelerate'][0], 4)
                                ay = round(last_frame_info[dir_id][lane_id][track_id]['accelerate'][1], 4)
                                if dir_id == 0:
                                    ax_write = ax
                                    ay_write = ay
                                if dir_id == 1:
                                    ax_write = ay
                                    ay_write = ax
                                if dir_id == 2:
                                    ax_write = ax
                                    ay_write = ay
                                if dir_id == 3:
                                    ax_write = ay
                                    ay_write = ax

                        with open(self.csvfile, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                splicing_csvdata(track_id,frames_to_timecode(self.fps, frame_index),  cls_id,
                                                 dir_id, lane_id,
                                                 round(self.scale * (bbox_left + bbox_right) / 2, 4),
                                                 round(self.scale * (bbox_top + bbox_bottom) / 2, 4),
                                                 round(self.scale * abs(bbox_right - bbox_left), 1),
                                                 round(self.scale * abs(bbox_bottom - bbox_top), 1),
                                                 vx_write,
                                                 vy_write,
                                                 ax_write,
                                                 ay_write
                                                 ))

                f.close()


            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            # cv2.imwrite('output/pic/frame' + str(frame_index) + '.jpg', Image_b)
            if self.vid_save:
                self.vid_writer.write(Image_b)
            frame_index = frame_index + 1


    def exit_process(self):
        self.cap.release()
        if self.vid_save:
            self.vid_writer.release()


class Segmentation_Cross(object):
    def __init__(self, filepath = None):
        if filepath is None:
            self.dir = None
            self.out = None
            self.size = None
            self.roi_zone = None
            self.scale = None
            self.size = None
            self.road_rules = None
            return

        # 文件加载
        self.video_path = filepath
        self.video = cv2.VideoCapture(filepath)

        # 保存路径根地址
        self.root = './output/'
        frame_index = 0
        ref, frame = self.video.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        while self.video.isOpened:
            ref, frame = self.video.read()
            if not ref:
                break
            if frame_index == 0:  # 选择没有白车在停止线上的帧进行处理
                # print('正在获取车道线信息')
                # cv2.imwrite('index2.jpg', frame)
                image = Image.fromarray(frame)
                image_l, image_s, image_x, image_lx = predict_road_pixel(image)
                # image_l.save("img_l.jpg")
                # image_s.save("img_s.jpg")
                # image_x.save("img_x.jpg")
                # image_lx.save("img_lx.jpg")
                image_s = np.array(image_s)
                image_l = np.array(image_l)
                image_x = np.array(image_x)
                # image_lx = np.array(image_lx)
                # self.dir,self.out, self.size = fit_lanes(image_s, frame, image_lx)
                self.dir,self.out, self.size = fit_lanes(image_s, frame, image_l, image_x)
                self.roi_zone, self.scale = get_roi(self.dir)
                self.size = [self.size[0] * self.scale, self.size[1] * self.scale]
                # frame0 = roi_mask(frame, self.roi_zone)
                # frame = draw_road_lines(frame, self.dir,'')
                # cv2.imwrite('output_image.jpg', frame)

                self.road_rules = {                         # dir_id:lane_id:rule
                    0:{1:['left','straight'],2:['right']},
                    1:{1:['left'],2:['left'],3:['straight'],4:['straight','right']},
                    2:{1:['left'],2:['straight']},
                    3:{1:['left'],2:['straight'],3:['right']}
                }

                break

    def get_ready(self, car_track_save, car_num_save,vid_save):
        self.car_track_save = car_track_save
        self.car_num_save = car_num_save
        self.vid_save = vid_save
        self.exit_flag = False
        self.video_path = self.video_path
        self.video = cv2.VideoCapture(self.video_path)
        self.fps = round(self.video.get(cv2.CAP_PROP_FPS))
        self.list_overlapping = {}
        self.counter_recording = []
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pts = {}
        self.crossing_car = {}

        # 保存路径根地址

        file_name = os.path.basename(self.video_path)
        cur = datetime.datetime.now()  # 获取当前系统时间
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if self.vid_save:
            self.vid_save_path = 'output/vid/' + file_name + '_output.mp4'
            self.vid_writer = cv2.VideoWriter('output/vid/' + file_name + '_output.mp4', fourcc, self.fps, (self.width, self.height))

        # 轨迹csv
        if self.car_track_save:
            self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S') + '.csv'
            f = open(self.csvfile, 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(["frame", "vehicle_id", "vehicle_type", "dir_id", "lane_id", "position_x(m)", "position_y(m)",
                             "width(m)", "height(m)", 'xVelocity(km/h)', 'yVelocity(km/h)', '车头时距(s)', '车道占有率（%）'])
            f.close()

        # 车流量统计csv
        if self.car_num_save:
            self.csvfile0 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' west.csv'
            self.csvfile1 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' north.csv'
            self.csvfile2 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' east.csv'
            self.csvfile3 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' south.csv'
            for dir_id in range(len(self.dir)):
                write_list = []
                for lane_id in range(len(self.dir[dir_id]) - 7):
                    write_list.append('时间')
                    write_list.append('车道' + str(lane_id-1) + '流量')
                    write_list.append('车道' + str(lane_id-1) + '车头时距')
                    write_list.append('车道' + str(lane_id-1) + '车道占有率')

                if self.dir[dir_id]['direction'] == 3:
                    output = self.csvfile0
                if self.dir[dir_id]['direction'] == 1:
                    output = self.csvfile2
                if self.dir[dir_id]['direction'] == 2:
                    output = self.csvfile1
                if self.dir[dir_id]['direction'] == 4:
                    output = self.csvfile3

                with open(output, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(write_list)
                f.close()

    def process(self):
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        # self.detection = detect_with_api_revise.detectapi(weights='weights/best407_att.pt')  # epoch_319.pt epoch_299.pt bestw6_exp75_386.pt   _att_396
        self.detection = detect_with_api_revise.detectapi(weights=r'D:\yjh\code\pytorch\highway_track_id_v_a_ui\weights\best407_att.pt')  # epoch_319.pt epoch_299.pt bestw6_exp75_386.pt   _att_396
        self.classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'e-bike']

        print('开始读取视频')
        frame_index = 0
        self.cap = cv2.VideoCapture(self.video_path)

        t_pass = {}
        t_in = {}
        while(self.cap.isOpened()):
            ret, frame = self.video.read()
            print("index:", frame_index)
            yield f"正在处理第{frame_index}帧\n"
            if not ret and type(frame) == type(None):
                if 30 > frame_index > 10:
                    print('frame' + str(frame_index) + '出现了问题跳过')
                    frame_index = frame_index + 1
                    continue
                elif frame_index > 30:
                    print("处理完成")
                    file_name = os.path.basename(self.video_path)
                    cur = datetime.datetime.now()  # 获取当前系统时间
                    outstr = ""
                    if self.vid_save:
                        outstr += f"运行结束，结果保存在{self.vid_save_path}\n"
                    if self.car_track_save:
                        self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                '%Y-%m-%d %H.%M.%S.csv')
                        outstr += f"运行结束，结果保存在{self.csvfile}\n"
                    # 车流量统计csv
                    if self.car_num_save:
                        if self.car_num_save:
                            self.csvfile0 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                     '%Y-%m-%d %H.%M.%S') + ' west.csv'
                            self.csvfile1 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                     '%Y-%m-%d') + ' north.csv'
                            self.csvfile2 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                     '%Y-%m-%d') + ' east.csv'
                            self.csvfile3 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                                     '%Y-%m-%d') + ' south.csv'
                            outstr += f"运行结束，结果保存在{self.csvfile0}\n"
                            outstr += f"运行结束，结果保存在{self.csvfile1}\n"
                            outstr += f"运行结束，结果保存在{self.csvfile2}\n"
                            outstr += f"运行结束，结果保存在{self.csvfile3}\n"
                    yield "视频处理完成。"
                    yield outstr
                    cv2.destroyAllWindows()
                    self.cap.release()
                    if self.vid_save:
                        self.vid_writer.release()
                    break

            if not self.exit_flag:
                this_frame_track = {}
                this_frame_info = {}
                for dir_id in range(len(self.dir)):
                    this_frame_info[dir_id] = {}
                    for lane_id in range(1,len(self.dir[dir_id])-4):
                        this_frame_info[dir_id][lane_id] =  {}

                # frame_roi = roi_mask(frame, self.roi_zone)  # 预处理 切割输入模型的图像
                frame_roi = frame

                img, outputs = self.detection.detect_video([frame_roi])  # outputs list_name

                if outputs is not None and len(outputs):
                    for item_bbox in outputs[:, :5]:
                        x1, y1, x2, y2, track_id = item_bbox
                        y = int((y1 + y2) / 2)
                        x = int((x1 + x2) / 2)
                        center = (x, y)
                        dir_id, lane_id, direction = get_position_id(x, y, self.dir,self.roi_zone)
                        if dir_id is not None and lane_id is not None:
                            this_frame_track[track_id] = {'dir_id':dir_id,'lane_id':lane_id}
                            this_frame_info[dir_id][lane_id][track_id] = {'last_pos':center, 'speed': (None,None)}

                            if self.car_track_save:
                                # 轨迹记录
                                if track_id in self.pts:
                                    self.pts[track_id].append(center)
                                    if len(self.pts[track_id]) > 30:
                                        del self.pts[track_id][0]
                                else:
                                    self.pts[track_id] = []
                                    self.pts[track_id].append(center)

                                if self.vid_save:
                                    # 画车辆行驶轨迹
                                    thickness = 3
                                    cv2.circle(frame, center, 1, [255, 255, 255], thickness)
                                    if len(self.pts[track_id]) < 30:
                                        draw_coords = self.pts[track_id]
                                    else:
                                        draw_coords = self.pts[track_id][-30:]
                                    if len(draw_coords) > 1:
                                        for j in range(1, len(draw_coords)):
                                            if draw_coords[j - 1] is None or draw_coords[j] is None:
                                                continue
                                            cv2.line(frame, (draw_coords[j - 1]), (draw_coords[j]), [0, 255, 255], thickness)

                            if self.car_num_save:
                                fit = self.dir[dir_id]['stop']['fit']
                                dis_car2stop = p2l_dis(x, y, fit)
                                range_in = 250
                                if range_in -2 <= dis_car2stop <= range_in + 2:
                                    if dir_id not in t_in:
                                        t_in[dir_id] = {}
                                    if lane_id not in t_in[dir_id]:
                                        t_in[dir_id][lane_id] = {}
                                    t_in[dir_id][lane_id][track_id] = frame_index

                                if dis_car2stop <= 2:
                                    if dir_id in t_in and lane_id in t_in[dir_id] and track_id in t_in[dir_id][lane_id]:
                                        dur = frame_index - t_in[dir_id][lane_id][track_id]
                                    else:
                                        dur = frame_index
                                    if dir_id not in t_pass:
                                        t_pass[dir_id] = {}
                                    if lane_id not in t_pass[dir_id]:
                                        t_pass[dir_id][lane_id] = []
                                        occupy_time = dur
                                    else:
                                        occupy_time = t_pass[dir_id][lane_id][-1][1] + dur
                                    info = (frame_index,occupy_time)  # 这辆车达到中间线的帧数，到此时刻该车道内从进入到中间线路段内的车道占有率
                                    t_pass[dir_id][lane_id].append(info)

                if outputs is not None and len(outputs) and self.car_track_save:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]
                    update_frame_info = {}
                    for dir_id in range(len(self.dir)):
                        update_frame_info[dir_id] = {}
                        for lane_id in range(1, len(self.dir[dir_id]) - 4):
                            update_frame_info[dir_id][lane_id] = {}
                    for key, val in this_frame_track.items():
                        dir_id = this_frame_track[key]['dir_id']
                        lane_id = this_frame_track[key]['lane_id']
                        direction = self.dir[dir_id]['direction']
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
                            speed_x,speed_y = estimateSpeed(this_frame_pos, last_frame_pos, self.scale, self.fps,dir_id,self.dir)
                            this_frame_track[key]['speed'] = (speed_x, speed_y)
                            if last_frame_vx is not None:
                                a_x = estimate_a(speed_x, last_frame_vx, this_frame_pos, last_frame_pos, self.scale)
                            else:
                                a_x = None
                            if last_frame_vy is not None:
                                a_y = estimate_a(speed_y, last_frame_vy, this_frame_pos, last_frame_pos, self.scale)
                            else:
                                a_y = None
                            if direction == 1 or direction == 3:
                                update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos, 'speed': (speed_x, speed_y), 'accelerate':(a_x,a_y)}
                            else:
                                update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,'speed': (speed_x, speed_y),'accelerate': (a_x,a_y)}
                        else:
                            # 插入 # 本帧位置
                            this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                            update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos, 'speed': (None,None),'accelerate':(None,None)}
                            this_frame_track[key]['speed'] = (None,None)
                    last_frame_info = update_frame_info
                    draw_boxes(frame, bbox_xyxy, [self.classes_names[i] for i in classes2], classes2, identities, this_frame_track)
                    # frame_f = draw_all_lines(frame, self.dir, last_frame_info, self.lights[frame_index])
                    frame_f = draw_road_lines(frame, self.dir, last_frame_info)
                else:
                    last_frame_info = this_frame_info
                    frame_f = draw_road_lines(frame, self.dir, last_frame_info)

                # CSV写
                # 流量
                if outputs is not None and len(outputs) and self.car_num_save:
                    if frame_index % self.fps == 0:
                        for dir_id in range(len(self.dir)):
                            write_list = []
                            for lane_id in range(len(self.dir[dir_id]) - 7):
                                if dir_id in t_pass:
                                    if lane_id in t_pass[dir_id]:
                                        if len(t_pass[dir_id][lane_id]) > 1:
                                            delta_t = (t_pass[dir_id][lane_id][-1][0] - t_pass[dir_id][lane_id][-2][
                                                0]) / self.fps
                                        zhanyoulv_w = t_pass[dir_id][lane_id][-1][1] / t_pass[dir_id][lane_id][-1][0]
                                        zhanyoulv_w = round(zhanyoulv_w * 100, 4)

                                    else:
                                        delta_t = ''
                                        zhanyoulv_w = ''
                                else:
                                    delta_t = ''
                                    zhanyoulv_w = ''
                                if dir_id in t_in:
                                    if lane_id in t_in[dir_id]:
                                        carnum = len(t_in[dir_id][lane_id])
                                else:
                                    carnum = 0
                                write_list.append(frames_to_timecode(self.fps, frame_index))
                                write_list.append(carnum)
                                write_list.append(delta_t)
                                write_list.append(zhanyoulv_w)

                            if self.dir[dir_id]['direction'] == 3:
                                output = self.csvfile0
                            if self.dir[dir_id]['direction'] == 1:
                                output = self.csvfile2
                            if self.dir[dir_id]['direction'] == 2:
                                output = self.csvfile1
                            if self.dir[dir_id]['direction'] == 4:
                                output = self.csvfile3

                            with open(output, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(write_list)
                            f.close()

                if len(outputs) != 0 and self.car_track_save:
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

                            vx_write = None
                            vy_write = None
                            ax_write = None
                            ay_write = None

                            if last_frame_info[dir_id][lane_id][track_id]['speed'][0] is not None:
                                vx = round(last_frame_info[dir_id][lane_id][track_id]['speed'][0], 4)
                                vy = round(last_frame_info[dir_id][lane_id][track_id]['speed'][1], 4)
                                if dir_id == 0:
                                    vx_write = vx
                                    vy_write = vy
                                if dir_id == 1:
                                    vx_write = vy
                                    vy_write = -vx
                                if dir_id == 2:
                                    vx_write = -vx
                                    vy_write = -vy
                                if dir_id == 3:
                                    vx_write = -vy
                                    vy_write = vx
                                if last_frame_info[dir_id][lane_id][track_id]['accelerate'][0] is not None:
                                    ax = round(last_frame_info[dir_id][lane_id][track_id]['accelerate'][0], 4)
                                    ay = round(last_frame_info[dir_id][lane_id][track_id]['accelerate'][1], 4)
                                    if dir_id == 0:
                                        ax_write = ax
                                        ay_write = ay
                                    if dir_id == 1:
                                        ax_write = ay
                                        ay_write = ax
                                    if dir_id == 2:
                                        ax_write = ax
                                        ay_write = ay
                                    if dir_id == 3:
                                        ax_write = ay
                                        ay_write = ax
                            with open(self.csvfile, 'a', newline='') as f:  # '2'+'.csv'
                                writer = csv.writer(f)
                                writer.writerow(
                                    splicing_csvdata(frames_to_timecode(self.fps, frame_index), track_id, cls_id,
                                                      dir_id, lane_id,
                                                      int(self.scale * (bbox_left + bbox_right) / 2),
                                                      int(self.scale * (bbox_top + bbox_bottom) / 2),
                                                      round(self.scale * abs(bbox_right - bbox_left),1),
                                                      round(self.scale * abs(bbox_bottom - bbox_top),1),
                                                     vx_write,
                                                     vy_write,
                                                     ax_write,
                                                     ay_write
                                                     ))

                    f.close()

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            if self.vid_save:
                self.vid_writer.write(frame_f)
            frame_index += 1

    def exit_process(self):
        self.cap.release()
        if self.vid_save:
            self.vid_writer.release()

    def save_xml(self):
        file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.xmlfile = self.root + file_name + '_lane.xml'
        write_crosses(self.dir, self.road_rules, self.out, self.scale, self.xmlfile)


if __name__ == '__main__':
    # 填入需要处理的视频文件夹路径，注意该python路径为utils文件夹
    filePath = r'D:\yjh\code\pytorch\highway_track_id_v_a\source\wq\DJI_0041_c.MP4'
