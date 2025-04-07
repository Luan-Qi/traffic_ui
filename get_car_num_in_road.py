'''
direction 1:E,2:山，3:反E，4:倒山
dir_id:从0开始
lane_id:从1开始
'''

import csv
import datetime
import os
import queue
import cv2
import time
import numpy as np
import torch
from PIL import Image
# from utils.main_utils import lanemark as lm, calculate_speedlane, roi_mask, roi_mask2
from utils.main_utils import lanemark as lm, calculate_speedlane, roi_mask
import detect_with_api_revise
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from utils.main_utils import counter_vehicles, splicing_csvdata, frames_to_timecode, estimateSpeed,estimate_a, draw_counter, draw_boxes, splicing_csvdata5
from utils.draw_stop_lane import draw_road_lines, get_roi, get_position_id,draw_all_lines
from utils.visdrone_lane_volume import detect
from yolov7.utils.torch_utils import time_synchronized
from unet.predict import predict_road_pixel
from unet.road_shiend import fit_lanes,p2l_dis

def xyxy_to_xywh(*xyxy):  # 从绝对像素值计算相对边界框。xyxy是绝对像素坐标，xyxy[0，1]是左上（？）的横纵坐标、[2，3]是对角的横纵坐标
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])  # t.item()将Tensor变量转换为python标量（int float等）
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])  # 得到左上角绝对像素坐标
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())  # abs()取绝对值，计算边界框的宽
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())  # 计算边界框的高
    x_c = (bbox_left + bbox_w / 2)  # 计算边界框的中心绝对像素坐标
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h  # 返回框的中心坐标和宽高

class Live(object):
    def __init__(self, filepath, flag=False):
        self.frame_queue = queue.Queue(maxsize=100)
        self.exit_flag = flag
        self.video_path = filepath
        self.video = cv2.VideoCapture(self.video_path)
        self.fps = round(self.video.get(cv2.CAP_PROP_FPS))
        self.list_overlapping = {}
        self.counter_recording = []
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pts = {}
        self.crossing_car = {}
        self.dirs = {}
        self.lights = {}


        # 保存路径根地址
        self.root = 'output/'
        cur = datetime.datetime.now()  # 获取当前系统时间
        self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S') + '.csv'
        f = open(self.csvfile, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_id", "vehicle_type", "dir_id", "lane_id", "position_x(m)", "position_y(m)",
                         "width(m)", "height(m)", 'xVelocity(km/h)','yVelocity(km/h)','车头时距(s)','车道占有率（%）'])
        f.close()

        self.csvfile1 = self.root + file_name + '_light_' + datetime.datetime.strftime(cur,
                                                                                       '%Y-%m-%d %H.%M.%S') + '.csv'
        f2 = open(self.csvfile1, 'a', newline='')
        writer2 = csv.writer(f2)
        writer2.writerow(["time", "dir_id", "lane_id", '灯', '持续时间'])
        f2.close()
        self.xmlfile = self.root + file_name + '_light_' + datetime.datetime.strftime(cur,
                                                                                       '%Y-%m-%d %H.%M.%S') + '.xml'

    def read_lane(self):
        frame_index = 0
        ref, frame = self.video.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        while (self.video.isOpened):
            ref, frame = self.video.read()
            if not ref:
                break
            if frame_index == 0: # 选择没有白车在停止线上的帧进行处理
                print('正在获取车道线信息')
                frame0 = frame.copy()
                savePath = 'index2.jpg'
                # cv2.imwrite(savePath, frame)
                image = Image.fromarray(frame)
                image_lx = predict_road_pixel(image)
                # image_lx.save("img_lx.jpg")
                image_lx = np.array(image_lx)
                self.dir = fit_lanes(frame, image_lx)
                print('成功获取车道线')

                self.roi_zone, self.scale = get_roi(self.dir)
                self.size = [self.size[0]*self.scale, self.size[1]*self.scale]
                # frame0 = roi_mask(frame, self.roi_zone)
                frame0 = draw_road_lines(frame, self.dir, {})
                cv2.imwrite('output_image.jpg', frame0)

                break


    def read_frame(self):
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        self.detection = detect_with_api_revise.detectapi(weights='weights/best407_att.pt')  # epoch_319.pt epoch_299.pt bestw6_exp75_386.pt   _att_396
        self.classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle',
                              'bus', 'e-bike']

        print('开始读取视频')
        frame_index = 0
        cap = cv2.VideoCapture(self.video_path)
        # print("cap:", cap.isOpened())

        t_pass = {}
        t_in = {}
        while(cap.isOpened()):
            ret, frame = self.video.read()
            print("index:", frame_index)
            if not ret and type(frame) == type(None):
                if 30 > frame_index > 10:
                    print('frame' + str(frame_index) + '出现了问题跳过')
                    frame_index = frame_index + 1
                    continue
                elif frame_index > 30:
                    print("Opening camera is failed or video error")
                    cv2.destroyAllWindows()
                    cap.release()
                    vid_writer.release()
                    break


            if not self.exit_flag:
                # while not self.exit_flag:
                # if not self.frame_queue.empty():
                # 从队列中获取一帧图像
                # frame = self.frame_queue.get()

                this_frame_track = {}
                this_frame_info = {}
                for dir_id in range(len(self.dir)):
                    this_frame_info[dir_id] = {}
                    for lane_id in range(1,len(self.dir[dir_id])-4):
                        this_frame_info[dir_id][lane_id] =  {'stopnumber':0,'counting_car':0, 'light_duration':0,'light': [0],'light_confidence':[0]}

                # frame_roi = roi_mask(frame, self.roi_zone)  # 预处理 切割输入模型的图像
                frame_roi = frame
                # t0 = time_synchronized()

                img, outputs = self.detection.detect_video([frame_roi])  # outputs list_name
                # self.counter_recording, self.counter, self.list_overlapping = counter_vehicles(outputs, polygon_mask,
                #                                                                                self.counter_recording,
                #                                                                                self.counter,
                #                                                                                self.list_overlapping)

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

                            # 轨迹记录
                            if track_id in self.pts:
                                self.pts[track_id].append(center)
                                if len(self.pts[track_id]) > 30:
                                    del self.pts[track_id][0]
                            else:
                                self.pts[track_id] = []
                                self.pts[track_id].append(center)

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


                if outputs is not None and len(outputs):
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]
                    update_frame_info = {}
                    for dir_id in range(len(self.dir)):
                        update_frame_info[dir_id] = {}
                        for lane_id in range(1, len(self.dir[dir_id]) - 4):
                            update_frame_info[dir_id][lane_id] = {'stopnumber':0,'counting_car':0, 'light_duration':0,'last_time':0,'min_dis': 620}
                            update_frame_info[dir_id][lane_id]['light'] = last_frame_info[dir_id][lane_id]['light']
                            update_frame_info[dir_id][lane_id]['light_confidence'] = last_frame_info[dir_id][lane_id]['light_confidence']
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
                                update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos, 'speed': (speed_x, speed_y), 'accelerate':a_x}
                            else:
                                update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,'speed': (speed_x, speed_y),'accelerate': a_y}
                            fit = self.dir[dir_id]['stop']['fit']
                            dis_car2stop = p2l_dis(this_frame_pos[0],this_frame_pos[1],fit)
                            update_frame_info[dir_id][lane_id]['counting_car'] = update_frame_info[dir_id][lane_id]['counting_car'] + 1

                            if direction == 1 or direction == 3:
                                if speed_x <= 10 or ( a_x is not None and a_x <= -1 and speed_x > 10):
                                    update_frame_info[dir_id][lane_id]['stopnumber'] = update_frame_info[dir_id][lane_id]['stopnumber'] + 1
                            else:
                                if speed_y <= 10 or (a_y is not None and a_y <= -1 and speed_y > 10):
                                    update_frame_info[dir_id][lane_id]['stopnumber'] = update_frame_info[dir_id][lane_id]['stopnumber'] + 1
                            if dis_car2stop < update_frame_info[dir_id][lane_id]['min_dis']:
                                update_frame_info[dir_id][lane_id]['min_dis'] = dis_car2stop
                                update_frame_info[dir_id][lane_id]['min_track_id'] = key
                        else:
                            # 插入 # 本帧位置
                            this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                            update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos, 'speed': (None,None),'accelerate':None}
                            this_frame_track[key]['speed'] = (None,None)



                    last_frame_info = update_frame_info
                    draw_boxes(frame, bbox_xyxy, [self.classes_names[i] for i in classes2], classes2, identities, this_frame_track)
                    # frame_f = draw_all_lines(frame, self.dir, last_frame_info, self.lights[frame_index])
                    frame_f = draw_all_lines(frame, self.dir, last_frame_info, {})

                else:
                    last_frame_info = this_frame_info
                    frame_f = draw_all_lines(frame, self.dir, last_frame_info, {})



                # CSV写
                if len(outputs) != 0 and frame_index % self.fps == 0:
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
                            if dir_id in t_pass:
                                if lane_id in t_pass[dir_id]:
                                    if len(t_pass[dir_id][lane_id]) > 1:
                                        delta_t = (t_pass[dir_id][lane_id][-1][0] - t_pass[dir_id][lane_id][-2][0]) / self.fps
                                    zhanyoulv_w = t_pass[dir_id][lane_id][-1][1] / t_pass[dir_id][lane_id][-1][0]
                                    zhanyoulv_w = round(zhanyoulv_w * 100, 4)
                            else:
                                delta_t = ''
                                zhanyoulv_w = ''

                            vx = last_frame_info[dir_id][lane_id][track_id]['speed'][0]
                            vy = last_frame_info[dir_id][lane_id][track_id]['speed'][1]
                            if vx is not None:
                                vx = round(vx,4)
                                vy = round(last_frame_info[dir_id][lane_id][track_id]['speed'][1],4)
                            with open(self.csvfile, 'a', newline='') as f:  # '2'+'.csv'
                                writer = csv.writer(f)
                                writer.writerow(
                                    splicing_csvdata(frames_to_timecode(self.fps, frame_index), track_id, cls_id,
                                                      dir_id, lane_id,
                                                      int(self.scale * (bbox_left + bbox_right) / 2),
                                                      int(self.scale * (bbox_top + bbox_bottom) / 2),
                                                      round(self.scale * abs(bbox_right - bbox_left),1),
                                                      round(self.scale * abs(bbox_bottom - bbox_top),1),
                                                      vx,
                                                      vy,
                                                      delta_t,
                                                      zhanyoulv_w
                                                     ))

                    f.close()

                    with open(self.csvfile1, 'a', newline='') as f2:  # '2'+'.csv'
                        for dir_id in range(len(self.dir)):
                            if dir_id in last_frame_info:
                                for lane_id in range(1, len(self.dir[dir_id]) - 4):
                                    if lane_id in last_frame_info[dir_id]:
                                        light = last_frame_info[dir_id][lane_id]['light']
                                        duration = last_frame_info[dir_id][lane_id]['light_duration'] / self.fps
                                        writer2 = csv.writer(f2)
                                        writer2.writerow(
                                            splicing_csvdata5(frames_to_timecode(self.fps, frame_index),
                                                              dir_id, lane_id, light, duration))
                    f2.close()



            # cv2.imshow('detect', frame_f)
            # cv2.imwrite('output/pic_cross/frame' + str(frame_index) + '.jpg', frame_f)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            vid_writer.write(frame_f)
            frame_index += 1


            # put frame into queue
            # self.frame_queue.put(frame)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filePath = 'source/20240829高架'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    files = os.listdir(filePath)
    for file in files:
        if os.path.splitext(file)[1] == '.mp4' or os.path.splitext(file)[1] == '.MP4':
        # if os.path.splitext(file)[1] == '.mp4':
            file_name = os.path.splitext(file)[0]
            source = os.path.join(filePath, file)
            live = Live(source)
            live.read_lane()
            vid_writer = cv2.VideoWriter('output/vid/' + file_name + '_output.mp4', fourcc, live.fps, (live.width, live.height))  # (2720, 1530)
            live.read_frame()
            break


