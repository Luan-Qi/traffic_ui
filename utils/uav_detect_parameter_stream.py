import csv
import math
import queue
import threading
from pathlib import Path

import cv2
import time
import numpy as np

import detect_with_api_revise
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
# 初始化摄像头和Yolo模型
from main_utils import counter_vehicles, splicing_csvdata8, frames_to_timecode
from visdrone_lane_volume import detect
from yolov7.utils.torch_utils import time_synchronized


class Live(object):
    def __init__(self, flag=False):
        self.frame_queue = queue.Queue(maxsize=100)
        self.exit_flag = flag
        # 自行设置
        self.rtmpUrl = "rtmp://119.45.71.40:1935/live/"
        self.video_path = 'source/DJI_0009_22_broken.mp4'
        # self.camera_path = "source/my111.avi"
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        self.detection = detect_with_api_revise.detectapi(weights='weights/epoch_299.pt')  # epoch_319.pt  best407_att.pt
        self.classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle',
                              'bus', 'e-bike']
        self.frame_idx = 0
        self.video = cv2.VideoCapture(self.rtmpUrl)
        self.fps = round(self.video.get(cv2.CAP_PROP_FPS))
        self.counter = [[[0 for m in range(len(classes_names))] for i in range(num_crossing)] for j in
                        range(num_crossing)]  #

        self.list_overlapping = {}
        self.counter_recording = []

    def read_frame(self):
        print("开启拉流")
        # cap = cv2.VideoCapture(self.rtmpUrl)  # self.camera_path'video/1280.mp4'uav0011.mp4 source/my111.aviDJI_0010.mp4
        cap = cv2.VideoCapture(self.rtmpUrl)
        # Get video information

        print(self.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # read webcamera
        print("cap:", cap.isOpened())
        while (cap.isOpened()):
            ret, frame = cap.read()
            if type(frame) == type(None) or not ret:
                print("Opening camera is failed or video error")
                cv2.destroyAllWindows()
                cap.release()
                self.exit_flag = True
                break

            # put frame into queue
            self.frame_queue.put(frame)

    def process_frame(self):
        # 防止多线程时 command 未被设置
        # while True:
        #     if len(self.command) > 0:
        #         # 管道配置
        #         p = sp.Popen(self.command, stdin=sp.PIPE)
        #         break
        print("flag", self.exit_flag)

        while not self.exit_flag:
            if not self.frame_queue.empty():
                # 从队列中获取一帧图像
                frame = self.frame_queue.get()
                self.frame_idx += 1
                print("index:", self.frame_idx)
                # t0 = time_synchronized()
                img, outputs, _ = self.detection.detect_stream([frame])  # outputs list_name
                # self.counter_recording, self.counter, self.list_overlapping = counter_vehicles(outputs, polygon_mask,
                #                                                                 self.counter_recording, self.counter,
                #                                                                 self.list_overlapping)
                self.frame_idx += 1  # first fps space is null
                self.this_frame_info = {}
                self.last_frame_info = {}
                if self.frame_idx > 0 and len(outputs) > 0: # add ......
                    for item_bbox in outputs[:, :5]:
                        x1, y1, x2, y2, track_id = item_bbox
                        # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                        y1_offset = int(y1 + ((y2 - y1) * 0.5))
                        x1_offset = int(x1 + ((x2 - x1) * 0.5))
                        head_pos = x1_offset, y1_offset
                        self.this_frame_info[track_id] = {'last_pos': head_pos, 'speed': 0}
                im0 = self.post_process(img, outputs, self.last_frame_info)
                cv2.imshow('Counting Demo', im0)  # output_image_frame
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                # 你处理图片的代码
                # write to pipe
                # p.stdin.write(frame.tostring())

    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.process_frame, args=(self,))
        ]
        [thread.setDaemon(False) for thread in threads]
        [thread.start() for thread in threads]

    def post_process(self, output_image_frame, outputs, last_frame_info):
        if len(outputs) > 0:  # >0   or type(outputs) != type(None)  # outputs is not None bad
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            classes2 = outputs[:, -1]
            update_frame_info = {}
            for key, val in self.this_frame_info.items():
                if key in last_frame_info:
                    # 更新
                    # 本帧位置
                    this_frame_pos = val['last_pos']
                    # 上帧位置
                    last_frame_pos = last_frame_info[key]['last_pos']
                    # 计算距离
                    speed = estimateSpeed1(this_frame_pos, last_frame_pos, 30)

                    # 速度
                    # speed = distance * 3.6
                    update_frame_info[key] = {'last_pos': this_frame_pos, 'speed': speed}
                else:
                    # 插入 # 本帧位置

                    this_frame_pos = val['last_pos']
                    update_frame_info[key] = {'last_pos': this_frame_pos, 'speed': 0}

            last_frame_info = update_frame_info
            im0 = cv2.add(output_image_frame, color_polygons_image)
            draw_boxes(im0, bbox_xyxy, [classes_names[i] for i in classes2], classes2, identities,
                       last_frame_info)  # names
            # draw_counter(im0, self.counter, classes_names, direction)
            return im0
        else:
            last_frame_info = self.this_frame_info
            return output_image_frame
        # 写车道流量
        # CSV写

        csvfile = 'test_speed' + '20.csv'  # file[0:len(file) - 4] + '_' +
        # f = open(csvfile, 'a', newline='')
        # writer = csv.writer(f)

        save_txt = True
        # Write MOT compliant results to file
        if save_txt and self.frame_idx % 30 == 0:  # and len(outputs) != 0    % self.fps
            for j, output in enumerate(outputs):
                bbox_left = output[0]
                bbox_top = output[1]
                bbox_right = output[2]
                bbox_bottom = output[3]
                identity = output[-2]
                cls_id = output[-1]
                # print(type(cls_id),type(identity),type(bbox_left))

                with open(txt_path, 'a') as f:
                    f.write(('%g ' * 10 + '\n') % (self.frame_idx, identity, bbox_left, bbox_top,
                                                   bbox_right, bbox_bottom, (bbox_left + bbox_right) / 2,
                                                   (bbox_top + bbox_bottom) / 2, cls_id, -1))  # label format
                with open(csvfile, 'a', newline='') as f:  # '2'+'.csv'
                    writer = csv.writer(f)
                    # writer.writerow(
                    #     splicing_csvdata7(identity, cls_id, frames_to_timecode(30, frame_idx),
                    #                      (bbox_left + bbox_right) / 2, (bbox_top + bbox_bottom) / 2,
                    #                      abs(bbox_right-bbox_left), abs(bbox_bottom-bbox_top)))
                    writer.writerow(
                        splicing_csvdata8(identity, cls_id, frames_to_timecode(self.fps, self.frame_idx),
                                          (bbox_left + bbox_right) / 2, (bbox_top + bbox_bottom) / 2,
                                          abs(bbox_right - bbox_left),
                                          abs(bbox_bottom - bbox_top), round(update_frame_info[identity]['speed'], 2)))
                    # max(center_w_h[2], center_w_h[3]),
                    # min(center_w_h[2], center_w_h[3])))  # int(frame_idx % fps) [classes_names[i] for i in cls_id]
                    # f.close()
            f.close()


if __name__ == '__main__':
    classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
                     'e-bike']
    num_crossing = 4
    carWidht = 4  # 1.85
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    txt_path = str(Path('output')) + '/results1.txt'
    width = 1920
    height = 1080
    # width = 2720
    # height = 1530
    # num_crossing = 4
    list_pts = [[[400, 500], [400, height - 100], [500, height - 100], [501, 500]],
                [[501, 400], [500, 500], [width - 401, 500], [width - 401, 400]],
                [[width - 300, 500], [width - 300, height - 100], [width - 400, height - 100], [width - 400, 500]],
                [[501, height - 120], [501, height - 20], [width - 401, height - 20],
                 [width - 401, height - 120]]]  # dji_10
    last_frame_info = {}
    # list_pts = [[[2126, 1101], [2206, 1706], [2351, 1704], [2256, 1107]], [[2503, 985], [2566, 1073], [2661, 1021], [2600,956]],
    # [[2054, 1042], [2334, 1274], [2226, 1329], [1987, 1099]], [[2068, 1457], [2160, 1523], [2345, 1349], [2259, 1281]]]

    # list_pts = [[[1050, 400], [1050, height - 500], [1100, height - 500], [1100, 400]],
    #             [[1100, 400], [1100, 350], [width - 1150, 350], [width - 1150, 400]],
    #             [[width - 1100, 400], [width - 1100, height - 550], [width - 1150, height - 550], [width - 1150, 400]],
    #             [[1100, height - 450], [1100, height - 500], [width - 1150, height - 500], [width - 1150, height - 450]]] #dji_12
    print('从我这里开始，检测线确定')
    direction = ['West', 'North', 'East', 'South']
    color = [[255, 0, 0],
             [0, 255, 0],
             [0, 0, 255],
             [255, 255, 0]]
    # 填充第一个撞线polygon（蓝色）,绿，红，白[255, 255, 255]

    polygon_mask = np.zeros((height, width, 1), dtype=np.uint8)
    color_polygons_image = np.zeros((height, width, 3), dtype=np.uint8)
    for num in range(num_crossing):
        # 填充第二个撞线polygon（黄色）
        mask_image_temp = np.zeros((height, width), dtype=np.uint8)

        ndarray_pts_yellow = np.array(list_pts[num], np.int32)
        polygon_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=num + 1)
        polygon_value = polygon_value[:, :, np.newaxis]

        # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask = polygon_value + polygon_mask

        # polygon_mask1 = cv2.resize(polygon_mask, (width, height))

        image = np.array(polygon_value * color[num], np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = color_polygons_image + image

        # 缩小尺寸
        # color_polygons_image = cv2.resize(color_polygons_image, (width, height))

    live = Live()
    live.run()

    # source = 'source/DJI_0010.mp4'
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # vid_writer = cv2.VideoWriter(
    #     "cars18_output.MP4", fourcc, 30, (1920, 1080))


def estimateSpeed1(location1, location2, fps):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    carWidht = 4
    ppm = 150 / carWidht  # pixel width/car width  #  pix_wid small then h high
    d_meters = d_pixels / ppm
    speed = 3.6 * d_meters * fps  # mySpeed +
    return speed


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


def draw_boxes(img, bbox, cls_names, classes2, identities=None, last_ids_info={}, offset=(0, 0)):
    this_ids_info = last_ids_info
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i] * 100))
        label = '%d %s' % (id, cls_names[i])
        if id in this_ids_info and this_ids_info[id]['speed'] != 0 and cls_names[i] == 'car':
            speed = round(this_ids_info[id]['speed'], 1)
            label = '%d %s %s km/h' % (id, cls_names[i], speed)
        # label = '%d %s' % (id, '')
        # label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
