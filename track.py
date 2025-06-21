import argparse
import csv
import os
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from unet.highway_selectbeforefit import fit_lanes as fit_lanes_line
from unet.cross import fit_lanes as fit_lanes_cross
from unet.predict import predict_road_pixel

from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.loaders import LoadStreams, LoadImagesAndVideos
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.tasks import attempt_load_weights
from deepsort_YOLO11_cross.deep_sort_pytorch.utils.parser import get_config
from deepsort_YOLO11_cross.deep_sort_pytorch.deep_sort import DeepSort
from deepsort_YOLO11_cross.utils.utils import scale_coords, draw_cross_lines
from deepsort_YOLO11_cross.utils.main_utils import calculate_speedlane, frames_to_timecode
from deepsort_YOLO11_cross.param_calculate import *
from utils.hand_draw_utils import *
from utils.save_xml import write_roads, write_crosses

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def _tlwh_to_xyxy(bbox_tlwh,height,width):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    x, y, w, h = bbox_tlwh
    x1 = max(int(x), 0)
    x2 = min(int(x + w), width - 1)
    y1 = max(int(y), 0)
    y2 = min(int(y + h), height - 1)
    return x1, y1, x2, y2

def calculate_line_kb(points):
    """计算直线斜率和截距"""
    try:
        (x1, y1), (x2, y2) = points
        if x2 - x1 == 0:
            return float('inf'), x1  # 垂直线返回无穷大和x截距

        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return round(k, 4), round(b, 2)
    except (TypeError, ValueError) as e:
        print(f"坐标点格式错误: {e}")
        return None, None
    except ZeroDivisionError:
        print("警告：两点x坐标相同，垂直线")
        return float('inf'), x1

def calculate_category(point1,point2):
    """计算车道线横竖"""
    (x1, y1)= point1
    (x2, y2)=point2
    if abs(x1-x2)>abs(y1-y2):
        cate=0
    else:
        cate=1
    return cate

def calculate_direction(point1, point2, point3):
    (x1, y1) = point1
    (x2, y2) = point2
    (x3, y3) = point3
    if abs(x1-x2) > abs(y1-y2):
        if ((x1+x2)/2) < x3:
            d = 3
        else:
            d = 1
    else:
        if ((y1 + y2) / 2) < y3:
            d = 4
        else:
            d = 2
    return d

def estimateSpeed(location1, location2, scale, fps, dir_id, dir):
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
    a_x = (v2**2 - v1**2) / (2 * x)
    return a_x

def get_position_id(x,y, dir,roi):
    dir_index = None
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
            line_number = len(dir[dir_index]) - 9
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

def p2l_dis(x,y,fit):  # 点到直线的距离
    b = fit[1]
    k = fit[0]
    return abs(k * x - y + b) / math.sqrt(k ** 2 + 1)

def splicing_csvdata(list1, list2, list3, list4, list5, list6, list7, list8,list9, list10, list11):
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

    return temp

def xyxy_to_xywh(*xyxy):
    """ Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def compute_color_for_cls(cls):
    """
    Simple function that adds fixed color depending on the class
    """
    c = [(120, 120, 120), (0, 120, 120), (120, 0, 120), (120, 120, 0), (120, 0, 0)]
    return c[cls]


def draw_boxes(img, bbox, identities=None, cls=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        cls_id = int(cls[i]) if cls is not None else 0
        # color = compute_color_for_labels(id)
        color = compute_color_for_cls(cls_id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_lanes_in_pic(img, windows):
    print('Please mark the lane!')
    location = lanelabelmark(img, windows)
    print('The Location has been marked successfully ')
    print('lane(location)', location)
    # 标记路口测停止线（1条）
    print('Please mark the speed lane!')
    windows.systerm_status_echo("请绘制其停止线！")
    count_line = calculate_speedlane(img, windows)
    count_line = np.concatenate((count_line[0][0], count_line[1][0]))   # xyxy
    print('The speed lane has been marked successfully ')
    print('speedlane(speed_lane)', count_line)

    cars_in_lane = {}
    dir_1 = {}
    crosspoint = []


    for line_id in range(len(location[0])):
        if line_id != len(location[0]) - 1:
            cars_in_lane[line_id] = set()
        line = np.concatenate((location[0][line_id], location[1][line_id]))
        point = get_line_cross_point(line, count_line)
        crosspoint.append(point)
        dis2start = distance_2d(point, location[0][line_id])
        dis2end = distance_2d(point, location[1][line_id])
        endIndex = 0 if dis2start > dis2end else 1

        points = (point, list(location[endIndex][line_id]))
        k, b = calculate_line_kb(points)
        dir_1['Line' + str(line_id)] = {
            'type': 'single_line',
            'L1': {'lane_type': 'shi',
                   'fit': np.array([k, b]),
                   'points': points}
            }

    stop_line_points = (crosspoint[0], crosspoint[-1])
    fit = calculate_line_kb(stop_line_points)

    # === 车道宽度计算 ===
    lane_widths = []
    for i in range(len(crosspoint) - 1):
        # 计算相邻车道线间距（示例用欧式距离）
        # line1_start = crosspoint[i]
        # line2_start = crosspoint[i + 1]
        width = distance_2d(crosspoint[i],crosspoint[i + 1])
        lane_widths.append(width)

    # 取平均作为车道宽度
    mean_lane_width = np.mean(lane_widths) if lane_widths else 0

    # 记录每个车道内通行的车辆数
    traffic_volume = np.zeros((len(location[0]) - 1), np.int32)
    return{
            'cars': cars_in_lane,
            'traffic_volume': traffic_volume,
            'lane_num': len(traffic_volume),
            'lane_width': mean_lane_width,
            'stop': {  # 新增停止线参数
                'points': [crosspoint[0], crosspoint[-1]],
                'fit': fit
            },
            'crosspoint': crosspoint,
            'category': calculate_category(location[0][0], location[1][0]),
            'direction': calculate_direction(location[0][0], location[1][0], crosspoint[0]),
            'roi': [crosspoint[0], location[endIndex][0], location[endIndex][-1], crosspoint[-1]],
            **dir_1
            }

def fix_dir_data(dir_data):
    for dir_id in range(len(dir_data)):
        lane_num = sum(1 for key in dir_data[dir_id].keys() if key.startswith('Line')) - 1
        cars_in_lane = {i: set() for i in range(lane_num)}
        traffic_volume = np.zeros(lane_num, np.int32)

        dir_data[dir_id]['cars'] = cars_in_lane
        dir_data[dir_id]['traffic_volume'] = traffic_volume
        dir_data[dir_id]['lane_num'] = lane_num
    return dir_data


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
        # 假设dir[dir_id]是一个字典
        line_keys_count = sum(1 for key in dir[dir_id].keys() if key.startswith('Line'))
        x1 = dir[dir_id]['Line0']['L1']['points'][1][0]
        y1 = dir[dir_id]['Line0']['L1']['points'][1][1]
        xx1 = dir[dir_id]['Line' + str(line_keys_count-1)]['L1']['points'][1][0]
        yy1 = dir[dir_id]['Line' + str(line_keys_count-1)]['L1']['points'][1][1]

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
    mean_lane = width_sum / len(dir)  # 一个车道宽3.75m，对应的像素
    scale = 3.75 / mean_lane

    return ranges, scale

class VehicleTracker:
    def __init__(self, video_path, vid_save = False, car_track_save = False, car_num_save = False):
        self.out_num_dist = None
        self.dir_num = 0

        # 配置参数
        self.out = 'inference/output'
        self.source = video_path
        self.yolo_weights = 'weights/best.pt'
        self.deep_sort_weights = 'weights/ckpt.t7'
        self.config_deepsort = "config/deep_sort.yaml"
        self.device = "cpu"
        self.imgsz = (1920, 1080)
        self.conf_thres = 0.2
        self.iou_thres = 0.5
        self.classes = [0, 1, 2]
        self.agnostic_nms = False
        self.augment = False
        self.save_txt = False
        self.save_vid = vid_save
        self.save_csv = car_num_save
        self.save_csv_speed = car_track_save
        self.show_vid = False
        self.evaluate = False
        self.scale = 0

        self.webcam = None
        self.deepsort = None
        self.half = None
        self.model = None
        self.stride = None
        self.names = None
        self.letterbox = None
        self.dataset = None
        self.data_dir = None
        self.last_frame_info = None
        self.data_pts = None
        self.update_frame_info = None

        self.t0 = None
        # 视频保存器和路径
        self.csv_name = []
        self.vid_path, self.vid_writer = None, None
        self.csv_speed, self.vid_name = None, None

        # 初始化变量
        self.xmlfile = None
        self.data_dir = {}
        self.this_frame_info = {}
        self.last_frame_info = {}
        self.data_pts = {}
        self.update_frame_info = {}

        self.run_init_dir = False
        self.run_init_video = False

    def initialication(self, vid_save, car_track_save, car_num_save, out_path = None):
        self.save_vid = vid_save
        self.save_csv = car_num_save
        self.save_csv_speed = car_track_save
        if out_path is not None:
            self.out = out_path

        if self.run_init_video is True:
            return

        self.webcam = self.source == '0' or self.source.startswith(('rtsp', 'http')) or self.source.endswith('.txt')

        # 数据集加载器
        if self.webcam:
            cudnn.benchmark = True
            self.dataset = LoadStreams(self.source)
        else:
            self.dataset = LoadImagesAndVideos(self.source)
        self.run_init_video = True

    def Data_Dir_Import(self, data):
        self.data_dir = data
        self.run_init_dir = True
        self.scale = 3.75 / self.data_dir[0]['lane_width']
        self.dir_num = len(data)

    def Hand_Draw(self, dir_num, windows):
        if self.run_init_dir:
            raise ValueError("程序异常，请联系开发者(Hand_Draw)")

        self.dir_num = dir_num

        for frame_idx, (path, im0s, _) in enumerate(self.dataset):
            im0s = np.array(im0s).squeeze()
            for i in range(self.dir_num):
                windows.systerm_status_echo(f"请绘制第{i + 1}条道路的车道线")
                self.data_dir[i] = draw_lanes_in_pic(im0s, windows)
            self.scale = 3.75 / self.data_dir[0]['lane_width']
            break  # 仅初始化第一帧即可

        self.run_init_dir = True

        return self.data_dir

    def Segmentation(self):
        # 文件加载
        video = cv2.VideoCapture(self.source)

        # 保存路径根地址
        frame_index = 0
        dir_data, dir_num_data, out_num, scale = None, None, None, None
        ref, frame = video.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        while video.isOpened:
            ref, frame = video.read()
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
                # image_l = np.array(image_l)
                # image_x = np.array(image_x)
                image_lx = np.array(image_lx)
                dir_data, out_num, size = fit_lanes_line(image_s, frame, image_lx)
                roi_zone, scale = get_roi(dir_data)
                # size = [size[0] * self.scale, size[1] * self.scale]

                break
        video.release()
        if dir_data is not None:
            dir_data = fix_dir_data(dir_data)
            self.data_dir = dir_data
            self.dir_num = len(dir_data)
            self.out_num_dist = out_num
            self.scale = scale
            self.run_init_dir = True

    def Segmentation_Cross(self):
        # 文件加载
        video = cv2.VideoCapture(self.source)

        # 保存路径根地址
        frame_index = 0
        dir_data, dir_num_data, out_num, scale = None, None, None, None
        ref, frame = video.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        while video.isOpened:
            ref, frame = video.read()
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
                dir_data, out_num, size = fit_lanes_cross(image_s, frame, image_l, image_x)
                roi_zone, scale = get_roi(dir_data)
                # size = [size[0] * scale, size[1] * scale]
                # frame0 = roi_mask(frame, self.roi_zone)
                # frame = draw_road_lines(frame, self.dir,'')
                # cv2.imwrite('output_image.jpg', frame)

                break
        video.release()
        if dir_data is not None:
            dir_data = fix_dir_data(dir_data)
            self.data_dir = dir_data
            self.dir_num = len(dir_data)
            self.out_num_dist = out_num
            self.scale = scale
            self.run_init_dir = True

    def run(self):
        if self.run_init_video == False or self.run_init_dir == False:
            raise -1

        self.t0 = time.time()
        file_name = os.path.splitext(Path(self.source).name)[0]

        # 初始化CSV路径
        for i in range(self.dir_num):
            self.csv_name.append(f"{self.out}/{file_name}_dir{i}_{datetime.datetime.now():%Y-%m-%d_%H.%M.%S}.csv")
        self.csv_speed = f"{self.out}/{file_name}_speed_{datetime.datetime.now():%Y-%m-%d_%H.%M.%S}.csv"
        self.vid_name = f"{self.out}/{file_name}.mp4"

        # 初始化csv文件
        if self.save_csv:
            for i in range(self.dir_num):
                with open(self.csv_name[i], 'w', newline='') as f:
                    fnames = ['时间'] + [f"车道{j}车流量" for j in range(len(self.data_dir[i]['traffic_volume']))]
                    writer = csv.DictWriter(f, fieldnames=fnames)
                    writer.writeheader()
        if self.save_csv_speed:
            with open(self.csv_speed, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "vehicle_id", "vehicle_type", "dir_id", "lane_id", "position_x(m)",
                                 "position_y(m)", "long(m)", "height(m)", "xVelocity(km/h)", "yVelocity(km/h)"])


        # DeepSORT 初始化
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        attempt_download_asset(self.deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=False)

        # 设备初始化
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'

        self.model = attempt_load_weights(self.yolo_weights, device=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_imgsz(self.imgsz, stride=self.stride)
        if self.half:
            self.model.half()

        # 预加载一帧用于加速
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz[0], self.imgsz[1]).to(self.device).type_as(next(self.model.parameters())))

        # 类别名
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.letterbox = LetterBox(self.imgsz, auto=True, stride=self.stride)

        # 初始化update_frame_info结构
        self.update_frame_info = {
            dir_id: {
                lane_id: {} for lane_id in range(1, self.data_dir[dir_id]['lane_num'] + 1)
            } for dir_id in range(len(self.data_dir))
        }

        frame_count = 0
        for frame_idx, (path, im0s, _) in enumerate(self.dataset):
            print('当前是' + str(frame_idx) + '帧')
            frame_count += 1
            # 每五帧yield一次进度信息
            if frame_count % 10 == 0:
                pass
                # yield f"正在处理第{frame_count}帧"
            fps = self.dataset.fps

            path = str(path[0])
            im0s = np.array(im0s).squeeze()
            # vertices = []
            # directions = []
            # for i in range(dir_num):
            #     vertices.append([np.array(data_dir[i]['roi'], dtype=np.int32)])
            #     directions.append(data_dir[i]['direction'])
            # im0s_roi = roi_mask(im0s, vertices)
            im0s_roi = roi_max_mask(im0s, self.data_dir)
            # cv2.imwrite("roi.jpg", im0s_roi)

            # im0s_roi = im0s

            img_roi = self.letterbox(image=im0s_roi)
            img_roi = img_roi[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img_roi = np.ascontiguousarray(img_roi)

            # cv2.imwrite("curim0.jpg",im0s)
            img_roi = torch.from_numpy(img_roi).to(self.device)
            img_roi = img_roi.half() if self.half else img_roi.float()  # uint8 to fp16/32
            img_roi /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_roi.ndimension() == 3:
                img_roi = img_roi.unsqueeze(0)

            # Inference
            pred = self.model(img_roi, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            outputs = None

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img_roi.shape[2:]  # print string

                # 判断流量信息
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img_roi.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    xywh_bboxs = []
                    confs = []
                    clss = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        clss.append(cls.item())

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    clsss = torch.tensor(clss, dtype=torch.int64)

                    outputs = self.deepsort.update(xywhs, confss, clsss, im0)
                    data_dir = get_lane_id(outputs, self.data_dir)
                    traffic_volume_count(data_dir)
                else:
                    self.deepsort.increment_ages()

                this_frame_track = {}
                this_frame_info = {}
                for dir_id in range(len(self.data_dir)):
                    this_frame_info[dir_id] = {}
                    for lane_id in range(1, self.data_dir[dir_id]['lane_num'] + 1):
                        this_frame_info[dir_id][lane_id] = {}

                if outputs is not None and len(outputs):
                    for item_bbox in outputs:
                        x1, y1, x2, y2, cls_id, track_id = item_bbox

                        y = int((y1 + y2) / 2)
                        x = int((x1 + x2) / 2)
                        center = (x, y)
                        dir_id, lane_id = get_cur_lane_id(center, self.data_dir)

                        if dir_id is not None and lane_id is not None:
                            this_frame_track[track_id] = {'dir_id': dir_id, 'lane_id': lane_id}
                            this_frame_info[dir_id][lane_id][track_id] = {'last_pos': center, 'speed': (None, None)}

                            # 轨迹记录
                            if track_id in self.data_pts:
                                self.data_pts[track_id].append(center)
                                if len(self.data_pts[track_id]) > 30:
                                    del self.data_pts[track_id][0]
                            else:
                                self.data_pts[track_id] = []
                                self.data_pts[track_id].append(center)

                    for key, val in this_frame_track.items():
                        dir_id = this_frame_track[key]['dir_id']
                        lane_id = this_frame_track[key]['lane_id']
                        direction = self.data_dir[dir_id]['direction']
                        if key in self.last_frame_info[dir_id][lane_id]:
                            # 更新
                            # 本帧位置
                            this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                            # 上帧位置
                            last_frame_pos = self.last_frame_info[dir_id][lane_id][key]['last_pos']
                            # 上帧速度
                            last_frame_vx = self.last_frame_info[dir_id][lane_id][key]['speed'][0]
                            last_frame_vy = self.last_frame_info[dir_id][lane_id][key]['speed'][1]
                            # 计算距离
                            speed_x, speed_y = estimateSpeed(this_frame_pos, last_frame_pos, self.scale, fps, dir_id,
                                                             self.data_dir)
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
                                self.update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,
                                                                           'speed': (speed_x, speed_y),
                                                                           'accelerate': a_x}
                            else:
                                self.update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,
                                                                           'speed': (speed_x, speed_y),
                                                                           'accelerate': a_y}

                        else:
                            # 插入 # 本帧位置
                            this_frame_pos = this_frame_info[dir_id][lane_id][key]['last_pos']
                            self.update_frame_info[dir_id][lane_id][key] = {'last_pos': this_frame_pos,
                                                                       'speed': (None, None), 'accelerate': None}
                            this_frame_track[key]['speed'] = (None, None)

                    self.last_frame_info = self.update_frame_info
                else:
                    self.last_frame_info = this_frame_info

            if self.save_vid:
                # -- 标志线标注 --
                draw_cross_lines(im0, self.data_dir)
                # -- detection 图像标注 --
                if outputs is not None and len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # draw_boxes(im0, bbox_xyxy, identities)
                    clses = outputs[:, -2]
                    draw_boxes(im0, bbox_xyxy, identities, clses)

                    if self.save_csv_speed:
                        # 画车辆行驶轨迹
                        for item_bbox in outputs:
                            x1, y1, x2, y2, cls_id, track_id = item_bbox
                            if track_id not in self.data_pts:
                                continue
                            y = int((y1 + y2) / 2)
                            x = int((x1 + x2) / 2)
                            center = (x, y)

                            thickness = 3
                            cv2.circle(im0, center, 1, [255, 255, 255], thickness)
                            if len(self.data_pts[track_id]) < 30:
                                draw_coords = self.data_pts[track_id]
                            else:
                                draw_coords = self.data_pts[track_id][-30:]
                            if len(draw_coords) > 1:
                                for j in range(1, len(draw_coords)):
                                    if draw_coords[j - 1] is None or draw_coords[j] is None:
                                        continue
                                    cv2.line(im0, (draw_coords[j - 1]), (draw_coords[j]), [0, 255, 255], thickness)
                # # -- 车流量标注 --
                # for i in range(len(traffic_volume)):
                #     cv2.putText(im0, f"{traffic_volume[i]}", (300 + 150 * i, 40), cv2.FONT_HERSHEY_SIMPLEX,
                #                 1,
                #                 (255, 255, 0), 2)
                # -- FPS标注 --
                im0 = cv2.putText(im0, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 255, 0), 2)

            if self.save_csv_speed:
                for j, output in enumerate(outputs):
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_right = output[2]
                    bbox_bottom = output[3]
                    track_id = output[-1]
                    cls_id = output[-2]
                    if track_id in this_frame_track:
                        dir_id = this_frame_track[track_id]['dir_id']
                        lane_id = this_frame_track[track_id]['lane_id']

                        vx = self.last_frame_info[dir_id][lane_id][track_id]['speed'][0]
                        vy = self.last_frame_info[dir_id][lane_id][track_id]['speed'][1]
                        if vx is not None:
                            vx = round(vx, 4)
                            vy = round(self.last_frame_info[dir_id][lane_id][track_id]['speed'][1], 4)
                        with open(self.csv_speed, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                splicing_csvdata(frames_to_timecode(fps, frame_idx), track_id, cls_id,
                                                 dir_id, lane_id,
                                                 int(self.scale * (bbox_left + bbox_right) / 2),
                                                 int(self.scale * (bbox_top + bbox_bottom) / 2),
                                                 round(self.scale * abs(bbox_right - bbox_left), 1),
                                                 round(self.scale * abs(bbox_bottom - bbox_top), 1),
                                                 vx,
                                                 vy
                                                 ))

            # Stream results
            if self.show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            if self.save_vid:
                if self.vid_path != self.vid_name:  # new video
                    self.vid_path = self.vid_name
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer
                    w, h = im0.shape[1], im0.shape[0]
                    self.vid_writer = cv2.VideoWriter(self.vid_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.vid_writer.write(im0)

            if self.save_csv and frame_idx % fps == 0:
                for dir_id in range(self.dir_num):
                    traffic_volume = self.data_dir[dir_id]['traffic_volume']
                    fnames = []
                    fnames.append('时间')
                    for i in range(len(traffic_volume)):
                        fnames.append('车道' + str(i) + '车流量')
                    data = {}
                    with open(self.csv_name[dir_id], 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fnames)
                        # 车道统计
                        data['时间'] = frames_to_timecode(fps, frame_idx)
                        for i in range(len(traffic_volume)):
                            data['车道' + str(i) + '车流量'] = str(traffic_volume[i])
                        writer.writerow(data)


        print('Done. (%.3fs)' % (time.time() - self.t0))
        outstr = "运行结束\n"
        file_name = os.path.splitext(Path(self.source).name)[0]
        if self.save_vid:
            self.vid_writer.release()
            outstr += f"视频结果保存在{os.path.abspath(f'{self.out}/{file_name}.mp4')}\n"
        if self.save_csv_speed:
            outstr += f"轨迹结果保存在{os.path.abspath(f'{self.out}/{file_name}_speed_{datetime.datetime.now():%Y-%m-%d_%H.%M.%S}.csv')}\n"
        # 车流量统计csv
        if self.save_csv:
            for i in range(self.dir_num):
                csvfilepath = os.path.abspath(f"{self.out}/{file_name}_dir{i}_{datetime.datetime.now():%Y-%m-%d_%H.%M.%S}.csv")
                outstr += f"流量结果保存在{csvfilepath}\n"
        print('Results saved to %s' % self.out)

        if outstr.endswith('\n'):
            outstr = outstr.rstrip('\n')

        yield outstr

    def save_xml(self, out_path = None, mode = 0):
        file_name = os.path.splitext(os.path.basename(self.source))[0]
        if out_path is not None:
            self.xmlfile = out_path + '/' + file_name + '_lane.xml'
        else:
            self.xmlfile = './output/' + file_name + '_lane.xml'

        directory = os.path.dirname(self.xmlfile)
        if not os.path.exists(directory):
            os.makedirs(directory)

        road_rules = {  # dir_id:lane_id:rule
            0: {1: ['left', 'straight'], 2: ['right']},
            1: {1: ['left'], 2: ['left'], 3: ['straight'], 4: ['straight', 'right']},
            2: {1: ['left'], 2: ['right'], 3: ['straight', 'right']},
            3: {1: ['left'], 2: ['straight'], 3: ['right']}
        }

        if mode == 0:
            write_roads(self.data_dir, self.scale, self.xmlfile)
        else:
            write_crosses(self.data_dir, road_rules, self.out_num_dist, self.scale, self.xmlfile)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=r'source/out_c.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output/out_c', help='output folder')  # output folder
    parser.add_argument('--img-size', type=tuple, default=(1920,1080), help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=False, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--save-csv', default=False, action='store_true')
    parser.add_argument('--save-csv-speed', default=False, action='store_true')
    parser.add_argument('--classes', nargs='+', default=[0, 1, 2], type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_imgsz(args.img_size)


    dir_num_input = 3
    with torch.no_grad():
        pass
