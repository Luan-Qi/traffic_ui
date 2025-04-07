import os
import torch
import torch.backends.cudnn as cudnn
from utils.main_utils import *
import shutil
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)  # , plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from pathlib import Path
# import dlib
import time
import threading
import math
import numpy as np
import datetime
from PIL import Image
from utils.main_utils import lanemark as lm
# from yolo_track2 import YOLO
# from yolov7 import detect
import csv
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
import sys

sys.path.insert(0, './yolov7')
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


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


def splicing_csvdata(list1, list2, list3, list4, list5):
    temp = []
    temp.append(list1)
    temp = temp + list2
    temp = temp + list3
    temp.append(list4)
    temp.append(list5)
    return temp


# 计算数组插值
def splicing_csvdata2(list1, list2, list3, list4, list5):
    temp = []
    temp.append(list1)
    temp = temp + list2
    temp = temp + list3
    temp = temp + list4
    temp = temp + list5
    return temp

def splicing_csvdata3(list1, list2): # , list3, list4, list5
    temp = []
    temp.append(list1)
    temp = temp + list2
    # temp = temp + list3
    # temp = temp + list4
    # temp = temp + list5
    return temp

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
def lane_cross(speedlane, lane):
    temp = []
    cross = []
    # temp_cross = [] * np.array(lane[0]).shape[0]
    for i in range(np.array(lane[0]).shape[0]):
        temp = []
        for j in range(np.array(speedlane[0]).shape[0]):
            temp_speed = speedlane[0][j] + speedlane[1][j]
            temp_lane = lane[0][i] + lane[1][i]
            temp_cross = get_line_cross_point(temp_speed, temp_lane)
            temp.append(temp_cross)
        cross.append(temp)
    cross.sort(key = lambda cross : cross[0][0])
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


def calculate_ylane():
    data = lm.lanemark(img)
    datasize = np.array(data[0]).shape[0]
    print('datasize', datasize)

    while datasize != 1:
        print('Wrong!! Format error! \n')
        print('Please try again \n')
        data = lm.lanemark(img)
        datasize = np.array(data[0]).shape[0]
        print('datasize', datasize)

    ylane = (data[0][0][1] + data[1][0][1]) // 2

    return ylane


def calculate_speedlane():
    data = lm.lanemark(img)
    datasize = np.array(data[0]).shape[0]
    # print('datasize', datasize)

    while datasize != 1:
        print('Wrong!! Format error! \n')
        print('Please try again \n')
        data = lm.lanemark(img)
        datasize = np.array(data[0]).shape[0]
    return data


def calculate_stoplane():
    stop = lm.lanemark(img)
    datasize = np.array(stop[0]).shape[0]
    # print('datasize', datasize)

    while datasize != 1:
        print('Wrong!! Format error! \n')
        print('Please try again \n')
        stop = lm.lanemark(img)
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
    cv2.fillPoly(mask, vertices, mask_color)  # 使用白色填充多边形，形成蒙板
    masked_img = cv2.bitwise_and(img, mask)  # img&mask，经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
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


def trackMultipleObjects(data, num):
    # fps=25
    fps =30
    frameCounter = 0
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2') ####
    outVideo = cv2.VideoWriter(
        "save_file\\cars2_output.MP4", fourcc, fps, size)  # 第一个参数是保存视频文件的绝对路径
    while True:

        start_time = time.time()
        read_flag, image = video.read()

        if type(image) == type(None) or not read_flag:
            print('video error')
            break

        # if pic_num % jumpfps == 0:
        resultImage = image.copy()
        image = roi_mask(image, roi_vtx)  # 经过此操作后，兴趣区域以外的部分被掩盖，只留下兴趣区域的图像
        # cv2.imencode('.jpg', image)[1].tofile(result_path1 + str(pic_num) + '.jpg')
        frameCounter = frameCounter + 1
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = Image.fromarray(np.uint8(rgb))
        with torch.no_grad():
            cars = detect(rgb)
            # cars, tags = yolo.detect_image(rgb) , tags, classes
            # cars = cars.astype(np.uint32)
            # tags = tags.astype(np.uint32)
            # classes = classes.astype(np.uint32)


            # 图片保存
            # cv2.imencode('.jpg', resultImage)[1].tofile(result_path2 + str(pic_num) + '.jpg')
            # img_copy = resultImage.copy()
            # 控制台输出
            # print('\r',"file " + format(filenum) + " progressing " + format(round((pic_num + 1) * 100 / frames_num , 2)) + "%", end='', flush=True)
        # print('\r', "file " + format(file) + " progressing " + format(pic_num) + "  ", end='', flush=True)
        # cv2.imencode('.jpg', resultImage)[1].tofile(result_path + str(pic_num) + '.jpg')
        # pic_num = pic_num + 1  # local variable 'pic_num' referenced before assignment

        # else:
        #     pic_num = pic_num + 1

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        end_time = time.time()
        seconds = end_time - start_time
        fps = 1 / seconds
        print(f'\r nowfps = {fps}')
        # outVideo.write(resultImage)
    video.release()
    cv2.destroyAllWindows()


def detect(img):
    # out, source, weights, view_img, save_txt, imgsz = \
    #     opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source = 'video/DJI_0026.mp4' # 'source/DJI_0012.mp4'video/out_6.avi
    out = 'output'
    # weights = 'weights/best407_att.pt'
    weights = 'weights/best319.pt'
    imgsz = 1920 # 1080
    rectangleColor = (0, 255, 0)
    frameCounter = 0

    k = np.array(location[0]).shape[0] - 1  # 车道数量（非车道线数量）
    num = k
    lane = [[] for _ in range(num)]
    stop = [[] for _ in range(num)]

    carnumlane = [0 for _ in range(num)]

    flag = True

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    frame_index = -2
    ##获得视频的帧宽高
    capture = cv2.VideoCapture(source)
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pic_num = 0

    # Initialize
    device = select_device('0') #opt.device   'cpu'
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
                     'e-bike']
    class_name = dict(zip(list(range(len(classes_names))), classes_names))

    ###设置计数器

    width = frame_fature[0]
    height = frame_fature[1]
    num_crossing = 4
    # list_pts = [[[400, 500], [400, height-100], [500, height-100], [501, 500]], [[501, 400], [500, 500], [width-401, 500], [width-401, 400]],
    #             [[width-300, 500], [width-300, height-100], [width-400, height-100], [width-400, 500]], [[501, height-100], [501, height], [width-401, height], [width-401, height-100]]]
    # list_pts = [[[2126, 1101], [2206, 1706], [2351, 1704], [2256, 1107]], [[2503, 985], [2566, 1073], [2661, 1021], [2600,956]],
    # [[2054, 1042], [2334, 1274], [2226, 1329], [1987, 1099]], [[2068, 1457], [2160, 1523], [2345, 1349], [2259, 1281]]]
    list_pts = [[[1050, 400], [1050, height - 500], [1100, height - 500], [1100, 400]],
                [[1100, 400], [1100, 350], [width - 1150, 350], [width - 1150, 400]],
                [[width - 1100, 400], [width - 1100, height - 550], [width - 1150, height - 550], [width - 1150, 400]],
                [[1100, height - 450], [1100, height - 500], [width - 1150, height - 500], [width - 1150, height - 450]]]
    print('检测线确定')

    # direction = ['West', 'North', 'East', 'South']
    # color = [[255, 0, 0],
    #          [0, 255, 0],
    #          [0, 0, 255],
    #          [255, 255, 255]]
    # 填充第一个撞线polygon（蓝色）,绿，红，白
    # counter = [[[0 for m in range(len(names))] for i in range(num_crossing)] for j in range(num_crossing)]
    # polygon_mask = np.zeros((height, width, 1), dtype=np.uint8)
    # color_polygons_image = np.zeros((height, width, 3), dtype=np.uint8)
    # list_overlapping = {}
    # counter_recording = []
    # for num in range(num_crossing):
    #     # 填充第二个撞线polygon（黄色）
    #     mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    #
    #     ndarray_pts_yellow = np.array(list_pts[num], np.int32)
    #     polygon_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=num + 1)
    #     polygon_value = polygon_value[:, :, np.newaxis]
    #
    #     # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    #     polygon_mask = polygon_value + polygon_mask
    #
    #     # polygon_mask1 = cv2.resize(polygon_mask, (width, height))
    #
    #     image = np.array(polygon_value * color[num], np.uint8)
    #
    #     # 彩色图片（值范围 0-255）
    #     color_polygons_image = color_polygons_image + image
    #
    #     # 缩小尺寸
    #     # color_polygons_image = cv2.resize(color_polygons_image, (width, height))

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    pts = {}
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS(非极大值抑制)
        pred = non_max_suppression(pred, 0.5, 0.6, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], agnostic=True)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size(用scale_coords函数来将图像缩放)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes_names[int(c)])  # add to string # names

                bbox_xywh = []
                confs = []
                classes = []
                speed = []
                img_h, img_w, _ = im0.shape
                print(im0.shape)
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = main_utils.bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])


                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, classes)
                tags = classes
                print('FPS:', round(cv2.CAP_PROP_FPS))
                frame_index += 1

                if outputs is not None:
                    for idx, datainoutput in enumerate(outputs):
                        if frameCounter % 1 == 0:
                            [x1, y1, x2, y2, i, classes_idx] = datainoutput

                            w1 = x2 - x1
                            h1 = y2 - y1

                            counpoint = (x1 + w1 / 2, y1 + h1)

                            if (get_point_line_distance(counpoint,
                                                        [speed_lane[0][0][0], speed_lane[0][0][1], speed_lane[1][0][0],
                                                         speed_lane[1][0][1]]) <= distance):

                                # 计算垂足
                                foot_x, foot_y = get_foot(speed_lane[0][0], speed_lane[1][0], counpoint) # 注意foot_x foot_y

                                for j in range(num):
                                    if (foot_y >= x_calculate[j] and foot_y <= x_calculate[j + 1]): #修改成y
                                        # draw rectangle & label ID
                                        cv2.rectangle(im0, (int(x1), int(y1)),
                                                      (int(x2), int(y2)), rectangleColor, 4)
                                        cv2.putText(im0, "id " + str(i), (int(x1 + w1 / 2), int(y1 - 30)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                        lane[j].append(i)



                # 清理lane中的重复元素
                for ii in range(num):
                    if lane[ii] != []:
                        lane[ii] = list(set(lane[ii]))

                # 清理stop中的重复元素
                for ii in range(num):
                    if stop[ii] != []:
                        stop[ii] = list(set(stop[ii]))

                # 车头时距计算+车道流量计算
                for i in range(num):
                    if lane[i] != []:
                        sum = 0
                        for j in lane[i]:
                            sum = sum + int(1)
                            carnumlane[i] = sum

                # # CSV写 # 写车道流量
                # # if frame_index > 0:
                # if frame_idx % fps ==0:
                #     if flag != True:
                #         f = open(csvfile, 'a', newline='')
                #         writer = csv.writer(f)
                #         writer.writerow(splicing_csvdata3(frames_to_timecode(fps, frame_idx), carnumlane))
                #         # writer.writerow(
                #         #     splicing_csvdata(frames_to_timecode(fps, frame_idx), carnumlane, fpstimeheadway, timeoccupation,
                #         #                      Saturation))
                #         f.close()
                #     else:
                #         f = open(csvfile, 'a', newline='')
                #         writer = csv.writer(f)
                #         writer.writerow(splicing_csvdata3(frames_to_timecode(fps, frame_idx), carnumlane))
                #         # writer.writerow(
                #         #     splicing_csvdata2(frames_to_timecode(fps, frame_idx), carnumlane, fpstimeheadway, timeoccupation,
                #         #                       Saturation))
                #         f.close()

                    # cv2.line(im0, speed_lane[0][0], speed_lane[1][0], (0, 255, 0), 10)  # 绿色，1个像素宽度
                    # for i in range(num + 1):
                    #     cv2.line(im0, (location[0][i][0], location[0][i][1]), (location[1][i][0], location[1][i][1]), [255, 0, 0],
                    #              1)  # 蓝色，3个像素宽度

            # 显示每一帧
            # cv2.resize(im0, (width /2, height/2))
            cv2.imshow('result', im0) # resultImage(1920, 1080)(2720, 1530)cv2.resize(im0, (1920, 1080))
            # cv2.waitKey(1)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dirpath = r'video'  # [[(4, 457), (358, 550), (596, 557), (749, 560), (596, 387)], [(22, 989), (669, 989), (947, 1001), (1300, 999), (1886, 899)]]

    filepath = 'video/DJI_0026.mp4'

    #摄像头加载
    # video = cv2.VideoCapture(0)
    #文件加载
    video = cv2.VideoCapture(filepath)
    frames_num = video.get(7)
    print(frames_num)
    cur = datetime.datetime.now()  # 获取当前系统时间
    no_use, img = video.read()
    # 检测阈值
    distance = 50
    # 跳帧调节
    jumpfps = 1
    while video.isOpened():
        process, img = video.read()

        if process:
            WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            fps = 30  # round(cv2.CAP_PROP_FPS)  # cv2.CAP_PROP_FRAME_COUNT
            # 多文件处理（单文件不用管）
            location = lm(img)  # main_utils.lanemark(img) #
            regular_location = location
            roi_vtx = np.array([[location[0][0], location[1][0], location[1][len(location[0]) - 1],
                                 location[0][len(location[0]) - 1]]])
            speed_lane = lm(img)
            regular_speedlane = speed_lane
            regular_stoplane = speed_lane
            # if filenum == 1:
            # 标记车道线 （从左往右，注意顺序）
            print('Please mark the lane!')
            # location = lm(img) # main_utils.lanemark(img) #
            # regular_location = location
            roi_vtx = np.array([[location[0][0], location[1][0], location[1][len(location[0]) - 1],
                                 location[0][len(location[0]) - 1]]])
            print('The Location has been marked successfully ')
            print('location', location)

            # 标记测流量线（1条）
            print('Please mark the speed lane!')
            # speed_lane = calculate_speedlane()
            # speed_lane = lm(img)#[[(800, 155)], [(800, 855)]]# [[(135, 161)], [(907, 155)]]
            # speed_lane = [[(152, 639)], [(1313, 620)]]
            # speed_lane = [[(176, 136)], [(530, 134)]]#cars.mp4
            # speed_lane =
            # regular_speedlane = speed_lane
            print('The speed lane has been marked successfully ')
            print('speed lane', speed_lane)

            # 标记停止线（1条）
            print('Please mark the stop lane!')
            stop_lane = speed_lane
            # stop_lane = [[(152, 639)], [(1313, 620)]]
            # stop_lane = [[(227, 46)], [(465, 49)]]#cars.mp4
            regular_stoplane = stop_lane
            print('The stop lane has been marked successfully ')
            print('stop lane', stop_lane)

            Pass_traffic_input = [1]

            # else:
            #     # 多帧统一车道线、流量线、停止线标记
            #     location = regular_location
            #     speed_lane = regular_speedlane
            #     stop_lane = regular_stoplane

            # 计算车道流量线（1条）
            # y_numcount = ((speed_lane[0][0][1] + speed_lane[1][0][1]) // 2)
            #
            # y_numcountstop = ((stop_lane[0][0][1] + stop_lane[1][0][1]) // 2)
            y_numcount = ((speed_lane[0][0][0] + speed_lane[1][0][0]) // 2)

            y_numcountstop = ((stop_lane[0][0][0] + stop_lane[1][0][0]) // 2)

            lanecross = lane_cross(speed_lane, location)

            k = np.array(location[0]).shape[0] - 1  # 车道数量（非车道线数量）

            # 创建csv文件
            csvfile = 'DJI_0026' + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S') + '   .csv' # file[0:len(file) - 4]
            f = open(csvfile, 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(["时间", "车道1车流量", "车道2车流量", "车道3车流量", "车道4车流量", "车道5车流量"])

            f.close()

            print('Start processing!')
            # 计算交点
            # x_calculate = calculate_linemark(location, k, y_numcount)
            x_calculate = []
            for i in range(k + 1):
                x_calculate.append(lanecross[i][0][1])

            x_calculate_stop = calculate_linemark(location, k, y_numcountstop)
            x_calculate_stop = x_calculate
            trackMultipleObjects(location, k)
            print('Processing Successfully')

        else:
            break
