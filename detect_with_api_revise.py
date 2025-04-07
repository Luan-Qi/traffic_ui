import time

import torch
from numpy import random
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, MyLoadImages, LoadStreams
from yolov7.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, xyxy2xywh
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
import cv2


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
class simulation_opt:
    def __init__(self, weights='models/yolov7.pt',
                 img_size=640, conf_thres=0.25,
                 iou_thres=0.45, device='', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok


class detectapi:
    def __init__(self, weights, img_size=640):
        self.opt = simulation_opt(weights=weights, img_size=img_size)
        weights, imgsz = self.opt.weights, self.opt.img_size
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle',
                              'bus', 'e-bike']
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def detect(self, source):  # 使用时，调用这个函数
        if type(source) != list:
            raise TypeError('source must be a list which contain  pictures read by cv2')
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)  # imgsz
        # 原来是通过路径加载数据集的，现在source里面就是加载好的图片，所以数据集对象的实现要
        # 重写。修改代码后附。在utils.dataset.py上修改。
        # dataset =LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        # t0 = time.time()
        # result = []
        '''
        for path, img, im0s, vid_cap in dataset:''' # img, im0s
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            result_conf = []
            outputs = []  # result_box = []
            bbox_xywh = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() #[0]
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xyxy_box = [int(_.item()) for _ in xyxy]
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c = (bbox_left + bbox_w / 2)
                    y_c = (bbox_top + bbox_h / 2)
                    # w = bbox_w
                    # h = bbox_h
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    line1 = [int(cls.item())]  # label
                    line2 = [conf.item()]
                    # result_box.append((xyxy_box))
                    result_txt.append(line1)
                    result_conf.append(line2)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(result_conf)
                classes = torch.Tensor(result_txt)
                outputs = self.deepsort.update(xywhs, confss, im0, classes)
                return outputs, self.names
            else:
                return None, None
        # outputs = self.deepsort.update(xywhs, confss, im0, classes)
            # result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        # return im0, xywhs, confss, classes, self.names  # result_txt result_conf result_txt im0,  outputs


    def draw_boxes(img, bbox, classes2, cls_names, identities=None, offset=(0, 0)):
        # this_ids_info = last_ids_info
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = detectapi.compute_color_for_labels(int(classes2[i] * 100))
            label = '%d %s' % (id, cls_names[i])
            if cls_names[i] == 'car': # id in this_ids_info and this_ids_info[id]['speed'] != 0 and
                # speed = round(this_ids_info[id]['speed'], 1)
                label = '%d %s' % (id, cls_names[i])  # %s km/h , speed
            # label = '%d %s' % (id, '')
            # label +='%'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)


        return img

    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def detect_stream(self, source):
        # dataset =LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        t0 = time.time()
        for img, im0s in dataset:
        # for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS_stream')
            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响 改[0]
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            result_conf = []
            outputs = []  # result_box = []
            bbox_xywh = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() #[0]
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xyxy_box = [int(_.item()) for _ in xyxy]
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c = (bbox_left + bbox_w / 2)
                    y_c = (bbox_top + bbox_h / 2)
                    # w = bbox_w
                    # h = bbox_h
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    line1 = [int(cls.item())]  # label
                    line2 = [conf.item()]
                    # result_box.append((xyxy_box))
                    result_txt.append(line1)
                    result_conf.append(line2)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(result_conf)
                classes = torch.Tensor(result_txt)
                outputs = self.deepsort.update(xywhs, confss, im0, classes)
                t5 = time_synchronized()
                if outputs is not None:   # 看运行程序的情况添加
                    for idx, _ in enumerate(outputs):
                        bbox_xyxy = outputs[:, :4]  # [x1,y1,x2,y2]
                        identities = outputs[:, -2]  # [i]
                        classes2 = outputs[:, -1]  # [cls]
                        # detectapi.draw_boxes(im0, bbox_xyxy, classes2,
                        #                                      [self.classes_names[i] for i in classes2],
                        #                                      identities)  # names
                # cv2.imshow("Detection", im0)
                print(f'Done. ({(1E3 * (time_synchronized() - t5)):.1f}ms)')  #  draw_boxes
                return im0, outputs, self.names
            else:
                return im0, None, None

    def detect_video(self, source):
        # dataset =LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        t0 = time.time()
        for img, im0s in dataset:
        # for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS_video')
            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响 改[0]
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            result_conf = []
            outputs = []  # result_box = []
            bbox_xywh = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() #[0]
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xyxy_box = [int(_.item()) for _ in xyxy]
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c = (bbox_left + bbox_w / 2)
                    y_c = (bbox_top + bbox_h / 2)
                    # w = bbox_w
                    # h = bbox_h
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    line1 = [int(cls.item())]  # label
                    line2 = [conf.item()]
                    # result_box.append((xyxy_box))
                    result_txt.append(line1)
                    result_conf.append(line2)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(result_conf)
                classes = torch.Tensor(result_txt)
                outputs = self.deepsort.update(xywhs, confss, im0, classes)
                # t5 = time_synchronized()
                # if outputs is not None:
                #     for idx, _ in enumerate(outputs):
                #         bbox_xyxy = outputs[:, :4]  # [x1,y1,x2,y2]
                #         identities = outputs[:, -2]  # [i]
                #         classes2 = outputs[:, -1]  # [cls]
                #         detectapi.draw_boxes(im0, bbox_xyxy, classes2,
                #                                              [self.classes_names[i] for i in classes2],
                #                                              identities)  # names

                # # cv2.imshow("Detection", im0)
                # print(f'Done. ({(1E3 * (time_synchronized() - t5)):.1f}ms) Inference')
                return im0, outputs
            else:
                return None, None