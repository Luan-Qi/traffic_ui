# -*- coding: utf-8 -*-
import csv
import os
import datetime

from PIL import Image
from unet.highway_selectbeforefit import fit_lanes
from unet.predict import predict_road_pixel

from utils.draw_stop_lane import get_roi
from utils.save_xml import write_roads
from utils.hand_draw_utils import *

class Segmentation(object):
    def __init__(self, filepath=None, preprocessed=False):
        if preprocessed:
            self.dir = None
            self.out = None
            self.size = None
            self.roi_zone = None
            self.scale = None
            self.size = None
            self.road_rules = None
            self.video_path = filepath
            return
        # 文件加载
        self.video_path = filepath
        self.video = cv2.VideoCapture(filepath)

        # 保存路径根地址
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
                # image_l = np.array(image_l)
                # image_x = np.array(image_x)
                image_lx = np.array(image_lx)
                self.dir,self.out, self.size = fit_lanes(image_s, frame, image_lx)
                self.roi_zone, self.scale = get_roi(self.dir)
                self.size = [self.size[0] * self.scale, self.size[1] * self.scale]
                # frame0 = roi_mask(frame, self.roi_zone)
                # frame = draw_road_lines(frame, self.dir,'')
                # cv2.imwrite('output_image.jpg', frame)

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
        self.root = './output/'
        file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        cur = datetime.datetime.now()  # 获取当前系统时间

        if self.vid_save:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vid_save_path = 'output/vid/' + file_name + '_output.mp4'
            self.vid_writer = cv2.VideoWriter('output/vid/' + file_name + '_output.mp4', fourcc, self.fps, (self.width, self.height))

        # csv
        if self.car_track_save:
            self.csvfile = self.root + file_name + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S') + '.csv'
            f = open(self.csvfile, 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(["frame", "vehicle_id", "vehicle_type", "dir_id", "lane_id", "position_x(m)", "position_y(m)",
                             "width(m)", "height(m)", 'xVelocity(km/h)', 'yVelocity(km/h)', '车头时距(s)', '车道占有率（%）'])
            f.close()
        if self.car_num_save:
            self.csvfile0 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' west.csv'
            self.csvfile2 = self.root + file_name + '_' + datetime.datetime.strftime(cur,
                                                                                     '%Y-%m-%d %H.%M.%S') + ' east.csv'
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
                with open(output, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(write_list)
                f.close()

    def exit_process(self):
        # self.quit = True
        self.cap.release()
        if self.vid_save:
            self.vid_writer.release()

    def save_xml(self):
        file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.xmlfile = './output/' + file_name + '_lane.xml'
        write_roads(self.dir, self.scale,self.xmlfile)



if __name__ == '__main__':
    # 填入需要处理的视频文件夹路径，注意该python路径为utils文件夹
    filePath = r'D:\yjh\code\pytorch\highway_track_id_v_a\source\wq\DJI_0041_c.MP4'

    # handdraw = Hand_Draw(filePath)
    segmentation = Segmentation(filePath)
