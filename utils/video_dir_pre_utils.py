# -*- coding: utf-8 -*-
import csv

import cv2
import pandas as pd
import numpy as np
import os
from moviepy.editor import *
from utils.main_utils import (lanemark, calculate_speedlane, lane_cross_ew,lane_cross_ns, roi_mask, get_foot, frames_to_timecode,
                        get_point_line_distance, splicing_csvdata2)


def get_video_times(video_path):
    """
    pip install moviepy
    获取指定的视频时长，单位是秒

    """
    from moviepy.editor import VideoFileClip
    video_clip = VideoFileClip(video_path)
    durantion = video_clip.duration
    return durantion

def time_convert(seconds):
    """
        将秒换成合适的时间，如果超过一分钟就换算成"分钟:秒",如果是小时，就换算成"小时:分钟:秒"单位换算
    """
    # print(f'时间换算{seconds}')
    seconds = int(seconds)
    M,H = 60,3600
    if seconds < M:
        return f'00:00:0{seconds}' if seconds < 10 else f'00:00:{str(seconds)}'
    elif seconds < H:
        _M = int(seconds/M)
        _S = int(seconds%M)
        return f'00:{f"0{_M}" if _M < 10 else str(_M)}:{f"0{_S}" if _S < 10 else str(_S)}'
    else:
        _H = int(seconds/H)
        _M = int(seconds%H/M)
        _S = int(seconds%H%M)
        return f'{f"0{_H}" if _H < 10 else str(_H)}:{f"0{_M}" if _M < 10 else str(_M)}:{f"0{_S}" if _S < 10 else str(_S)}'

#读取中文路径
def cv_imread(file_paht):
    cv_img=cv2.imdecode(np.fromfile(file_paht,dtype=np.uint8),-1)
    return cv_img

#保存中文路径
def cv_imwrite(savePath,tem):
    cv2.imencode('.jpg',tem)[1].tofile(savePath)  # 保存图片



def duration_load(filePath):
    os.makedirs(os.path.join('..'), exist_ok=True)
    data_file = os.path.join(filePath + '/data.csv')
    with open(data_file, 'w') as f:
        f.write('视频名称,开始时间o,结束时间o,是否裁剪,开始时间,结束时间\n')  # 列名 roi_vtx,speedlane
    files = os.listdir(filePath)
    # for root, dirs, files in os.walk(filePath):
    #     # 按文件名排序
    files.sort()
    for file in files:
        # 如果后缀名为 .mp4
        if os.path.splitext(file)[1] == '.MP4' or os.path.splitext(file)[1] == '.mp4':
            file_name = os.path.splitext(file)[0]
            duration_s = get_video_times(os.path.join(filePath, file))
            duration = time_convert(duration_s)
            with open(data_file, 'a') as f:
                f.write(file_name+ ',00:00:00,' + duration +'\n')  # 每行表示一个数据样本

def video_cut(filePath):
    data_file = os.path.join(filePath + '/data.csv')
    data = pd.read_csv(data_file, encoding='gbk')  # encoding="utf-8"会报错
    # 指定列data['是否裁剪']
    # 视频名称,开始时间_0,结束时间_0,是否裁剪,开始时间,结束时间
    new_data_file = os.path.join(filePath + '/new_data.csv')
    with open(new_data_file, 'w') as f:
        f.write('视频名称,视频时长,路口数\n')  # 列名 roi_vtx,speedlane

    for root, dirs, files in os.walk(filePath):
        files.sort()
        for file in files:
            if os.path.splitext(file)[1] == '.MP4':
                file_name = os.path.splitext(file)[0]
                for i in range(len(data)):
                    if str(data['视频名称'][i]) == file_name:
                        if str(data['是否裁剪'][i]) == "是":
                            # print(data['视频名称'][i])
                            file_name = data['视频名称'][i]
                            old_path = os.path.join(filePath, file_name + '.mp4')
                            new_path = os.path.join(filePath, file_name + '_cut.mp4')
                            print(file_name + '.mp4' + '剪切为：' + file_name + '_cut.mp4')
                            # 载入视频并剪切
                            video = VideoFileClip(old_path)
                            video = video.subclip(data['开始时间'][i], data['结束时间'][i])
                            video.to_videofile(new_path)
                            # 计算生成视频时长
                            duration_s = get_video_times(new_path)
                            duration = time_convert(duration_s)
                            file_name = data['视频名称'][i] + '_cut'
                            with open(new_data_file, 'a') as f:
                                f.write(file_name + ',' + duration +'\n')  # 每行表示一个数据样本
                        else:
                            duration = data['结束时间o'][i]
                            with open(new_data_file, 'a') as f:
                                f.write(file_name + ',' + duration + '\n')  # 每行表示一个数据样本
                        print('写入' + file_name + '.MP4')
                        vid2pic(filePath + '/', file_name)




def vid2pic(video_path, video_name):
    # 输出图片到pic文件夹下
    outPutDirName =  video_path + 'pic/'
    if not os.path.exists(outPutDirName):   # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)

    cap = cv2.VideoCapture(video_path + video_name + '.mp4')
    frame_index = 0
    ref, frame = cap.read()
    if not ref:
        raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
    while (cap.isOpened):
        ref, frame = cap.read()
        if not ref:
            break
        if frame_index == 2:
            # cv2.imshow('image', frame)
            # cv2.waitKey(0)
            savePath = outPutDirName + str(video_name) + '.jpg'
            cv_imwrite(savePath, frame)
            # cv2.imwrite(outPutDirName + str(video_name) + '.jpg', frame)
            print('保存图片为' + outPutDirName + str(video_name) + '.jpg')
            # print('图片提取结束')
            break
        frame_index += 1

    cap.release()


def get_points(filePath):
    pic_file = os.path.join(filePath + '/pic')
    data_file = os.path.join(filePath + '/new_data.csv')
    output = os.path.join(filePath + '/data_load.csv')

    with open(data_file, 'r') as file:
        # 读取CSV文件内容
        reader = csv.reader(file)
        # 获取CSV文件的表头
        headers = next(reader)
        # 表头元素添加
        for n in range(2):
            location_name = f'location{n}'
            roi_name = f'roi{n}'
            speedline_name = f'speedline{n}'
            lanecross_name = f'lanecross{n}'
            lanenum_name = f'lanenum{n}'
            add_col(headers, location_name, roi_name, speedline_name, lanecross_name, lanenum_name)

        with open(output, 'w', newline='') as outfile:
            # 写入表头
            writer = csv.writer(outfile)
            writer.writerow(headers)

    data = pd.read_csv(data_file, encoding='gbk')  # encoding="utf-8"会报错
    # '视频名称,视频时长,路口数,（roi_vtx,speedlane）
    num_pic = len(data)
    LOCATION = list(range(num_pic))
    ROI = list(range(num_pic))
    SPEED = list(range(num_pic))
    LANE_CROSS = list(range(num_pic))
    LANE_NUM    = list(range(num_pic))
    for root, dirs, files in os.walk(pic_file):
        files.sort()
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                file_name = os.path.splitext(file)[0]
                for i in range(len(data)):
                    if str(data['视频名称'][i]) == file_name:
                        pic_path = os.path.join(pic_file, file_name + '.jpg')
                        # LOCATION[i], ROI[i], SPEED[i], LANE_CROSS[i], LANE_NUM[i] = draw_all_lines(pic_path, i)
                        draw_all_lines(pic_path, i)

    # return LOCATION, ROI, SPEED, LANE_CROSS, LANE_NUM


def draw_all_lines(pic_path, row_num):
    input = os.path.join(filePath + '/new_data.csv')
    with open(input, 'r') as file:
        # 读取CSV文件内容
        reader = csv.reader(file)
        rows = list(reader)
        rows = rows[1:]

    row = rows[row_num]
    row.append('')
    for k in range(2):
        row_update = draw_lines(pic_path, k, row)
        rows[row_num] =row_update


    output = os.path.join(filePath + '/data_load.csv')
    with open(output, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        # 填写数据
        writer.writerow(rows[row_num])
    print('-------标注完一张图片--------')

    # return location, roi, speed, lane_cross,lane_num



def draw_lines(pic_path, i, row):
    # img = cv2.imread(pic_path)
    img= cv_imread(pic_path)
    # 划线，东西方向按从下到上的顺序从左往右画，南北方向按从左到右的顺序从上往下画
    print('Please mark the lane!')
    location = lanemark(img)  # .lanemark
    # 标记路口测流量线（1条）
    print('Please mark the speed lane!')
    speed_line = calculate_speedlane(img)

    # 计算交点，东西方向按从下到上从小到大的顺序，南北方向按从左到右从小到大的顺序
    if location != ' ':
        if i%4 ==0:
            lanecross = lane_cross_ew(speed_line, location)
            roi_vtx = np.array(
                [[location[0][0], lanecross[len(lanecross) - 1][0], lanecross[0][0], location[0][len(location[0]) - 1]]])
            k = np.array(location[0]).shape[0] - 1  # 车道数量（非车道线数量）
        if i%4 ==1:
            lanecross = lane_cross_ew(speed_line, location)
            roi_vtx = np.array(
                [[location[1][0],  lanecross[len(lanecross) - 1][0],lanecross[0][0], location[1][len(location[0]) - 1]]])
            k = np.array(location[0]).shape[0] - 1  # 车道数量（非车道线数量）
        if roi_vtx != ' ':
            roi_vtx = roi_vtx.flatten()  # 展平数组
    else:
        roi_vtx = ' '
        speed_line = ' '
        lanecross = ' '
        k = ' '

    add_value(row, location,  roi_vtx, speed_line, lanecross, k)

    # return location, roi_vtx, speed_line, lanecross, k
    return row



def add_points(filePath):
    LOCATION, ROI, SPEED, LANE_CROSS, LANE_NUM = get_points(filePath)
    input = os.path.join(filePath + '/new_data.csv')
    output = os.path.join(filePath + '/data_load.csv')
    with open(input, 'r') as file:
        # 读取CSV文件内容
        reader = csv.reader(file)
        # 获取CSV文件的表头
        headers = next(reader)
        # 表头元素添加
        for n in range(2):
            location_name = f'location{n}'
            roi_name = f'roi{n}'
            speedline_name = f'speedline{n}'
            lanecross_name = f'lanecross{n}'
            lanenum_name = f'lanenum{n}'
            add_col(headers, location_name, roi_name, speedline_name, lanecross_name, lanenum_name)
        # 创建一个新的CSV文件
        with open(output, 'w', newline='') as outfile:
            # 写入表头
            writer = csv.writer(outfile)
            writer.writerow(headers)
            # 遍历每一行数据
            row_num = 0
            for row in reader:
                row.append('')
                location_pic = LOCATION[row_num]
                roi_pic = ROI[row_num]
                speedline_pic = SPEED[row_num]
                lanecross_pic = LANE_CROSS[row_num]
                lanenum_pic = LANE_NUM[row_num]
                # 在每一行中添加新的列的值
                for n in range(2):
                    location = location_pic[n]
                    roi = roi_pic[n]
                    if roi != ' ':
                        roi = roi.flatten()  # 展平数组
                    speedline = speedline_pic[n]
                    lanecross = lanecross_pic[n]
                    lanenum = lanenum_pic[n]
                    add_value(row, location, roi, speedline, lanecross, lanenum)

                row_num = row_num + 1
                # 写入新的一行数据
                writer.writerow(row)
                print('写入一张图片路口划线坐标')


def add_col(headers, name1, name2, name3, name4, name5):
    headers.append(name1)
    headers.append(name2)
    headers.append(name3)
    headers.append(name4)
    headers.append(name5)

def add_value(row, value1, value2, value3, value4, value5):
    row.append(value1)
    row.append(value2)
    row.append(value3)
    row.append(value4)
    row.append(value5)


def revise_data_load(filePath):
    input = os.path.join(filePath + '/data_load.csv')
    output = os.path.join(filePath + '/data_load_n.csv')
    data = pd.read_csv(input, encoding='gbk')  # encoding="utf-8"会报错
    with open(input, 'r') as file:
        # 读取CSV文件内容
        reader = csv.reader(file)
        # 获取CSV文件的表头
        headers = next(reader)
        # 表头元素添加
        for n in range(2):
            roi_name = f'roi{n}'
            headers.append(roi_name)
        # 创建一个新的CSV文件
        with open(output, 'w', newline='') as outfile:
            # 写入表头
            writer = csv.writer(outfile)
            writer.writerow(headers)
            # 遍历每一行数据
            row_num = 0
            for row in reader:
                for n in range(2):
                    new_r = new(row_num, n)
                    row.append(new_r)
                row_num = row_num + 1
                writer.writerow(row)



def new(row_num, n):
    input = os.path.join(filePath + '/data_load.csv')
    data = pd.read_csv(input, encoding='gbk')  # encoding="utf-8"会报错
    if data['location'+ str(n)][row_num] != ' ':
        location = eval(data['location' + str(n)][row_num])
        points = location[0]
        points.extend(location[1])
        max_x = 0
        max_y = 0
        min_x = 10000
        min_y = 10000
        for point in points:
            x = point[0]
            y = point[1]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            roi = np.array([max_x, max_y, max_x, min_y, min_x, min_y, min_x, max_y])    # 保存下来的单元格内容中存在逗号，可用ctrl+H批量替换为空
    else:
        roi = np.array([[]])

    return roi



if __name__ == '__main__':
    # 填入需要处理的视频文件夹路径，注意该python路径为utils文件夹
    filePath = '../source/20240913高架'

    # step1:运行duration_load(filePath)，
    #    生成data.csv，在需要裁剪的视频栏填入对应信息,不裁剪不填，裁剪填后三栏
    # duration_load(filePath)

    # step2:生成new_data.csv，同时保存new_data中每个视频的指定帧为图片存放在pic文件夹中,new_data中路口数不用填写，按4个画
    data_file = os.path.join(filePath + '/data.csv')
    video_cut(filePath)

    # step3:根据new_data.csv对每张图片的每个路口进行划线（车道线和速度线）,并将获得的点坐标添加到后面加4组（roi，speed）列,生成data_load.csv
    get_points(filePath)

    # step4:运行revise_data_load，修改roi值，打开生成的data_load_n.csv,删除原本的roi列  (需改进,直接保存更新的roi)
    revise_data_load(filePath)



















    # 单独调试vidpic函数
    # vid2pic('../source/20231106/', 'DJI_0052_cut')
    # 单独调试draw_lines函数
    # draw_lines('../source/20231106/pic/DJI_0052_cut.jpg', 1)
    # 单独调试draw_all_lines函数
    # draw_all_lines('../source/20231106/pic/DJI_0052_cut.jpg')


    # roi_mask的操作
    # a = np.array([[[6,701], [748,670], [739,463], [29,490]]])
    # b = np.array([[]])
    # c = np.array([[[1855,435], [1308,218], [1312,462], [1769,200]]])
    # d = np.array([[[1855,435], [1308,218], [1312,462], [1769,100]]])
    # pic_path = '../source/20231106/pic/DJI_0052_cut.jpg'
    # img = cv2.imread(pic_path)
    # v = []
    # if len(a[0])>0:
    #     v.append(a)
    # if len(b[0])>0:
    #     v.append(b)
    # if len(c[0])>0:
    #     v.append(c)
    # if len(d[0])>0:
    #     v.append(d)
    # image = roi_mask(img, v)