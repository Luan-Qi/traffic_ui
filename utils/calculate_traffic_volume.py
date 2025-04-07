import csv
import os
import datetime
import cv2
import numpy as np
import detect_with_api
from visdrone_lane_volume import detect
from utils.main_utils import lanemark, calculate_speedlane, lane_cross, roi_mask, get_foot, get_point_line_distance, \
    splicing_csvdata2, frames_to_timecode



def detect_lanes(image, y_calculate=None):
    # 对图像进行车道检测的逻辑
    # 返回每个车道的位置信息或车道数量
    print('Please mark the lane!')
    location = lanemark(image)
    roi_vtx = np.array([[location[0][0], location[1][0], location[1][len(location[0]) - 1],
                         location[0][len(location[0]) - 1]]])
    k = np.array(location[0]).shape[0] - 1
    print('Please mark the speed lane!')
    speed_lane = calculate_speedlane(image)
    lanecross = lane_cross(speed_lane, location)
    for i in range(k + 1):
        y_calculate.append(lanecross[i][0][1])

    return location, roi_vtx, k, speed_lane, y_calculate


def calculate_traffic(image, num_lanes, y_calculate, speed_lane):
    # 对每个车道进行车流量计算的逻辑
    # 根据车道位置和图像中的车辆数等进行计算
    # 返回每个车道的车流量
    outputs, list_name = detection.detect([image])
    lane = [[] for _ in range(num_lanes)]
    carnumlane = [0 for _ in range(num_lanes)]
    if outputs is not None:
        for idx, datainoutput in enumerate(outputs):
            [x1, y1, x2, y2, i, cls] = datainoutput

            w1 = x2 - x1
            h1 = y2 - y1

            counpoint = (x1 + w1 / 2, y1 + h1 / 2)
            if (get_point_line_distance(counpoint,
                                        [speed_lane[0][0][0], speed_lane[0][0][1], speed_lane[1][0][0],
                                         speed_lane[1][0][1]]) <= distance):
                # 计算垂足
                foot_x, foot_y = get_foot(speed_lane[0][0], speed_lane[1][0], counpoint)

                for j in range(num_lanes):
                    if (foot_y >= y_calculate[j] and foot_y <= y_calculate[j + 1]):
                        # draw rectangle & label ID
                        # cv2.rectangle(resultImage, (int(x1), int(y1)),
                        #               (int(x2), int(y2)), rectangleColor, 4)
                        # cv2.putText(resultImage, "id " + str(i), (int(x1), int(y1 - 1)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        bbox_xyxy = outputs[:, :4]  # [x1,y1,x2,y2]
                        identities = outputs[:, -2]  # [i]
                        classes2 = outputs[:, -1]  # [cls]
                        detect_with_api.detectapi.draw_boxes(image, bbox_xyxy, classes2,
                                                             [classes_names[i] for i in classes2], identities)  # names
                        lane[j].append(i)
    for ii in range(num_lanes):
        if lane[ii] != []:
            lane[ii] = list(set(lane[ii]))
    for i in range(num_lanes):
        if lane[i] != []:
            sum = 0
            for j in lane[i]:
                sum = sum + int(1)
                carnumlane[i] = sum
    return carnumlane



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    source = 'source/20231106/DJI_0054_cut.mp4'  # 'video/1280.mp4'   1280.mp4 video/ uav0011.mp4  DJI_0012  source/my111.avi DJI_test1.mp4
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    vid_writer = cv2.VideoWriter("cars33_output_test_new.MP4", fourcc, 30, (1920, 1080))
    video = cv2.VideoCapture(source)
    frames_num = video.get(7)
    print('frames_num:', frames_num)
    frame_index = 0
    classes_names = ['pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'e-bike']
    detection = detect_with_api.detectapi(weights='weights/best407_att.pt')
    distance = 50
    cur = datetime.datetime.now()
    csvfile = source[0:len(source) - 4] + '_' + datetime.datetime.strftime(cur, '%Y-%m-%d %H.%M.%S') + '   .csv'
    while video.isOpened():
        flag, img = video.read()
        if flag:
            frame_index +=1
            # 检测进口道车道
            lane_positions, roi_vtx, num_lanes, speed_lane, y_calculate = detect_lanes(img)
            image = roi_mask(img, [roi_vtx])

            # 计算每个车道的车流量
            traffic_flow = calculate_traffic(image, num_lanes, lane_positions, speed_lane)

            # 输出每个车道的车流量
            if frame_index % frames_num == 0:
                if flag != True:
                    f = open(csvfile, 'a', newline='')
                    writer = csv.writer(f)
                    writer.writerow(
                        splicing_csvdata2(frames_to_timecode(frames_num, frame_index+2), traffic_flow))
                    print('carnumlane')
                    f.close()
            cv2.line(img, speed_lane[0][0], speed_lane[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
            for i in range(num_lanes + 1):
                cv2.line(img, (lane_positions[0][i][0], lane_positions[0][i][1]), (lane_positions[1][i][0], lane_positions[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            vid_writer.write(img)
            cv2.imshow("multi_pic", img)
            # for lane_id, flow in enumerate(traffic_flow):
            #     print(f"Lane {lane_id + 1}: {flow} vehicles")
        elif type(img) == type(None) or not flag:
            # print(len(image), flag)
            print('video error')
            if isinstance(vid_writer, cv2.VideoWriter) and frame_index > 30:
                video.release()
                break
        elif (cv2.waitKey(1) & 0xFF) == ord('q'):
            break