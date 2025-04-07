# -*- coding: utf-8 -*-
import os
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, \
    QComboBox, QTextEdit, QCheckBox
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QMouseEvent
import cv2
import numpy as np
import time
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer
# from utils.main_utils import lanemark as lm, calculate_speedlane, roi_mask, roi_mask2
from utils.cross_process import Hand_Draw_Cross, Segmentation_Cross
from utils.highway_process import Hand_Draw, Segmentation
from utils.main_utils import lanemark as lm, calculate_speedlane, roi_mask
import detect_with_api_revise
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
# 初始化摄像头和Yolo模型
from utils.main_utils import counter_vehicles, splicing_csvdata, frames_to_timecode, estimateSpeed, estimate_a, draw_counter, draw_boxes, splicing_csvdata5
from utils.draw_stop_lane import draw_road_lines, get_roi, get_position_id,draw_all_lines
# from unet.cross import fit_lanes,p2l_dis
from utils.save_xml import write_crosses, write_roads

from qt_material import apply_stylesheet


class tQTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(30)
        self._init_ui()
        self._init_style()
        self._drag_pos = QPoint()
        self.update()

    def _init_ui(self):
        # 创建控件
        self.title_label = QLabel("交通流信息提取")
        self.min_btn = QPushButton('-')
        self.max_btn = QPushButton('□')
        self.close_btn = QPushButton('×')

        # 设置按钮属性
        buttons = [self.min_btn, self.max_btn, self.close_btn]
        for btn in buttons:
            btn.setFixedSize(QSize(30, 30))
            btn.setFocusPolicy(Qt.NoFocus)

        # 布局设置
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0)
        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)

        # 信号连接
        self.min_btn.clicked.connect(self.window().showMinimized)
        self.max_btn.clicked.connect(self.toggle_maximize)
        self.close_btn.clicked.connect(self.window().close)

    def _init_style(self):
        self.setStyleSheet("""
            /* 使用ID选择器提高优先级 */
            QWidget#titleBar QPushButton {
                color: white !important;
                font-family: "Segoe UI Symbol";
                font-size: 16px;
            }
            QWidget#titleBar QPushButton#min_btn:hover {
                background: #9CFF65 !important;
            }
            QWidget#titleBar QPushButton#max_btn:hover {
                background: #FEFF5C !important;
            }
            QWidget#titleBar QPushButton#close_btn:hover {
                background: #FF9AB6 !important;
            }
        """)
        self.min_btn.setObjectName("min_btn")
        self.max_btn.setObjectName("max_btn")
        self.close_btn.setObjectName("close_btn")

    def toggle_maximize(self):
        window = self.window()
        if window.isMaximized():
            window.showNormal()
            self.max_btn.setText('□')
        else:
            window.showMaximized()
            self.max_btn.setText('❐')

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            window = self.window()
            if window.isMaximized():
                # 计算点击位置比例用于平滑过渡
                screen_rect = QApplication.primaryScreen().availableGeometry()
                mouse_x = event.globalPosition().toPoint().x()
                width = window.width()
                normalized_x = mouse_x * (width / screen_rect.width())

                window.showNormal()
                new_x = mouse_x - normalized_x
                window.move(new_x, 0)
                self._drag_pos = QPoint(normalized_x, event.position().y())

            # 移动窗口
            delta = event.globalPosition().toPoint() - self._drag_pos
            window.move(window.x() + delta.x(), window.y() + delta.y())
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.toggle_maximize()
        event.accept()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 设置界面
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.title_bar = None
        self.btn_open = None
        self.play_original_button = None
        self.pauseBtn = None
        self.stopBtn = None
        self.processing_method_combobox = None
        self.processing_case = None
        self.save_case1 = None
        self.save_case2 = None
        self.save_case3 = None
        self.btn_open_xml = None
        self.execute_button = None
        self.save_xml_button = None
        self.execute_button1 = None
        self.stop_process_button = None
        self.text_edit = None
        self.image_label = None

        self.video_path = None
        self.cap = None
        self.paused = False
        self.stopped = False

        self.timer = None
        self.routes = None
        self.video_capture = None

        self.setupUI()

        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

    def setupUI(self):
        #self.resize(800, 600)
        self.setWindowTitle('交通流信息提取')
        # 设置窗口的最小和最大大小
        self.setMinimumSize(800, 600)  # 最小宽度400，最小高度300
        self.setMaximumSize(1920, 1080)

        layout = QVBoxLayout()

        # 添加标题栏
        self.title_bar = tQTitleBar(self)
        layout.addWidget(self.title_bar)

        btnLayout0 = QHBoxLayout()
        self.btn_open = QPushButton('选择视频')
        self.btn_open.clicked.connect(self.open_video)
        btnLayout0.addWidget(self.btn_open)

        self.play_original_button = QPushButton("播放原视频")
        self.play_original_button.clicked.connect(self.play_original_video)
        self.play_original_button.setEnabled(False)
        btnLayout0.addWidget(self.play_original_button)

        btnLayout = QHBoxLayout()
        self.pauseBtn = QPushButton('⏸暂停')
        self.pauseBtn.clicked.connect(self.pause)
        self.pauseBtn.setEnabled(False)
        btnLayout.addWidget(self.pauseBtn)

        self.stopBtn = QPushButton('🛑结束')
        self.stopBtn.clicked.connect(self.stop)
        self.stopBtn.setEnabled(False)
        btnLayout.addWidget(self.stopBtn)
        btnLayout0.addLayout(btnLayout)
        layout.addLayout(btnLayout0)

        btnLayout1 = QHBoxLayout()
        self.processing_method_combobox = QComboBox()
        self.processing_method_combobox.addItems(["利用手划线车道线作为车道位置", "利用语义分割模型识别车道线"])  # 根据实际处理方式添加选项
        btnLayout1.addWidget(self.processing_method_combobox)

        self.processing_case = QComboBox()
        self.processing_case.addItems(["高速/高架", "路口"])  # 根据实际处理方式添加选项
        btnLayout1.addWidget(self.processing_case)

        checkLayout = QHBoxLayout()
        self.save_case1 = QCheckBox("视频")
        checkLayout.addWidget(self.save_case1)
        self.save_case2 = QCheckBox("轨迹")
        checkLayout.addWidget(self.save_case2)
        self.save_case3 = QCheckBox("分车道车流量")
        checkLayout.addWidget(self.save_case3)
        btnLayout1.addLayout(checkLayout)
        layout.addLayout(btnLayout1)


        btnLayout2 = QHBoxLayout()
        self.btn_open_xml = QPushButton('选择道路结构')
        self.btn_open_xml.clicked.connect(self.load_xml)
        btnLayout2.addWidget(self.btn_open_xml)

        btnLayout3 = QHBoxLayout()
        self.execute_button = QPushButton("获取车道线信息")
        self.execute_button.clicked.connect(self.execute_processing)
        self.execute_button.setEnabled(False)  # 初始未选视频时不可用
        btnLayout3.addWidget(self.execute_button)

        self.save_xml_button = QPushButton("保存路网结构")
        self.save_xml_button.clicked.connect(self.save_xml)
        self.save_xml_button.setEnabled(False)  # 初始未生成路网结构时不可用
        btnLayout3.addWidget(self.save_xml_button)
        btnLayout2.addLayout(btnLayout3)

        btnLayout4 = QHBoxLayout()
        self.execute_button1 = QPushButton("获取交通流参数")
        self.execute_button1.clicked.connect(self.get_out_csv)
        self.execute_button1.setEnabled(False)  # 初始未生成路网结构时不可用
        btnLayout4.addWidget(self.execute_button1)

        self.stop_process_button = QPushButton("🛑结束！")
        self.stop_process_button.clicked.connect(self.stop_process)
        self.stop_process_button.setEnabled(False)  # 初始未生成路网结构时不可用
        btnLayout4.addWidget(self.stop_process_button)
        btnLayout2.addLayout(btnLayout4)
        layout.addLayout(btnLayout2)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.resize(400, 100)
        layout.addWidget(self.text_edit)

        layoutVid = QHBoxLayout()
        self.image_label = QLabel(self)
        layoutVid.addWidget(self.image_label)
        layoutVid.setAlignment(self.image_label, Qt.AlignCenter)
        layout.addLayout(layoutVid)


        self.setLayout(layout)

        # self.timer = QTimer(self)  # 定义定时器，用于控制显示视频的帧率
        # self.timer.timeout.connect(self.update_frame)  # 定时到了，回调 self.update_frame


    def open_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.execute_button.setEnabled(True)
            self.play_original_button.setEnabled(True)
            # self.showMaximized()  # 切换为最大化显示，保留标题栏
            # flags = self.windowFlags()
            # flags = flags | Qt.WindowMaximized
            # self.setWindowFlags(flags)

        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.show_processed_frame(frame)

    def execute_processing(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                processing_method_index = self.processing_method_combobox.currentIndex()
                processing_case_index = self.processing_case.currentIndex()
                self.text_edit.insertPlainText(f"正在获取车道线结构\n")
                self.scroll_to_bottom()
                processed_frame = self.process_video(frame, processing_method_index,processing_case_index)
                self.text_edit.insertPlainText("车道线结构获取完成！\n")
                self.scroll_to_bottom()
                self.show_processed_frame(processed_frame)
            self.video_capture.release()  # 处理完一帧后释放视频资源
            if self.processing_method_combobox.currentIndex() == 1:
                self.save_xml_button.setEnabled(True)
            self.execute_button1.setEnabled(True)

    def load_xml(self):
        print('load')

    def get_out_csv(self):
        self.stop_process_button.setEnabled(True)
        vid_save = self.save_case1.isChecked()
        car_track_save = self.save_case2.isChecked()
        car_num_save = self.save_case3.isChecked()

        self.routes.get_ready(car_track_save,car_num_save, vid_save)
        for terminal_text in self.routes.process():
            self.text_edit.insertPlainText(terminal_text)
            self.scroll_to_bottom()


    def save_xml(self):
        # 语义分割才能保存
        self.routes.save_xml()
        print('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))
        self.text_edit.insertPlainText('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))

    def show_processed_frame(self, processed_frame):
        height, width = processed_frame.shape[:2]
        bytes_per_line = 1 * width if len(processed_frame.shape) == 2 else 3 * width
        q_image_format = QImage.Format_Grayscale8 if len(processed_frame.shape) == 2 else QImage.Format_RGB888
        q_image = QImage(processed_frame.data, width, height, bytes_per_line, q_image_format).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def process_video(self, frame, method_index, case_index):
        """
        根据选择的不同方式对视频帧进行处理
        """
        if method_index == 0:         # 方式1 手划线
            if case_index == 0:
                self.routes = Hand_Draw(self.video_path)
                if self.routes.speed_lane != 0:
                    cv2.line(frame, self.routes.speed_lane[0][0], self.routes.speed_lane[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k + 1):
                        cv2.line(frame, (self.routes.location[0][i][0], self.routes.location[0][i][1]),
                                 (self.routes.location[1][i][0], self.routes.location[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
                if self.routes.speed_lane2 != 0:
                    cv2.line(frame, self.routes.speed_lane2[0][0], self.routes.speed_lane2[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k2 + 1):
                        cv2.line(frame, (self.routes.location2[0][i][0], self.routes.location2[0][i][1]),
                                 (self.routes.location2[1][i][0], self.routes.location2[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            if case_index == 1:
                self.routes = Hand_Draw_Cross(self.video_path)
                if self.routes.speed_lane != 0:
                    cv2.line(frame, self.routes.speed_lane[0][0], self.routes.speed_lane[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k + 1):
                        cv2.line(frame, (self.routes.location[0][i][0], self.routes.location[0][i][1]),
                                 (self.routes.location[1][i][0], self.routes.location[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
                if self.routes.speed_lane1 != 0:
                    cv2.line(frame, self.routes.speed_lane1[0][0], self.routes.speed_lane1[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k1 + 1):
                        cv2.line(frame, (self.routes.location1[0][i][0], self.routes.location1[0][i][1]),
                                 (self.routes.location1[1][i][0], self.routes.location1[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
                if self.routes.speed_lane2 != 0:
                    cv2.line(frame, self.routes.speed_lane2[0][0], self.routes.speed_lane2[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k2 + 1):
                        cv2.line(frame, (self.routes.location2[0][i][0], self.routes.location2[0][i][1]),
                                 (self.routes.location2[1][i][0], self.routes.location2[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
                if self.routes.speed_lane3 != 0:
                    cv2.line(frame, self.routes.speed_lane3[0][0], self.routes.speed_lane3[1][0], (0, 255, 0), 1)  # 绿色，1个像素宽度
                    for i in range(self.routes.k3 + 1):
                         cv2.line(frame, (self.routes.location3[0][i][0], self.routes.location3[0][i][1]),
                             (self.routes.location3[1][i][0], self.routes.location3[1][i][1]), [255, 0, 0], 1)  # 蓝色，3个像素宽度
            return frame
        elif method_index == 1:
            # 方式2 语义分割
            if case_index == 0:
                self.routes = Segmentation(self.video_path)
                frame = draw_road_lines(frame, self.routes.dir, '')
            if case_index == 1:
                self.routes = Segmentation_Cross(self.video_path)
                frame = draw_road_lines(frame, self.routes.dir, '')
            return frame

    def play_original_video(self):
        if self.video_capture is not None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_video_frame)
            self.timer.start(30)  # 大约30ms更新一帧，可调整帧率
            self.pauseBtn.setEnabled(True)
            self.stopBtn.setEnabled(True)

    def update_video_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.show_original_frame(frame)
        else:
            self.timer.stop()
            self.timer.deleteLater()
            self.video_capture.release()
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)

    def show_original_frame(self, frame):
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def stop(self):
        self.stopped = True
        self.timer.stop()  # 关闭定时器
        if self.cap:
            self.cap.release()  # 释放视频流
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self.image_label.clear()  # 清空视频显示区域

    def pause(self):
        if self.paused:
            self.paused = False
            self.pauseBtn.setText('⏸暂停')
            self.timer.start()
        else:
            self.paused = True
            self.pauseBtn.setText('▶继续')
            self.timer.stop()

    def stop_process(self):
        self.routes.exit_process()
        self.text_edit.insertPlainText(f"已经结束该视频的处理，可以重新进行选择。\n")
        self.scroll_to_bottom()
        self.stop_process_button.setEnabled(False)

    def scroll_to_bottom(self):
        scroll_bar = self.text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())



if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
