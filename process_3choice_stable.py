# -*- coding: utf-8 -*-
import os
import sys
import traceback

from PySide6.QtWidgets import QApplication, QWidget, QBoxLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, \
    QComboBox, QTextEdit, QCheckBox, QSizeGrip, QSizePolicy
from PySide6.QtCore import QTimer, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QImage, QPixmap
import cv2
import time
import numpy as np
import random
from track import VehicleTracker
from datetime import datetime
# from utils.main_utils import draw_bounding_boxes
# from utils.cross_process import Segmentation_Cross
# from utils.highway_process import Segmentation
# 初始化摄像头和Yolo模型
from utils.draw_stop_lane import draw_road_lines
from utils.read_xml import xml_to_dict_for_dir
from utils.hand_draw_utils import ImageLabel
from threading import Thread
from qt_material import apply_stylesheet

window_exit_flag = False

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
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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
        self.close_btn.clicked.connect(self.function_close)

    def _init_style(self):
        self.min_btn.setObjectName("min_btn")
        self.max_btn.setObjectName("max_btn")
        self.close_btn.setObjectName("close_btn")
        self.title_label.setObjectName("title_label")
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
            QWidget#titleBar QLabel#title_label {
                font-weight: bold !important;  /* 设置字体为加粗 */
                color: white !important;
                font-family: "Segoe UI Symbol";
                font-size: 14px;  /* 根据需要调整字体大小 */
        }
        """)

    def function_close(self):
        self.window().close()
        self.window().window_exit_flag = True

    def toggle_maximize(self):
        window_param = self.window()
        if window_param.isMaximized():
            window_param.showNormal()
            self.max_btn.setText('□')
        else:
            window_param.showMaximized()
            self.max_btn.setText('❐')

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            window_param = self.window()
            if window_param.isMaximized():
                # 获取当前鼠标所在的屏幕
                current_screen = QApplication.screenAt(event.globalPosition().toPoint())
                screen_num = QApplication.screens().index(current_screen)  # 通过查找索引来获取屏幕编号
                screen_rect = QApplication.screens()[screen_num].availableGeometry()
                # 计算点击位置比例用于平滑过渡
                # screen_rect = QApplication.primaryScreen().availableGeometry()
                window_param.showNormal()
                mouse_x = event.globalPosition().toPoint().x()
                new_width = window_param.width()
                normalized_x = new_width * (mouse_x / screen_rect.width())  # 调整为相对屏幕的x坐标

                new_x = mouse_x - normalized_x  # 假设你希望窗口中心与鼠标位置对齐
                window_param.move(int(new_x), 0)
                self._drag_pos = QPoint(int(normalized_x), int(event.position().y()))

            delta = event.globalPosition().toPoint() - self._drag_pos
            window_param.move(window_param.x() + delta.x(), window_param.y() + delta.y())
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.toggle_maximize()
        event.accept()


class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(AspectRatioLabel, self).__init__(*args, **kwargs)
        self.original_pixmap = QPixmap()

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        super(AspectRatioLabel, self).setPixmap(pixmap)

    def resizeEvent(self, event):
        if not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            super(AspectRatioLabel, self).setPixmap(scaled_pixmap)
        super(AspectRatioLabel, self).resizeEvent(event)

def clear_layout_only_widght(layout: QBoxLayout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        layout.removeWidget(widget)
        widget.setParent(None)  # 将小部件的父类设置为None

def extract_border_color(stylesheet):
    # 使用正则表达式解析样式表以提取边框颜色
    import re
    match = re.search(r'border:\s*\d+px\s+solid\s+(#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3});', stylesheet)
    if match:
        return match.group(1)
    return "black"  # 默认颜色


class MainWindow(QWidget):
    update_label_signal = Signal()
    update_label_img_signal = Signal(np.ndarray)
    system_echo_signal = Signal(str)
    timer_wait_signal = Signal(bool)
    def __init__(self):
        super().__init__()
        # 设置界面
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowMinMaxButtonsHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.title_bar = None
        self.layout_label = None
        self.btnLayout_H0 = None
        self.btnLayout_H1 = None
        self.btn_xml_process = None
        self.btn_video_open = None
        self.btnLayout_video = None
        self.btn_video_play = None
        self.btn_video_stop = None
        self.btn_execute_road = None
        self.btn_execute_traffic = None
        self.btn_execute_stop = None
        self.size_grip = None

        self.processing_method_combobox = None
        self.save_case1 = None
        self.save_case2 = None
        self.save_case3 = None
        self.text_edit = None
        self.image_label = None
        self.image_draw_label = None

        self.video_path = None
        self.video_capture = None
        self.xml_path = None
        self.csv_path = None

        self.data_road_dir_ready = False
        self.data_xml_ready = False
        self.video_played = False
        self.video_stopped = True

        self.timer_play_frame = QTimer(self)
        self.timer_update_frame = QTimer(self)
        self.timer_update_frame_flag = False
        self.timer_update_frame.timeout.connect(self.timer_func_update_frame)
        self.timer_scroll_txt = QTimer(self)
        self.timer_scroll_txt_flag = False
        self.timer_scroll_txt.setInterval(200)  # 设置定时器间隔为500毫秒
        self.timer_scroll_txt.setSingleShot(True)  # 设置为单次触发
        self.timer_scroll_txt.timeout.connect(self.scroll_to_bottom)
        self.timer_wait_sec = QTimer(self)
        self.timer_wait_sec.timeout.connect(self.timer_wait_echo)
        self.timer_wait_times = 0

        self.thread_get_road_lines = None
        self.thread_get_traffic_out = None
        self.thread_running_flag = False

        self.routes = None
        self.show_current_frame = None
        self.transparent_pixmap = QPixmap(4000, 3000)
        self.transparent_pixmap.fill(Qt.GlobalColor.transparent)
        self.windows_Maximized_width = 0
        self.windows_Normal_width = 0
        self.window_exit_flag = False

        self.setupUI()
        self.update_label_signal.connect(self.update_label_slot)
        self.update_label_img_signal.connect(self.show_processed_frame_later)
        self.timer_wait_signal.connect(self.timer_wait_sec_func)
        self.system_echo_signal.connect(self.systerm_status_echo)

        self.multi_thread_lock = False
        self.is_drawing_lane = False

        self.developer_lock = True

    def timer_wait_sec_func(self, statue):
        if statue and self.timer_wait_sec.isActive() == False:
            self.timer_wait_times = 1
            self.timer_wait_sec.start(5000)
        else:
            self.timer_wait_sec.stop()

    def timer_wait_echo(self):
        echo_str = f"视频已经处理{self.timer_wait_times * 5}秒，请稍等..."
        self.systerm_status_echo(echo_str)
        self.timer_wait_times += 1


    def timer_func_update_frame(self):
        self.show_processed_frame()
        self.timer_update_frame.stop()
        self.timer_update_frame_flag = False

        if not self.video_stopped:# 如果视频在播放中则恢复播放
            self.timer_play_frame.start(30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 重新设置 QSizeGrip 的位置
        self.size_grip.move(self.width() - 20, self.height() - 20)
        if self.show_current_frame is not None:
            if not self.video_stopped:# 防止视频刷新从而影响窗口大小的更新
                self.timer_play_frame.stop()

            if self.image_draw_label is not None:
                self.image_draw_label.setPixmap(self.transparent_pixmap)
            else:
                self.image_label.setPixmap(self.transparent_pixmap)
            self.timer_update_frame_flag = True
            if self.timer_update_frame_flag and self.timer_update_frame.isActive() == False:
                self.timer_update_frame.start(50)

        if self.isMaximized():
            self.windows_Maximized_width = self.width()
        else:
            self.windows_Normal_width = self.width()

    def update_label_slot(self):
        if self.image_draw_label is None:
            self.image_draw_label = ImageLabel(self)
            self.image_draw_label.setStyleSheet("border: 2px solid #0CFF4A;")
            self.layout_label.removeWidget(self.image_label)
            self.image_label.setParent(None)
            self.layout_label.addWidget(self.image_draw_label, stretch=1)
            self.image_draw_label.setPixmap(self.transparent_pixmap)
            self.layout_label.setAlignment(self.image_draw_label, Qt.AlignmentFlag.AlignCenter)
            self.image_draw_label.setFocus()  # 确保获得键盘焦点
        else:
            self.layout_label.removeWidget(self.image_draw_label)
            self.image_draw_label.setParent(None)
            self.image_draw_label.deleteLater()
            self.image_draw_label = None
            self.layout_label.addWidget(self.image_label)
            self.image_label.setPixmap(self.transparent_pixmap)
            self.layout_label.setAlignment(self.image_label, Qt.AlignmentFlag.AlignCenter)# 确保边框紧贴
            if self.show_current_frame is not None:
                if not self.timer_update_frame.isActive():
                    self.timer_update_frame.start(50)

    def setupUI(self):
        #self.resize(800, 600)
        self.setWindowTitle('交通流信息提取')
        # 设置窗口的最小和最大大小
        self.setMinimumSize(800, 600)  # 最小宽度400，最小高度300
        self.setMaximumSize(3840, 2160)

        layout_main = QVBoxLayout()
        layout_main_HA = QVBoxLayout()

        # 添加标题栏
        self.title_bar = tQTitleBar(self)
        layout_main.addWidget(self.title_bar, stretch=1)

        self.btnLayout_H0 = QHBoxLayout()

        self.btnLayout_video = QHBoxLayout()
        self.btn_video_open = QPushButton('选择视频')
        self.btn_video_open.clicked.connect(self.button_video_open)
        self.btnLayout_video.addWidget(self.btn_video_open)
        self.btnLayout_H0.addLayout(self.btnLayout_video)

        self.btn_xml_process = QPushButton('导入道路结构')
        self.btn_xml_process.clicked.connect(self.button_xml_process)
        self.btnLayout_H0.addWidget(self.btn_xml_process)

        self.btn_video_play = QPushButton("播放原视频")
        self.btn_video_play.clicked.connect(self.button_video_play)
        self.btn_video_stop = QPushButton('关闭视频')
        self.btn_video_stop.clicked.connect(self.button_video_stop)

        layout_main_HA.addLayout(self.btnLayout_H0, stretch=2)

        self.btnLayout_H1 = QHBoxLayout()

        chooseLayout = QHBoxLayout()
        self.processing_method_combobox = QComboBox()
        self.processing_method_combobox.addItems(["手划车道线获取路网结构-1 ",
                                                  "手划车道线获取路网结构-2",
                                                  "手划车道线获取路网结构-3",
                                                  "手划车道线获取路网结构-4",
                                                  "语义分割识别路网结构-道路",
                                                  "语义分割识别路网结构-路口"])  # 根据实际处理方式添加选项
        self.processing_method_combobox.setStyleSheet("QComboBox{ color: white; font-weight: bold; }")
        chooseLayout.addWidget(self.processing_method_combobox)

        checkLayout = QHBoxLayout()
        self.save_case1 = QCheckBox("视频")
        checkLayout.addWidget(self.save_case1)
        self.save_case2 = QCheckBox("轨迹")
        checkLayout.addWidget(self.save_case2)
        self.save_case3 = QCheckBox("分车道车流量")
        checkLayout.addWidget(self.save_case3)

        self.btnLayout_H1.addLayout(chooseLayout, stretch=1)
        self.btnLayout_H1.addLayout(checkLayout, stretch=1)
        layout_main_HA.addLayout(self.btnLayout_H1, stretch=1)

        self.btn_execute_road = QPushButton("获取车道线信息")
        self.btn_execute_road.clicked.connect(self.start_process_road)

        self.btn_execute_traffic = QPushButton("获取交通流信息")
        self.btn_execute_traffic.clicked.connect(self.start_process_traffic)

        self.btn_execute_stop = QPushButton("🛑结束！")
        self.btn_execute_stop.clicked.connect(self.stop_process)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout_main_HA.addWidget(self.text_edit, stretch=1)

        self.layout_label = QHBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setObjectName("image_label")  # 设置控件的名称
        source_stylesheet = self.btn_execute_stop.styleSheet()
        border_color = extract_border_color(source_stylesheet)
        self.image_label.setStyleSheet(f"border: 2px solid {border_color};")  # 设置2像素宽的红色边框
        self.image_label.setStyleSheet("border: 2px solid #20CBA2;")  # 设置2像素宽的红色边框
        self.image_label.setPixmap(self.transparent_pixmap)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout_label.addWidget(self.image_label, stretch=1)
        self.layout_label.setAlignment(self.image_label, Qt.AlignmentFlag.AlignCenter)

        layout_main.addLayout(layout_main_HA, stretch=1)
        layout_main.addLayout(self.layout_label, stretch=9)

        self.setLayout(layout_main)
        self.resize(800, 600)

        # 创建一个 QSizeGrip 对象，并将其添加到窗口中
        self.size_grip = QSizeGrip(self)
        self.size_grip.setGeometry(self.width() - 20, self.height() - 20, 20, 20)


    def button_video_open(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.btnLayout_video.removeWidget(self.btn_video_open)
            self.btn_video_open.setParent(None)
            self.btnLayout_video.addWidget(self.btn_video_play)
            self.btnLayout_video.addWidget(self.btn_video_stop)
        else:
            self.systerm_status_echo("未选择视频文件！")
            return

        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.show_current_frame = frame.copy()
                self.show_processed_frame(frame)
                self.btnLayout_H0.addWidget(self.btn_execute_road)
                self.btnLayout_H1.addWidget(self.btn_execute_traffic, stretch=1)
                self.btn_video_play.setText("播放原视频")
                self.systerm_status_echo("视频已打开！")
                self.routes = VehicleTracker(self.video_path)

    def button_video_play(self):
        if self.thread_running_flag:
            self.systerm_status_echo("视频处理中，无法播放视频，请等待处理完成！")
            return

        if self.video_capture is not None:
            if not self.video_played:
                self.timer_play_frame.timeout.connect(self.update_video_frame)
                self.timer_play_frame.start(30)  # 大约30ms更新一帧，可调整帧率
                self.btn_video_play.setText('⏸暂停')
                self.video_played = True
                self.video_stopped = False
                self.systerm_status_echo("视频正在播放！")
            else:
                if self.video_stopped:
                    self.timer_play_frame.start()
                    self.btn_video_play.setText('⏸暂停')
                    self.video_stopped = False
                    self.systerm_status_echo("视频播放已继续！")
                else:
                    self.timer_play_frame.stop()
                    self.btn_video_play.setText('▶继续')
                    self.video_stopped = True
                    self.systerm_status_echo("视频播放已暂停！")

    def button_video_stop(self):
        if self.thread_running_flag:
            self.systerm_status_echo("视频处理中，无法关闭！")
            return
        if self.video_played and not self.video_stopped:
            self.timer_play_frame.stop()  # 关闭定时器
        if self.video_capture is not None:
            self.video_capture.release()
        self.image_label.setPixmap(self.transparent_pixmap)
        self.show_current_frame = None
        del self.routes
        self.routes = None
        self.video_path = None
        self.video_capture = None
        self.xml_path = None
        self.csv_path = None
        self.data_road_dir_ready = False
        self.data_xml_ready = False
        self.video_played = False
        self.video_stopped = True
        self.timer_play_frame = QTimer(self)
        self.text_edit.clear()
        self.btn_xml_process.setText("导入道路结构")

        self.btnLayout_H0.removeItem(self.btnLayout_H0.itemAt(2))
        self.btn_execute_road.setParent(None)
        self.btnLayout_H1.removeItem(self.btnLayout_H1.itemAt(2))
        self.btn_execute_traffic.setParent(None)
        self.btnLayout_video.removeWidget(self.btn_video_play)
        self.btnLayout_video.removeWidget(self.btn_video_stop)
        self.btn_video_play.setParent(None)
        self.btn_video_stop.setParent(None)
        self.btnLayout_video.addWidget(self.btn_video_open)

        self.systerm_status_echo("视频已关闭！")

    def update_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                self.video_capture.release()
                self.video_capture = cv2.VideoCapture(self.video_path)
                ret, frame = self.video_capture.read()
                if not ret:
                    self.systerm_status_echo("视频播放异常，错误未知！")
                    if self.timer_play_frame is not None:
                        self.timer_play_frame.stop()
                    return
            self.show_current_frame = frame.copy()
            if self.routes is not None:
                self.show_processed_frame(frame, self.routes.data_dir)
            else:
                self.show_processed_frame(frame)

    def button_xml_process(self):
        if self.video_capture is None:
            self.systerm_status_echo("视频未导入，请先导入视频！")
            return
        if self.thread_running_flag:
            self.systerm_status_echo("视频处理中，请等待处理完成！")
            return

        if self.data_road_dir_ready:
            if self.data_xml_ready:
                self.xml_path = QFileDialog.getExistingDirectory(self, "选择保存路网的文件夹", "")
                if not self.xml_path:
                    self.systerm_status_echo('未选择路径！')
                    return
                method_index = self.processing_method_combobox.currentIndex()
                self.routes.save_xml(self.xml_path, method_index)
                print('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))
                self.systerm_status_echo('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))
            else:
                self.systerm_status_echo('路网文件从外部导入，请勿重复导出！')
        else:
            self.xml_path, _ = QFileDialog.getOpenFileName(self, "选择XML文件", "", "路网文件 (*.xml)")
            if self.xml_path:
                self.routes.Data_Dir_Import(xml_to_dict_for_dir(self.xml_path))

                self.show_processed_frame(self.show_current_frame, self.routes.data_dir)
                self.data_xml_ready = False
                self.data_road_dir_ready = True
                self.btn_xml_process.setText("道路结构已经导入")
                self.systerm_status_echo('路网结构文件已导入！')
            else:
                self.systerm_status_echo('未选择路网结构文件！')

    def start_process_road(self):
        if self.thread_running_flag:
            self.systerm_status_echo("视频处理中，请耐心等待！")
            return

        if self.data_road_dir_ready:
            self.systerm_status_echo("道路信息已经存在，请勿重复获取！")
            return

        # 先停止播放视频，并且确定当前画面为处理的画面
        if self.video_played and not self.video_stopped:
            self.button_video_play()

        self.thread_get_road_lines = Thread(target=self.get_road_lines, args=())
        self.thread_get_road_lines.start()
        print('start_process_video')
        self.thread_running_flag = True

    def start_process_traffic(self):
        if self.thread_running_flag:
            self.systerm_status_echo("视频处理中，请耐心等待！")
            return

        if not self.data_road_dir_ready:
            self.systerm_status_echo("请先获取道路结构信息！")
            return

        # 先停止播放视频，并且确定当前画面为处理的画面
        if self.video_played and not self.video_stopped:
            self.button_video_play()

        self.systerm_status_echo('请选择输出路径')
        self.csv_path = QFileDialog.getExistingDirectory(self, "选择保存车流的文件夹", "")
        if not self.csv_path:
            self.systerm_status_echo('未选择路径！')
            return
        self.systerm_status_echo(f'选择的保存文件夹路径为: {self.csv_path}')

        if self.data_road_dir_ready:
            self.thread_get_traffic_out = Thread(target=self.get_traffic_out_csv, args=())
            self.thread_get_traffic_out.start()
            print('start_process_traffic')
            self.thread_running_flag = True

    def get_road_lines(self):
        processing_method_index = self.processing_method_combobox.currentIndex()
        self.system_echo_signal.emit(f"正在获取车道线结构")

        processed_frame = self.process_video(self.show_current_frame, processing_method_index)
        if processed_frame is not None:
            self.system_echo_signal.emit("车道线结构获取完成！")
        else:
            self.thread_running_flag = False
            self.data_road_dir_ready = False
            self.system_echo_signal.emit("车道线结构获取失败！请检查问题重新开始！")
            self.btn_xml_process.setText("导入道路结构")
            return
        self.show_processed_frame(processed_frame, self.routes.data_dir)
        self.btn_xml_process.setText("保存道路结构")
        self.data_xml_ready = True
        self.data_road_dir_ready = True
        self.thread_running_flag = False
        self.is_drawing_lane = False

    def process_video(self, frame, method_index):
        try:
            time.sleep(0.1)
            vid_save = self.save_case1.isChecked()
            car_track_save = self.save_case2.isChecked()
            car_num_save = self.save_case3.isChecked()
            self.routes.initialication(vid_save, car_track_save, car_num_save)
            if method_index <= 3:         # 方式1 手划线
                self.system_echo_signal.emit("请手动划车道线")
                self.routes.Hand_Draw(method_index + 1, self)
            elif method_index == 4:
                self.timer_wait_signal.emit(True)
                self.routes.Segmentation()
                self.timer_wait_signal.emit(False)
            elif method_index == 5:
                self.timer_wait_signal.emit(True)
                self.routes.Segmentation_Cross()
                self.timer_wait_signal.emit(False)
            return frame
        except Exception as e:
            tb = traceback.format_exc()  # 获取完整的回溯信息
            print(f"在处理时发生异常: {tb}")
            self.system_echo_signal.emit(f"在处理时发生异常: {e}")
            self.timer_wait_sec_func(False)
            self.is_drawing_lane = False
            time.sleep(0.1)
            return None

    def show_processed_frame(self, frame = None, frame_dir = None):
        if frame is None:
            frame = self.show_current_frame.copy()

        if not isinstance(frame, QPixmap):
            if frame_dir is not None:
                frame = draw_road_lines(frame, frame_dir, '')
            height, width = frame.shape[:2]
            bytes_per_line = 1 * width if len(frame.shape) == 2 else 3 * width
            q_image_format = QImage.Format.Format_Grayscale8 if len(frame.shape) == 2 else QImage.Format.Format_RGB888
            q_image = QImage(frame.data, width, height, bytes_per_line, q_image_format).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
        else:
            pixmap = frame

        if self.image_draw_label is not None:
            scaled_pixmap = pixmap.scaled(self.image_draw_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_draw_label.setPixmap(scaled_pixmap)
        else:
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)


    def show_processed_frame_later(self, frame, delay=1):
        if self.multi_thread_lock:
            return
        if self.show_current_frame is not None:
            self.timer_update_frame_flag = True
            self.show_current_frame = frame.copy()
            if not self.timer_update_frame.isActive():
                self.timer_update_frame.start(delay)


    def stop_process(self):
        # self.routes.exit_process()
        self.systerm_status_echo("已经结束该视频的处理，可以重新进行选择。")
        if self.thread_get_road_lines is not None:
            if self.thread_get_road_lines.is_alive():
                self.thread_get_road_lines.stop()
        if self.thread_get_traffic_out is not None:
            if self.thread_get_traffic_out.is_alive():
                self.thread_get_traffic_out.stop()
        self.thread_running_flag = False

    def get_traffic_out_csv(self):
        self.system_echo_signal.emit("正在获取交通流参数")

        try:
            bounding_box_save = []
            vid_save = self.save_case1.isChecked()
            car_track_save = self.save_case2.isChecked()
            car_num_save = self.save_case3.isChecked()
            self.routes.initialication(vid_save, car_track_save, car_num_save, self.csv_path)
            for terminal_text in self.routes.run():
                if isinstance(terminal_text, np.ndarray):
                    if terminal_text.ndim > 2:
                        pass
                        # bounding_frame = draw_bounding_boxes(terminal_text, bounding_box_save)
                        # self.update_label_img_signal.emit(bounding_frame)
                        # self.show_processed_frame_later(bounding_frame)
                        # self.show_processed_frame(terminal_text)
                    else:
                        pass
                        #bounding_box_save = terminal_text
                if isinstance(terminal_text, str):
                    self.system_echo_signal.emit(terminal_text)
            self.thread_running_flag = False
            print("get_traffic_out_csv\n")
        except Exception as e:
            tb = traceback.format_exc()  # 获取完整的回溯信息
            print(f"在处理时发生异常: {tb}")
            self.system_echo_signal.emit(f"在处理时发生异常: {e}")
            time.sleep(0.1)
            self.thread_running_flag = False

    def systerm_status_echo(self, text):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_edit.insertPlainText(f"{current_time} " + text + "\n")
        self.scroll_to_bottom()

    def set_scroll_to_bottom(self):
        if not self.timer_scroll_txt_flag and not self.timer_scroll_txt.isActive():
            self.timer_scroll_txt.start()
            self.timer_scroll_txt_flag = True

    def scroll_to_bottom(self):
        scroll_bar = self.text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        self.timer_scroll_txt_flag = False

def select_random_theme():
    themes = ['dark_amber.xml', 'dark_blue.xml', 'dark_cyan.xml', 'dark_lightgreen.xml', 'dark_medical.xml',
              'dark_pink.xml', 'dark_purple.xml', 'dark_red.xml', 'dark_teal.xml', 'dark_yellow.xml']
    # themes = ['dark_amber.xml', 'dark_blue.xml', 'dark_cyan.xml', 'dark_lightgreen.xml', 'dark_medical.xml',
    #           'dark_pink.xml', 'dark_purple.xml', 'dark_red.xml', 'dark_teal.xml', 'dark_yellow.xml',
    #           'light_amber.xml', 'light_blue.xml', 'light_blue_500.xml', 'light_cyan.xml', 'light_cyan_500.xml',
    #           'light_lightgreen.xml', 'light_lightgreen_500.xml', 'light_orange.xml', 'light_pink.xml',
    #           'light_pink_500.xml', 'light_purple.xml', 'light_purple_500.xml', 'light_red.xml', 'light_red_500.xml',
    #           'light_teal.xml', 'light_teal_500.xml', 'light_yellow.xml']
    return random.choice(themes)


def check_files_in_directory(directory_path, required_files):
    """检查目录中是否包含所有必需的文件"""
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 {directory_path} 不存在！")
        return
    existing_files = set(os.listdir(directory_path))
    missing_files = [file for file in required_files if file not in existing_files]

    if missing_files:
        print(f"错误: 目录 {directory_path} 中缺少以下文件: {', '.join(missing_files)}")

def system_check_files():
    check_files_in_directory('config', ['deep_sort.yaml'])
    check_files_in_directory('weights', ['best.pt', 'best407_att.pt', 'ckpt.t7'])


if __name__ == "__main__":
    system_check_files()
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    # selected_theme = select_random_theme()
    # print(f"selected_theme: {selected_theme}")
    # apply_stylesheet(app, theme=selected_theme)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
