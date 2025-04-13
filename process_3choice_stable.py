# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import QApplication, QWidget, QBoxLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, \
    QComboBox, QTextEdit, QCheckBox, QSizeGrip
from PySide6.QtCore import QTimer, QPoint, QSize, Qt
from PySide6.QtGui import QMouseEvent, QResizeEvent, QImage, QPixmap
import cv2
from datetime import datetime
# from utils.main_utils import lanemark as lm, calculate_speedlane, roi_mask, roi_mask2
from utils.cross_process import Hand_Draw_Cross, Segmentation_Cross
from utils.highway_process import Hand_Draw, Segmentation
# 初始化摄像头和Yolo模型
from utils.draw_stop_lane import draw_road_lines
# from unet.cross import fit_lanes,p2l_dis
# from utils.save_xml import write_crosses, write_roads
from threading import Thread
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
                # 计算点击位置比例用于平滑过渡
                screen_rect = QApplication.primaryScreen().availableGeometry()
                mouse_x = event.globalPosition().toPoint().x()
                width = window_param.width()
                normalized_x = mouse_x * (width / screen_rect.width())

                window_param.showNormal()
                new_x = mouse_x - normalized_x
                window_param.move(int(new_x), 0)
                self._drag_pos = QPoint(int(normalized_x), int(event.position().y()))

            # 移动窗口
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

class ResizableQLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        print(f"Image label new size: {event.size()}")


def clear_layout(layout: QBoxLayout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        layout.removeWidget(widget)
        widget.setParent(None)  # 将小部件的父类设置为None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 设置界面
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowMinMaxButtonsHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.title_bar = None
        self.btn_xml_process = None
        self.btn_video_open = None
        self.btnLayout_video = None
        self.btn_video_play = None
        self.btn_video_stop = None
        self.btnLayout_H2 = None
        self.btn_execute_line = None
        self.btn_execute_traffic = None
        self.btn_execute_stop = None
        self.size_grip = None

        self.processing_method_combobox = None
        self.processing_case = None
        self.save_case1 = None
        self.save_case2 = None
        self.save_case3 = None
        self.text_edit = None
        self.image_label = None

        self.video_path = None
        self.video_capture = None
        self.data_xml_ready = False
        self.video_played = False
        self.video_stopped = True

        self.timer_play_frame = QTimer(self)
        self.timer_update_frame = QTimer(self)
        self.timer_update_frame_flag = False
        self.timer_update_frame.timeout.connect(self.timer_func_update_frame)

        self.thread_get_road_lines = None
        self.thread_get_traffic_out = None
        self.thread_running_flag = False

        self.routes = None
        self.show_current_frame = None
        self.transparent_pixmap = QPixmap(4000, 4000)
        self.transparent_pixmap.fill(Qt.GlobalColor.transparent)

        self.setupUI()

        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []


    def timer_func_update_frame(self):
        if self.show_current_frame is not None:
            self.show_processed_frame(self.show_current_frame)
            self.timer_update_frame.stop()
            self.timer_update_frame_flag = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 重新设置 QSizeGrip 的位置
        self.size_grip.move(self.width() - 20, self.height() - 20)
        if self.show_current_frame is not None:
            self.image_label.setPixmap(self.transparent_pixmap)
            self.timer_update_frame_flag = True
            if self.timer_update_frame_flag and self.timer_update_frame.isActive() == False:
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

        btnLayout_H0 = QHBoxLayout()
        self.btn_xml_process = QPushButton('导入道路结构')
        self.btn_xml_process.clicked.connect(self.button_xml_process)
        btnLayout_H0.addWidget(self.btn_xml_process)

        self.btnLayout_video = QHBoxLayout()
        self.btn_video_open = QPushButton('选择视频')
        self.btn_video_open.clicked.connect(self.button_video_open)
        self.btnLayout_video.addWidget(self.btn_video_open)
        btnLayout_H0.addLayout(self.btnLayout_video)

        self.btn_video_play = QPushButton("播放原视频")
        self.btn_video_play.clicked.connect(self.button_video_play)
        self.btn_video_stop = QPushButton('关闭视频')
        self.btn_video_stop.clicked.connect(self.button_video_stop)

        layout_main_HA.addLayout(btnLayout_H0, stretch=2)

        btnLayout_H1 = QHBoxLayout()
        self.processing_method_combobox = QComboBox()
        self.processing_method_combobox.addItems(["利用手划线车道线作为车道位置", "利用语义分割模型识别车道线"])  # 根据实际处理方式添加选项
        self.processing_method_combobox.setStyleSheet("QComboBox{ color: white; }")
        btnLayout_H1.addWidget(self.processing_method_combobox)

        self.processing_case = QComboBox()
        self.processing_case.addItems(["高速/高架", "路口"])  # 根据实际处理方式添加选项
        self.processing_case.setStyleSheet("QComboBox{ color: white; }")
        btnLayout_H1.addWidget(self.processing_case)

        checkLayout = QHBoxLayout()
        self.save_case1 = QCheckBox("视频")
        checkLayout.addWidget(self.save_case1)
        self.save_case2 = QCheckBox("轨迹")
        checkLayout.addWidget(self.save_case2)
        self.save_case3 = QCheckBox("分车道车流量")
        checkLayout.addWidget(self.save_case3)
        btnLayout_H1.addLayout(checkLayout)
        layout_main_HA.addLayout(btnLayout_H1, stretch=2)

        self.btnLayout_H2 = QHBoxLayout()
        self.btn_execute_line = QPushButton("获取车道线信息")
        self.btn_execute_line.clicked.connect(self.start_process_video)
        self.btn_execute_line.setEnabled(False)  # 初始未选视频时不可用
        self.btnLayout_H2.addWidget(self.btn_execute_line)

        self.btn_execute_traffic = QPushButton("获取交通流参数")
        self.btn_execute_traffic.clicked.connect(self.start_process_traffic)
        self.btn_execute_traffic.setEnabled(False)  # 初始未生成路网结构时不可用
        self.btnLayout_H2.addWidget(self.btn_execute_traffic)

        self.btn_execute_stop = QPushButton("🛑结束！")
        self.btn_execute_stop.clicked.connect(self.stop_process)

        layout_main_HA.addLayout(self.btnLayout_H2, stretch=2)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout_main_HA.addWidget(self.text_edit, stretch=1)

        layoutVid = QHBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px solid #20CBA2;")  # 设置2像素宽的红色边框
        self.image_label.setPixmap(self.transparent_pixmap)
        layoutVid.addWidget(self.image_label, stretch=1)
        layoutVid.setAlignment(self.image_label, Qt.AlignmentFlag.AlignCenter)

        layout_main.addLayout(layout_main_HA, stretch=1)
        layout_main.addLayout(layoutVid, stretch=9)

        self.setLayout(layout_main)
        self.resize(800, 600)

        # 创建一个 QSizeGrip 对象，并将其添加到窗口中
        self.size_grip = QSizeGrip(self)
        self.size_grip.setGeometry(self.width() - 20, self.height() - 20, 20, 20)
        # self.timer = QTimer(self)  # 定义定时器，用于控制显示视频的帧率
        # self.timer.timeout.connect(self.update_frame)  # 定时到了，回调 self.update_frame


    def button_video_open(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            # self.showMaximized()  # 切换为最大化显示，保留标题栏
            # flags = self.windowFlags()
            # flags = flags | Qt.WindowMaximized
            # self.setWindowFlags(flags)
            self.btnLayout_video.removeWidget(self.btn_video_open)
            self.btn_video_open.setParent(None)
            self.btnLayout_video.addWidget(self.btn_video_play)
            self.btnLayout_video.addWidget(self.btn_video_stop)
            self.btn_execute_line.setEnabled(True)

        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.show_current_frame = frame
                self.show_processed_frame(frame)
                self.btn_video_play.setText("播放原视频")
                self.systerm_status_echo("视频已打开！")

    def button_video_play(self):
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
                    self.systerm_status_echo("视频播放已暂停！")
                else:
                    self.timer_play_frame.stop()
                    self.btn_video_play.setText('▶继续')
                    self.video_stopped = True
                    self.systerm_status_echo("视频播放已继续！")

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
        self.video_played = False
        self.video_stopped = True
        self.btn_execute_line.setEnabled(False)
        self.btn_execute_traffic.setEnabled(False)
        self.systerm_status_echo("视频已关闭！")

        self.btnLayout_video.removeWidget(self.btn_video_play)
        self.btnLayout_video.removeWidget(self.btn_video_stop)
        self.btn_video_play.setParent(None)
        self.btn_video_stop.setParent(None)
        self.btnLayout_video.addWidget(self.btn_video_open)

    def update_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                self.show_current_frame = frame
                self.show_original_frame(frame)

    def button_xml_process(self):
        print('button_xml_process')
        if self.data_xml_ready:
            # 语义分割才能保存
            self.routes.save_xml()
            print('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))
            self.systerm_status_echo('将路网信息保存到xml中，地址为' + str(self.routes.xmlfile))
            self.data_xml_ready = False

    def start_process_video(self):
        self.thread_get_road_lines = Thread(target=self.get_road_lines, args=())
        # self.thread_get_road_lines.start()
        print('start_process_video')
        self.thread_running_flag = True
        clear_layout(self.btnLayout_H2)
        self.btnLayout_H2.addWidget(self.btn_execute_line)
        self.btnLayout_H2.addWidget(self.btn_execute_stop)

    def start_process_traffic(self):
        self.thread_get_traffic_out = Thread(target=self.get_traffic_out_csv, args=())
        # self.thread_get_traffic_out.start()
        print('start_process_traffic')
        self.thread_running_flag = True
        clear_layout(self.btnLayout_H2)
        self.btnLayout_H2.addWidget(self.btn_execute_traffic)
        self.btnLayout_H2.addWidget(self.btn_execute_stop)

    def get_road_lines(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                processing_method_index = self.processing_method_combobox.currentIndex()
                processing_case_index = self.processing_case.currentIndex()
                self.systerm_status_echo(f"正在获取车道线结构")

                processed_frame = self.process_video(frame, processing_method_index,processing_case_index)
                if processed_frame != -1:
                    self.systerm_status_echo("车道线结构获取完成！")
                else:
                    self.thread_running_flag = False
                    self.systerm_status_echo("车道线结构获取失败！")
                    return
                self.show_processed_frame(processed_frame)
            self.video_capture.release()  # 处理完一帧后释放视频资源
            if self.processing_method_combobox.currentIndex() == 1:
                self.data_xml_ready = True
                self.systerm_status_echo('保存路网结构')
                self.btn_execute_traffic.setEnabled(True)
        self.thread_running_flag = False

    def process_video(self, frame, method_index, case_index):
        """
        根据选择的不同方式对视频帧进行处理
        """
        try:
            if method_index == 0:         # 方式1 手划线
                self.systerm_status_echo("请手动划车道线")
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
            elif method_index == 1:
                # 方式2 语义分割
                if case_index == 0:
                    self.routes = Segmentation(self.video_path)
                    frame = draw_road_lines(frame, self.routes.dir, '')
                if case_index == 1:
                    self.routes = Segmentation_Cross(self.video_path)
                    frame = draw_road_lines(frame, self.routes.dir, '')
            self.btn_execute_traffic.setEnabled(True)
            return frame
        except Exception as e:
            self.systerm_status_echo(f"在处理 case_index={case_index} 时发生异常: {e}")
            return -1

    def show_original_frame(self, frame):
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def show_processed_frame(self, processed_frame):
        height, width = processed_frame.shape[:2]
        bytes_per_line = 1 * width if len(processed_frame.shape) == 2 else 3 * width
        q_image_format = QImage.Format.Format_Grayscale8 if len(processed_frame.shape) == 2 else QImage.Format.Format_RGB888
        q_image = QImage(processed_frame.data, width, height, bytes_per_line, q_image_format).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

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
        clear_layout(self.btnLayout_H2)
        self.btnLayout_H2.addWidget(self.btn_execute_line)
        self.btnLayout_H2.addWidget(self.btn_execute_traffic)

    def get_traffic_out_csv(self):
        self.systerm_status_echo("正在获取交通流参数")
        vid_save = self.save_case1.isChecked()
        car_track_save = self.save_case2.isChecked()
        car_num_save = self.save_case3.isChecked()

        self.routes.get_ready(car_track_save,car_num_save, vid_save)
        for terminal_text in self.routes.process():
            self.text_edit.insertPlainText(terminal_text)
            self.scroll_to_bottom()
        self.thread_running_flag = False

    def systerm_status_echo(self, text):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_edit.insertPlainText(f"{current_time} " + text + "\n")
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        scroll_bar = self.text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())



if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
