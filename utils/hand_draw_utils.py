import cv2
import numpy as np
import time

global point1, point2

def lanemark(img: np.ndarray):
    def on_mouse(event, x, y, flags):
        global point1, point2
        img2 = img.copy()
        # 左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            point1 = (x, y)
            record1.append(point1)
            cv2.circle(img2, point1, 10, (0, 255, 0), 2)
            for i in range(np.array(record2).shape[0]):
                cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)  # 画之前划过的线
            cv2.imshow('image', img2)
        # 按住左键拖曳
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            for i in range(np.array(record2).shape[0]):
                cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)
            cv2.line(img2, point1, (x, y), (255, 0, 0), 2)  # 显示拖拽时上一次左键点击和点钱拖拽点间的连线
            cv2.imshow('image', img2)
        # 左键释放
        elif event == cv2.EVENT_LBUTTONUP:
            point2 = (x, y)
            record2.append(point2)
            for i in range(np.array(record1).shape[0]):
                cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)

            cv2.imshow('image', img2)
            # min_x = min(point1[0], point2[0])
            # min_y = min(point1[1], point2[1])
            # width = abs(point1[0] - point2[0])
            # height = abs(point1[1] - point2[1])
            # cut_img = img[min_y:min_y + height, min_x:min_x + width]
            #cv2.imwrite('lena3.jpg', cut_img)
        # 中键点击（撤销画的上一条线）
        elif event == cv2.EVENT_MBUTTONDOWN:
            record1.pop(np.array(record1).shape[0] - 1)
            record2.pop(np.array(record2).shape[0] - 1)
            for i in range(np.array(record1).shape[0]):
                cv2.line(img2, record1[i], record2[i], (0, 0, 255), 2)
            cv2.imshow('image', img2)


    # WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    # HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

    record = []
    record1 = []
    record2 = []

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    record.append(record1)
    record.append(record2)

    if record ==[[],[]]:
        record = ' '
    else:
        print('The Location has been marked successfully ')
        print('lane(location)', record)

    return record


from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QPixmap, QImage
from PySide6.QtWidgets import QLabel


def array_to_qimage(array):
    format_type = QImage.Format.Format_Invalid
    if array.dtype == np.uint8:
        if len(array.shape) == 2:
            format_type = QImage.Format.Format_Grayscale8
        elif len(array.shape) == 3:
            if array.shape[2] == 3:
                format_type = QImage.Format.Format_RGB888
            elif array.shape[2] == 4:
                format_type = QImage.Format.Format_ARGB32
    else:
        raise ValueError("Unsupported array type")

    height, width = array.shape[:2]
    bytes_per_line = array.strides[0]
    qimage = QImage(array.data, width, height, bytes_per_line, format_type).rgbSwapped()
    return qimage


class ImageLabel(QLabel):
    userFinished = Signal()  # 定义一个用户操作完成的信号
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_init = False
        self.windows = parent
        self.record1 = []
        self.record2 = []
        self.dragging = False
        self.current_pos = None
        self.original_frame = self.windows.show_current_frame
        if not isinstance(self.original_frame, QPixmap):
            self.original_frame = QPixmap.fromImage(array_to_qimage(self.original_frame))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # 允许接收键盘事件

    def frame_reflash(self):
        self.original_frame = self.windows.show_current_frame
        if not isinstance(self.original_frame, QPixmap):
            self.original_frame = QPixmap.fromImage(array_to_qimage(self.original_frame))

    def mousePressEvent(self, event):
        # print("mousePressEvent")
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._convert_pos(event.pos())
            if pos:
                self.record1.append(pos)
                self.dragging = True
                self.update_display()
        elif event.button() == Qt.MouseButton.MiddleButton:  # 中键撤销
            if self.record1:
                self.record1.pop()
                self.record2.pop()
                self.update_display()

    def mouseMoveEvent(self, event):
        # print(event.pos())
        if self.dragging:
            self.current_pos = self._convert_pos(event.pos())
            self.update_display()

    def mouseReleaseEvent(self, event):
        # print("mouseReleaseEvent")
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            pos = self._convert_pos(event.pos())
            if pos and self.record1:
                self.record2.append(pos)
                self.dragging = False
                self.current_pos = None
                self.update_display()

    def keyPressEvent(self, event):  # 捕获键盘事件
        # print("keyPressEvent")
        self.userFinished.emit()
        event.accept()  # 阻止事件继续传播

    def _convert_pos(self, qpoint):
        """将控件坐标转换为图像原始坐标"""
        if self.original_frame is None:
            return None

        # 计算缩放后的显示区域
        img_w = self.original_frame.width()
        img_h = self.original_frame.height()
        lbl_w = self.width()
        lbl_h = self.height()

        scale = min(lbl_w / img_w, lbl_h / img_h)

        return (
            int(qpoint.x() / scale),
            int(qpoint.y() / scale)
        )

    def update_display(self):
        pixmap = self.original_frame.copy()
        painter = QPainter(pixmap)

        # 绘制已确定的线段
        pen = QPen(Qt.GlobalColor.red, 5)
        painter.setPen(pen)
        for p1, p2 in zip(self.record1, self.record2):
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # 绘制当前拖拽的临时线段
        if self.dragging and self.current_pos and self.record1:
            pen.setColor(Qt.GlobalColor.blue)
            painter.setPen(pen)
            last_point = self.record1[-1]
            painter.drawLine(last_point[0], last_point[1],
                             self.current_pos[0], self.current_pos[1])

        painter.end()
        self.windows.update_label_img_signal.emit(pixmap)


def lanelabelmark(img, windows):
    # 如果控件不存在或不是ImageLabel类型，创建并替换
    if windows.image_draw_label is None:
        windows.update_label_signal.emit()
        time.sleep(0.1)
    else:
        return ' '

    # 设置图像
    if not windows.is_drawing_lane:
        windows.update_label_img_signal.emit(img)
        time.sleep(0.1)
        windows.image_draw_label.frame_reflash()
        windows.is_drawing_lane = True
        time.sleep(0.01)
    else:
        windows.show_processed_frame()

    # 连接用户操作完成的信号到槽函数
    def handle_user_finished():
        # 设置一个标志来表示操作已完成
        windows.user_operation_complete = True
        # windows.user_result = result

    windows.image_draw_label.userFinished.connect(handle_user_finished)

    # 使用一个标志来等待用户操作完成
    windows.user_operation_complete = False
    while not windows.user_operation_complete:
        if windows.window_exit_flag:
            raise SystemExit

    #输出结果
    if not windows.image_draw_label.record1:
        result = ' '
    else:
        result = [windows.image_draw_label.record1, windows.image_draw_label.record2]
    windows.update_label_signal.emit()
    time.sleep(0.01)
    return result

