import cv2


def save_first_frame(video_path, output_image_path):
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("错误: 无法打开视频文件!")
            return

        # 读取第一帧
        ret, frame = cap.read()

        if ret:
            # 保存第一帧为图片
            cv2.imwrite(output_image_path, frame)
            print(f"第一帧已保存为 {output_image_path}")
        else:
            print("错误: 无法读取视频帧!")

        # 释放视频捕获对象
        cap.release()
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")


if __name__ == "__main__":
    video_path = 'output_DJI_0013_c.mp4'  # 请替换为你的视频文件路径
    output_image_path = 'first_frame.jpg'  # 输出图片的路径
    save_first_frame(video_path, output_image_path)