from moviepy.editor import VideoFileClip


def export_30_seconds_in_range(input_video_path, output_video_path, start_time,dur):
    try:
        # 加载视频文件
        clip = VideoFileClip(input_video_path)
        # 获取视频总时长
        video_duration = clip.duration

        # 计算结束时间
        end_time = start_time + dur
        # 如果结束时间超出视频总时长，则将结束时间设为视频总时长
        if end_time > video_duration:
            end_time = video_duration

        # 若开始时间超出视频总时长，给出提示并返回
        if start_time >= video_duration:
            print("开始时间超出视频总时长，请检查输入。")
            clip.close()
            return

        # 剪辑指定时间范围内的 30 秒视频
        new_clip = clip.subclip(start_time, end_time)

        # 保存剪辑后的视频
        new_clip.write_videofile(output_video_path, codec="libx264")

        # 关闭视频剪辑对象
        clip.close()
        new_clip.close()

        print(f"从 {start_time} 秒到 {end_time} 秒的视频已成功导出到 {output_video_path}")
    except Exception as e:
        print(f"导出视频时出现错误: {e}")


# 示例用法
input_video = r"E:\无人机数据-交管局\中山门上太阳城路路口\DJI_20250319075843_0002_S.mp4"
output_video = r"../source/out_c.mp4"
# 开始时间（单位：秒）
start_time = 0
dur = 10
export_30_seconds_in_range(input_video, output_video, start_time,dur)