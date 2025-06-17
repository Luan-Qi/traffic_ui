from moviepy.editor import VideoFileClip


def video_to_gif(video_path, gif_path, start_time=0, end_time=None):
    try:
        # 加载视频文件
        video = VideoFileClip(video_path)

        # 选择要转换的时间段
        if end_time is None:
            end_time = video.duration

        video = video.subclip(start_time, end_time)

        # 保存为 GIF
        video.write_gif(gif_path)

        # 关闭视频文件
        video.close()
        print(f"成功将视频转换为 GIF 并保存到 {gif_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


if __name__ == "__main__":
    video_file = "../inference/output/out_c/out_c.mp4"
    gif_file = "../output/output.gif"
    video_to_gif(video_file, gif_file)