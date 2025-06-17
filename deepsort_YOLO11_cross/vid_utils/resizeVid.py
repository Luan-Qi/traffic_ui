from moviepy.editor import VideoFileClip


def convert_to_1080p(input_path, output_path):
    try:
        clip = VideoFileClip(input_path)
        # 将视频分辨率调整为 1920x1080
        resized_clip = clip.resize(height=1080)
        resized_clip.write_videofile(output_path, codec="libx264")
        clip.close()
        print(f"视频已成功转换并保存到 {output_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")


if __name__ == "__main__":
    input_video = "DJI_0013_c.mp4"
    output_video = "output_DJI_0013_c.mp4"
    convert_to_1080p(input_video, output_video)