import cv2
import sys

def concat_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open one of the videos.")
        return

    # 获取视频参数
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = min(fps1, fps2)

    # 统一高度，宽度相加
    out_height = max(height1, height2)
    out_width = width1 + width2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # 调整帧大小
        frame1 = cv2.resize(frame1, (width1, out_height))
        frame2 = cv2.resize(frame2, (width2, out_height))

        concat_frame = cv2.hconcat([frame1, frame2])
        out.write(concat_frame)

    cap1.release()
    cap2.release()
    out.release()
    print(f"Saved concatenated video to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare.py video1.mp4 video2.mp4 output.mp4")
    else:
        concat_videos(sys.argv[1], sys.argv[2], sys.argv[3])