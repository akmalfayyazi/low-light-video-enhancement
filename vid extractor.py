import cv2
import os

# Path video
video_path = r"/Users/fayyazi/programing/sem4/pcd/fp2/tessss/IMG_1993.MOV"

# Folder output
output_folder = r"/Users/fayyazi/programing/sem4/pcd/fp2/tessss/frames3"
os.makedirs(output_folder, exist_ok=True)

# Buka video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video gagal dibuka!")
    exit()

# Informasi video
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS          : {fps}")
print(f"Total Frames : {total_frames}")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    filename = os.path.join(
        output_folder,
        f"frame_{frame_count:06d}.png"
    )

    cv2.imwrite(filename, frame)

    if frame_count % 100 == 0:
        print(f"Menyimpan frame {frame_count}/{total_frames}")

    frame_count += 1

cap.release()

print(f"\nSelesai!")
print(f"Total frame tersimpan: {frame_count}")