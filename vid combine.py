import cv2
import os

# ==========================
# Konfigurasi
# ==========================
input_folder = r"/Users/fayyazi/programing/sem4/pcd/fp2/tessss/frames3-enhanced"
output_video = r"/Users/fayyazi/programing/sem4/pcd/fp2/tessss/hasil_video.mp4"

# Gunakan FPS yang sama dengan video asli
fps = 30

# ==========================
# Ambil daftar gambar
# ==========================
images = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(images) == 0:
    print("Tidak ada gambar ditemukan!")
    exit()

# ==========================
# Ambil ukuran gambar pertama
# ==========================
first_image_path = os.path.join(input_folder, images[0])
frame = cv2.imread(first_image_path)

height, width, _ = frame.shape

print(f"Jumlah gambar : {len(images)}")
print(f"Resolusi      : {width} x {height}")

# ==========================
# Buat VideoWriter
# ==========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# ==========================
# Masukkan semua gambar
# ==========================
for i, image_name in enumerate(images):
    image_path = os.path.join(input_folder, image_name)
    frame = cv2.imread(image_path)

    video.write(frame)

    if i % 100 == 0:
        print(f"Memproses {i}/{len(images)}")

video.release()

print("\nSelesai!")
print("Video tersimpan di:")
print(output_video)