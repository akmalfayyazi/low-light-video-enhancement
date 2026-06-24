import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import differential_evolution
from skimage.metrics import structural_similarity as ssim

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Batas attempt
# ─────────────────────────────────────────────
MAX_ATTEMPTS = 100

# ─────────────────────────────────────────────
# Global state untuk tracking progress
# ─────────────────────────────────────────────
_best_ssim   = -np.inf
_best_params = None
_iteration   = 0
_pbar        = None


# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def apply_enhancement(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation_scale: float,
) -> np.ndarray:
    """
    Menerapkan pipeline image enhancement (Brightness + Contrast + Saturation).

    Parameters
    ----------
    image            : BGR uint8 image (H x W x 3)
    brightness       : nilai tambah brightness, range -127 hingga +127
                       (negatif = gelap, positif = terang)
    contrast         : faktor pengali contrast, range 0.0 – 3.0
                       (< 1.0 = kontras rendah, > 1.0 = kontras tinggi)
    saturation_scale : faktor pengali saturasi HSV, range 0.0 – 3.0
                       (0.0 = grayscale, > 1.0 = lebih jenuh)

    Returns
    -------
    np.ndarray : BGR uint8 image hasil enhancement
    """
    img = image.astype(np.float32)

    # ── 1. Brightness & Contrast Adjustment ─────────────────────
    # Formula: out = contrast * in + brightness
    # Penyesuaian dilakukan di ruang BGR secara langsung.
    img = contrast * img + brightness

    # Clamp ke [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # ── 2. Saturation Adjustment (HSV) ──────────────────────────
    # Skala channel S pada ruang HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


class AttemptLimitReached(Exception):
    """Exception untuk menghentikan DE saat batas attempt tercapai."""
    pass


def objective_function(
    params: np.ndarray,
    dark_img: np.ndarray,
    target_img: np.ndarray,
) -> float:
    """
    Fungsi objektif untuk Differential Evolution.

    Menghitung (1 - SSIM) sehingga minimisasi DE sama dengan
    memaksimalkan SSIM terhadap gambar target.
    Berhenti otomatis setelah MAX_ATTEMPTS evaluasi.

    Parameters
    ----------
    params     : array [brightness, contrast, saturation_scale]
    dark_img   : BGR uint8 gambar gelap (input)
    target_img : BGR uint8 gambar terang (ground truth)

    Returns
    -------
    float : 1 - SSIM  (semakin kecil = semakin baik)
    """
    global _best_ssim, _best_params, _iteration, _pbar

    # ── Cek batas attempt ────────────────────────────────────────
    if _iteration >= MAX_ATTEMPTS:
        raise AttemptLimitReached(
            f"Batas {MAX_ATTEMPTS} attempt telah tercapai."
        )

    brightness, contrast, saturation_scale = params

    # Terapkan enhancement
    enhanced = apply_enhancement(
        dark_img, brightness, contrast, saturation_scale
    )

    # Hitung SSIM (skimage mengharapkan gambar dengan channel_axis)
    score = ssim(
        target_img,
        enhanced,
        channel_axis=2,   # gambar BGR (H x W x C)
        data_range=255,
    )

    # Update iterasi & progress bar
    _iteration += 1
    if _pbar is not None:
        _pbar.update(1)

    # Simpan & logging bila ditemukan solusi lebih baik
    if score > _best_ssim:
        _best_ssim   = score
        _best_params = params.copy()
        logger.info(
            "✨ Solusi baru ditemukan!\n"
            "   Attempt          : %d / %d\n"
            "   brightness       : %.4f\n"
            "   contrast         : %.4f\n"
            "   saturation_scale : %.4f\n"
            "   SSIM             : %.6f",
            _iteration, MAX_ATTEMPTS,
            brightness, contrast, saturation_scale, score,
        )

    return 1.0 - score


def plot_histograms(
    dark_img: np.ndarray,
    target_img: np.ndarray,
    enhanced_img: np.ndarray,
) -> None:
    """Menampilkan histogram perbandingan ketiga gambar."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Perbandingan Histogram (B / G / R)", fontsize=14, fontweight="bold")

    titles = ["Dark Image", "Target Image", "Best Enhanced"]
    images = [dark_img, target_img, enhanced_img]
    colors = ["blue", "green", "red"]

    for row, (img, title) in enumerate(zip(images, titles)):
        for col, (ch_idx, color) in enumerate(zip([0, 1, 2], colors)):
            ax = axes[row][col]
            hist = cv2.calcHist([img], [ch_idx], None, [256], [0, 256])
            ax.plot(hist, color=color, linewidth=0.8)
            ax.fill_between(range(256), hist.ravel(), alpha=0.3, color=color)
            ax.set_xlim([0, 255])
            ax.set_title(f"{title} – {'BGR'[col]}", fontsize=9)
            ax.set_xlabel("Intensitas Piksel")
            ax.set_ylabel("Frekuensi")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("histograms.png", dpi=150, bbox_inches="tight")
    logger.info("Histogram disimpan sebagai histograms.png")


def show_comparison(
    dark_img: np.ndarray,
    target_img: np.ndarray,
    enhanced_img: np.ndarray,
    best_ssim_val: float,
) -> None:
    """Menampilkan perbandingan visual tiga gambar."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Hasil Optimasi Enhancement  |  SSIM terbaik: {best_ssim_val:.4f}",
        fontsize=13,
    )

    pairs = [
        (dark_img,     "Dark Image (Input)"),
        (target_img,   "Target Image (Ground Truth)"),
        (enhanced_img, f"Best Enhanced  (SSIM = {best_ssim_val:.4f})"),
    ]

    for ax, (img, title) in zip(axes, pairs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    logger.info("Gambar perbandingan disimpan sebagai comparison.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    global _pbar

    # ── Input path ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  IMAGE ENHANCEMENT OPTIMIZER")
    print(f"  Batas Attempt    : {MAX_ATTEMPTS}")
    print(f"  Metode           : Brightness + Contrast + Saturation")
    print("=" * 55)

    dark_path   = Path(r"C:\Users\asuss\Documents\semester 4\PCD\FP\try bruteforce\dark.png")
    target_path = Path(r"C:\Users\asuss\Documents\semester 4\PCD\FP\try bruteforce\target.png")

    for p in [dark_path, target_path]:
        if not p.exists():
            logger.error("File tidak ditemukan: %s", p)
            sys.exit(1)

    # ── Load gambar ──────────────────────────────────────────────
    dark_img   = cv2.imread(str(dark_path))
    target_img = cv2.imread(str(target_path))

    if dark_img is None or target_img is None:
        logger.error("Gagal membaca gambar. Pastikan file valid.")
        sys.exit(1)

    # Resize target ke ukuran dark agar dimensi cocok
    if dark_img.shape != target_img.shape:
        logger.info(
            "Ukuran gambar berbeda. Meresize target %s → %s",
            target_img.shape[:2], dark_img.shape[:2],
        )
        target_img = cv2.resize(
            target_img,
            (dark_img.shape[1], dark_img.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    logger.info("Gambar berhasil dimuat: %s dan %s", dark_path, target_path)
    logger.info("Ukuran gambar: %s", dark_img.shape)

    # ── Batas parameter (bounds) ─────────────────────────────────
    # 3 parameter: brightness, contrast, saturation_scale
    bounds = [
        (-127.0, 127.0),  # brightness  (offset piksel)
        (0.0,    3.0),    # contrast    (faktor pengali)
        (0.0,    3.0),    # saturation_scale
    ]

    # popsize dan maxiter disesuaikan agar total evaluasi ≤ MAX_ATTEMPTS
    # Total evaluasi DE ≈ popsize * n_params * maxiter
    # Dengan n_params=3, popsize=5 → 5*3=15 per generasi → 250/15 ≈ 16 generasi
    popsize  = 5
    n_params = len(bounds)
    maxiter  = max(1, MAX_ATTEMPTS // (popsize * n_params))

    logger.info(
        "Konfigurasi DE\n"
        "   popsize  : %d | maxiter : %d\n"
        "   Estimasi evaluasi : ~%d (dibatasi %d)",
        popsize, maxiter,
        popsize * n_params * (maxiter + 1),
        MAX_ATTEMPTS,
    )

    # ── Progress bar ─────────────────────────────────────────────
    _pbar = tqdm(
        total=MAX_ATTEMPTS,
        desc="Optimasi DE",
        unit="attempt",
        ncols=80,
        colour="cyan",
    )

    # ── Optimasi ────────────────────────────────────────────────
    result = None
    try:
        result = differential_evolution(
            func=objective_function,
            bounds=bounds,
            args=(dark_img, target_img),
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=False,         # polish dimatikan agar attempt terkontrol
            init="latinhypercube",
            workers=1,
            disp=False,
        )
    except AttemptLimitReached as e:
        logger.info("🛑 %s", e)
    finally:
        _pbar.close()

    # ── Tentukan parameter terbaik ───────────────────────────────
    # Utamakan hasil dari scipy result (bila selesai normal),
    # fallback ke _best_params yang dicatat selama iterasi.
    if result is not None and (1.0 - result.fun) >= _best_ssim:
        best_params_arr  = result.x
        best_ssim_value  = 1.0 - result.fun
    else:
        best_params_arr  = _best_params
        best_ssim_value  = _best_ssim

    if best_params_arr is None:
        logger.error("Tidak ada hasil evaluasi yang berhasil.")
        sys.exit(1)

    brightness, contrast, sat_scale = best_params_arr

    # ── Cetak hasil terbaik ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  HASIL TERBAIK SETELAH {} ATTEMPT".format(_iteration))
    print("=" * 55)
    print(f"  brightness       : {brightness:.4f}")
    print(f"  contrast         : {contrast:.4f}")
    print(f"  saturation_scale : {sat_scale:.4f}")
    print("-" * 55)
    print(f"  SSIM Terbaik     : {best_ssim_value:.6f}")
    print("=" * 55 + "\n")

    # ── Generate & simpan hasil terbaik ─────────────────────────
    best_enhanced = apply_enhancement(
        dark_img, brightness, contrast, sat_scale
    )

    output_path = "best_result.jpg"
    cv2.imwrite(output_path, best_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info("Hasil terbaik disimpan sebagai %s", output_path)

    # ── Visualisasi ──────────────────────────────────────────────
    show_comparison(dark_img, target_img, best_enhanced, best_ssim_value)
    plot_histograms(dark_img, target_img, best_enhanced)

    plt.figure(figsize=(15, 10))
    img_hist = cv2.imread("histograms.png")
    plt.imshow(cv2.cvtColor(img_hist, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    logger.info("Selesai! Semua output tersimpan.")


if __name__ == "__main__":
    main()