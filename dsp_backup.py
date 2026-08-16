"""
DSP Module: Phát hiện Voice / Unvoice / Silence trong tín hiệu tiếng nói

1. Tiền xử lý (chuẩn hóa, lọc DC)
2. Phân khung + Hamming window
3. Trích xuất đặc trưng: STE, ZCR, ACF R_max, Spectral Flatness, Centroid, Peakiness
4. Phân loại ngưỡng thích ứng => 0 (Silence) / 1 (Unvoice) / 2 (Voice)
5. Hậu xử lý: Median filter làm mượt nhãn
"""

import numpy as np
from scipy.signal import butter, lfilter

# THAM SỐ CẤU HÌNH THUẬT TOÁN (ALGORITHM CONFIGURATION PARAMETERS)

# 1. Cấu hình Tiền xử lý & Phân khung (Pre-processing & Framing)
HIGH_PASS_CUTOFF_HZ = 80.0     # Tần số cắt cho bộ lọc thông cao loại DC (Hz)
HIGH_PASS_FILTER_ORDER = 4     # Bậc của bộ lọc High-pass Butterworth
FRAME_SIZE_MS = 25.0           # Độ dài mỗi khung tín hiệu (ms)
FRAME_STRIDE_MS = 10.0         # Bước dịch giữa các khung (ms)

# 2. Cấu hình Autocorrelation (ACF - Pitch Detection Range)
F0_MIN_HZ = 50.0               # Tần số cơ bản Pitch tối thiểu của giọng nói (Hz)
F0_MAX_HZ = 400.0              # Tần số cơ bản Pitch tối đa của giọng nói (Hz)

# 3. Các Ngưỡng & Tham Số Phân Loại (Classification Thresholds & Factors)
DEFAULT_NOISE_FRAME_COUNT = 20 # Số khung đầu tiên dùng để ước lượng nhiễu nền

# Short-Time Energy (STE)
STE_NOISE_MULTIPLIER = 3.0     # Hệ số K cho T_E = noise_mean + K * noise_std
STE_FLOOR_OFFSET = 1e-4        # Sàn tối thiểu cộng thêm cho T_E

# Zero-Crossing Rate (ZCR)
ZCR_NOISE_MULTIPLIER = 2.0     # Hệ số cho T_ZCR dựa trên nhiễu nền
ZCR_FLOOR_DEFAULT = 0.18       # Ngưỡng ZCR sàn tối thiểu (Default Floor)
ZCR_NOISE_FACTOR = 1.15        # Hệ số nhân T_ZCR để phát hiện tín hiệu giống nhiễu (Unvoice)

# Autocorrelation Peak (ACF R_max)
T_R_DEFAULT = 0.30             # Ngưỡng R_max tối thiểu khẳng định tính chu kỳ (Voice)

# Spectral Flatness (SF)
SF_NOISE_MULTIPLIER = 2.0      # Hệ số cho T_SF dựa trên nhiễu nền
SF_FLOOR_DEFAULT = 0.25        # Ngưỡng Spectral Flatness sàn tối thiểu (Unvoice có SF cao)

# Spectral Centroid & Peakiness
T_C_DEFAULT = 3500.0           # Ngưỡng Spectral Centroid tối đa cho Voice (Hz)
T_P_DEFAULT = 150.0            # Ngưỡng Spectral Peakiness tối đa cho Voice

# Hậu xử lý (Post-processing)
MEDIAN_FILTER_SIZE = 7         # Kích thước cửa sổ lọc trung vị (Median filter) làm mượt nhãn

# 1. TIỀN XỬ LÝ (PREPROCESSING)
def preprocess(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = HIGH_PASS_CUTOFF_HZ,
    filter_order: int = HIGH_PASS_FILTER_ORDER
) -> np.ndarray:
    """
    Tiền xử lý tín hiệu:
      - Chuyển về float64
      - Chuẩn hóa biên độ về [-1, 1]
      - Lọc thông cao (High-pass) để loại DC offset và nhiễu tần thấp

    Tham số:
        signal       : Mảng tín hiệu âm thanh 1 chiều
        sample_rate  : Tần số lấy mẫu (Hz)
        cutoff_hz    : Tần số cắt High-pass (Hz)
        filter_order : Bậc bộ lọc Butterworth

    Trả về:
        Mảng tín hiệu đã tiền xử lý (float64)
    """
    sig = signal.astype(np.float64)

    # Chuẩn hóa biên độ
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val

    # Thiết kế bộ lọc thông cao Butterworth
    nyquist = sample_rate / 2.0
    fc = cutoff_hz / nyquist  # Tần số cắt chuẩn hóa
    b, a = butter(filter_order, fc, btype='high')
    sig = lfilter(b, a, sig)

    return sig


# 2. PHÂN KHUNG VÀ HAMMING WINDOW (FRAMING + WINDOWING)
def create_frames(
    signal: np.ndarray,
    sample_rate: int,
    frame_size_ms: float = FRAME_SIZE_MS,
    frame_stride_ms: float = FRAME_STRIDE_MS
):
    """
    Chia tín hiệu thành các khung ngắn có chồng lấp và nhân cửa sổ Hamming.

    Tham số:
        signal         : Tín hiệu 1 chiều (float)
        sample_rate    : Tần số lấy mẫu
        frame_size_ms  : Độ dài khung (ms), mặc định 25 ms
        frame_stride_ms: Bước dịch khung (ms), mặc định 10 ms (overlap 60%)

    Trả về:
        frames       : Ma trận (num_frames × frame_length) đã nhân Hamming
        frame_length : Số mẫu mỗi khung
        frame_step   : Số mẫu mỗi bước dịch
    """
    frame_length = int(round(frame_size_ms * 1e-3 * sample_rate))
    frame_step   = int(round(frame_stride_ms * 1e-3 * sample_rate))
    signal_length = len(signal)

    # Số khung cần thiết
    num_frames = int(np.ceil(float(max(signal_length - frame_length, 0)) / frame_step)) + 1

    # Padding bằng 0 để khung cuối đầy đủ
    pad_length = (num_frames - 1) * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(max(pad_length - signal_length, 0)))

    # Tạo ma trận chỉ số rồi index vào mảng
    row_idx = np.tile(np.arange(frame_length), (num_frames, 1))
    col_idx = np.tile(np.arange(num_frames) * frame_step, (frame_length, 1)).T
    indices = (row_idx + col_idx).astype(np.int32)

    frames = pad_signal[indices]

    # Nhân cửa sổ Hamming — giảm rò rỉ phổ (Spectral Leakage)
    frames *= np.hamming(frame_length)

    return frames, frame_length, frame_step


# 3. TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)
def compute_ste(frames: np.ndarray) -> np.ndarray:
    """
    Short-Time Energy (STE) — Năng lượng ngắn hạn.
      E[i] = Sigma(x[n]^2)

    Voice   → E lớn (dây thanh quản rung mạnh)
    Unvoice → E vừa phải
    Silence → E ≈ 0
    """
    return np.sum(frames ** 2, axis=1)


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """
    Zero-Crossing Rate (ZCR) — Tốc độ qua điểm 0.
      ZCR[i] = Σ|sign(x[n]) − sign(x[n−1])| / (2N)

    Voice   → ZCR thấp (dao động chậm, có chu kỳ)
    Unvoice → ZCR cao  (dao động nhanh như nhiễu)
    """
    signs = np.sign(frames)
    signs[signs == 0] = 1  # tránh 0 gây lỗi khi tính hiệu

    sign_changes = np.abs(np.diff(signs, axis=1))
    zcr = np.sum(sign_changes, axis=1) / (2.0 * frames.shape[1])
    return zcr


def compute_acf_max(
    frames: np.ndarray,
    sample_rate: int,
    f0_min: float = F0_MIN_HZ,
    f0_max: float = F0_MAX_HZ
) -> np.ndarray:
    """
    Đỉnh hàm tự tương quan chuẩn hóa (R_max) — Mức độ có tính chu kỳ.

    Tính autocorrelation bằng FFT (nhanh hơn tính trực tiếp O(N^2)):
      R = IFFT(|FFT(x)|^2)
    Sau đó tìm đỉnh max trong dải lag tương ứng với f0 ∈ [f0_min, f0_max].

    Voice   → R_max cao (đỉnh rõ, chu kỳ pitch ổn định)
    Unvoice → R_max thấp (không có tính chu kỳ)
    """
    num_frames, frame_length = frames.shape
    r_max = np.zeros(num_frames)

    # Quy đổi tần số Pitch sang khoảng lag (số mẫu)
    k_min = max(int(sample_rate / f0_max), 1)
    k_max = min(int(sample_rate / f0_min), frame_length - 1)

    if k_max <= k_min:
        return r_max  # Không đủ dải → trả về 0

    # Kích thước FFT tối thiểu để tránh circular aliasing
    n_fft = int(2 ** np.ceil(np.log2(2 * frame_length - 1)))

    for i in range(num_frames):
        frame = frames[i]
        energy = np.sum(frame ** 2)
        if energy < 1e-10:
            continue  # Khung lặng, bỏ qua

        fft_frame = np.fft.rfft(frame, n=n_fft)
        power = np.abs(fft_frame) ** 2
        autocorr = np.fft.irfft(power)[:frame_length]

        # Chuẩn hóa: R[0] = 1
        autocorr_norm = autocorr / (autocorr[0] + 1e-10)
        r_max[i] = np.max(autocorr_norm[k_min:k_max])

    return r_max


def compute_spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """
    Spectral Flatness (Wiener Entropy) — Độ phẳng phổ.
      SF = geometric_mean(|X[k]|^2) / arithmetic_mean(|X[k]|^2)

    Voice   → SF thấp  (phổ có đỉnh rõ tại harmonics)
    Unvoice → SF cao   (phổ phẳng giống nhiễu trắng)
    """
    n_fft = frames.shape[1]
    power_spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 + 1e-12  # tránh log(0)

    log_mean = np.mean(np.log(power_spectrum), axis=1)
    arith_mean = np.mean(power_spectrum, axis=1)

    # SF = exp(log-mean) / arith-mean  (tương đương geometric/arithmetic mean)
    sf = np.exp(log_mean) / (arith_mean + 1e-12)
    return sf


def compute_spectral_centroid(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Spectral Centroid — trung tâm tần số của phổ từng khung.
    Voice   → thường tập trung ở dải tần thấp hơn so với âm nhạc phức tạp.
    Unvoice → phổ dịch về dải tần cao/đặc trưng riêng hơn.
    """
    n_fft = frames.shape[1]
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    magnitude = spectrum + 1e-12

    centroid = np.sum(freqs[None, :] * magnitude, axis=1) / np.sum(magnitude, axis=1)
    return centroid


def compute_spectral_peakiness(frames: np.ndarray) -> np.ndarray:
    """
    Spectral Peakiness — mức độ phổ bị tập trung vào một vài bin.
    Pure-tone/music-like signals có peakiness rất cao.
    Speech voiced thường có peakiness thấp hơn vì phổ có nhiều harmonic/formant.
    """
    n_fft = frames.shape[1]
    power_spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2 + 1e-12
    peak_energy = np.max(power_spectrum, axis=1)
    mean_energy = np.mean(power_spectrum, axis=1)
    return peak_energy / mean_energy


# 4. PHÂN LOẠI VOICE/UNVOICE CLASSIFICATION
def classify_frames(
    ste: np.ndarray,
    zcr: np.ndarray,
    r_max: np.ndarray,
    spectral_flatness: np.ndarray,
    spectral_centroid: np.ndarray,
    spectral_peakiness: np.ndarray,
    noise_frame_count: int = DEFAULT_NOISE_FRAME_COUNT
):
    """
    Phân loại từng khung thành Silence (0), Unvoice (1), Voice (2).

    Thuật toán: Dual-Threshold thích ứng dựa trên STE + ZCR + ACF R_max.

    Bước 1 — Ước lượng nhiễu nền từ `noise_frame_count` khung đầu tiên
             (giả định các khung này là nhiễu nền / khoảng lặng trước khi nói).

    Bước 2 — Xác định ngưỡng thích ứng từ các cấu hình mặc định.

    Bước 3 — Phân loại sơ bộ:
             Silence  ← STE < T_E
             Voice    ← STE ≥ T_E VÀ có tính chu kỳ rõ (R_max ≥ T_R VÀ ZCR < T_ZCR)
                         VÀ phổ có đặc trưng speech-like (SF thấp, centroid thấp, peakiness vừa phải)
             Unvoice  ← còn lại

    Bước 4 — Làm mượt nhãn bằng Median Filter (loại nhiễu phân loại đơn lẻ).
    """
    n = noise_frame_count

    # Bước 1: Ước lượng nhiễu nền
    noise_ste_mean = np.mean(ste[:n])
    noise_ste_std  = np.std(ste[:n])

    noise_zcr_mean = np.mean(zcr[:n])
    noise_zcr_std  = np.std(zcr[:n])
    noise_sf_mean  = np.mean(spectral_flatness[:n])
    noise_sf_std   = np.std(spectral_flatness[:n])

    # Bước 2: Ngưỡng thích ứng từ hằng số cấu hình
    T_E   = noise_ste_mean + STE_NOISE_MULTIPLIER * noise_ste_std + STE_FLOOR_OFFSET
    T_ZCR = max(noise_zcr_mean + ZCR_NOISE_MULTIPLIER * noise_zcr_std, ZCR_FLOOR_DEFAULT)
    T_R   = T_R_DEFAULT
    T_SF  = max(noise_sf_mean + SF_NOISE_MULTIPLIER * noise_sf_std, SF_FLOOR_DEFAULT)
    T_C   = T_C_DEFAULT
    T_P   = T_P_DEFAULT

    # Bước 3: Phân loại sơ bộ
    labels = np.zeros(len(ste), dtype=int)  # mặc định: Silence

    for i in range(len(ste)):
        if ste[i] < T_E:
            labels[i] = 0  # Silence
        else:
            is_periodic = (r_max[i] >= T_R) and (zcr[i] < T_ZCR)
            is_noise_like = (zcr[i] >= T_ZCR * ZCR_NOISE_FACTOR) or (spectral_flatness[i] >= T_SF)
            is_speech_like = (
                is_periodic
                and not is_noise_like
                and (spectral_centroid[i] < T_C)
                and (spectral_peakiness[i] < T_P)
            )

            if is_speech_like:
                labels[i] = 2  # Voice
            else:
                labels[i] = 1  # Unvoice (bao gồm tiếng vỗ tay, nhiễu, âm nhạc và âm vô thanh)

    # Bước 4: Làm mượt với Median Filter
    smoothed_labels = median_smoothing(labels, size=MEDIAN_FILTER_SIZE)

    return smoothed_labels, T_E, T_ZCR, T_R, T_SF, T_C, T_P


def median_smoothing(labels: np.ndarray, size: int = MEDIAN_FILTER_SIZE) -> np.ndarray:
    """Smooth labels with a median filter using NumPy only."""
    pad = size // 2
    padded = np.pad(labels, pad, mode="edge")
    smoothed = np.empty_like(labels)
    for i in range(len(labels)):
        smoothed[i] = int(np.median(padded[i : i + size]))
    return smoothed


# 5. PIPELINE HOÀN CHỈNH
def run_analysis(
    signal: np.ndarray,
    sample_rate: int,
    noise_frame_count: int = DEFAULT_NOISE_FRAME_COUNT,
    frame_size_ms: float = FRAME_SIZE_MS,
    frame_stride_ms: float = FRAME_STRIDE_MS
) -> dict:
    """
    Chạy toàn bộ pipeline phân tích từ đầu đến cuối.
    """
    # 1. Tiền xử lý
    sig = preprocess(signal, sample_rate)

    # 2. Phân khung
    frames, frame_length, frame_step = create_frames(
        sig, sample_rate,
        frame_size_ms=frame_size_ms,
        frame_stride_ms=frame_stride_ms
    )

    # 3. Trích xuất đặc trưng
    ste = compute_ste(frames)
    zcr = compute_zcr(frames)
    r_max = compute_acf_max(frames, sample_rate)
    sf  = compute_spectral_flatness(frames)
    spectral_centroid = compute_spectral_centroid(frames, sample_rate)
    spectral_peakiness = compute_spectral_peakiness(frames)

    # 4. Phân loại
    labels, T_E, T_ZCR, T_R, T_SF, T_C, T_P = classify_frames(
        ste, zcr, r_max, sf, spectral_centroid, spectral_peakiness,
        noise_frame_count=noise_frame_count
    )

    # 5. Trục thời gian
    num_frames = len(frames)
    frame_times = (np.arange(num_frames) * frame_step + frame_length / 2.0) / sample_rate
    time_axis = np.arange(len(sig)) / sample_rate

    return {
        "signal_preprocessed": sig,
        "frames": frames,
        "frame_length": frame_length,
        "frame_step": frame_step,
        "frame_times": frame_times,
        "time_axis": time_axis,
        "ste": ste,
        "zcr": zcr,
        "r_max": r_max,
        "spectral_flatness": sf,
        "labels": labels,
        "T_E": T_E,
        "T_ZCR": T_ZCR,
        "T_R": T_R,
        "T_SF": T_SF,
        "T_C": T_C,
        "T_P": T_P,
    }

