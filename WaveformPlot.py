import matplotlib.pyplot as plt
import time

LABEL_COLORS = {2: ('green', 0.25), 1: ('orange', 0.25), 0: ('gray', 0.15)}

class WaveformPlot():
    def draw(res):
        sig = res["signal_preprocessed"]
        t_sig = res["time_axis"]
        t_fr = res["frame_times"]
        ste = res["ste"]
        zcr = res["zcr"]
        r_max = res["r_max"]
        sf = res["spectral_flatness"]
        labels = res["labels"]
        T_E = res["T_E"]
        T_ZCR = res["T_ZCR"]
        T_R = res["T_R"]

        # Đồ thị
        fig, axs = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
        fig.suptitle("Phân tích Voice / Unvoice / Silence", fontsize=14, fontweight='bold')

        # 1) Waveform
        axs[0].plot(t_sig, sig, color='#2563EB', linewidth=0.6, alpha=0.85)
        axs[0].set_title("1. Tín hiệu tiếng nói (Waveform)")
        axs[0].set_ylabel("Biên độ")
        axs[0].grid(True, alpha=0.4)

        # 2) STE
        axs[1].plot(t_fr, ste, color='#DC2626', linewidth=1.2, label='STE')
        axs[1].axhline(T_E, color='black', linestyle='--', linewidth=1.0,
                       label=f'T_E = {T_E:.5f}')
        axs[1].set_title("2. Short-Time Energy (STE) + Ngưỡng")
        axs[1].set_ylabel("Năng lượng")
        axs[1].legend(loc='upper right', fontsize=8)
        axs[1].grid(True, alpha=0.4)

        # 3) ZCR
        axs[2].plot(t_fr, zcr, color='#16A34A', linewidth=1.2, label='ZCR')
        axs[2].axhline(T_ZCR, color='black', linestyle='--', linewidth=1.0,
                       label=f'T_ZCR = {T_ZCR:.3f}')
        axs[2].set_title("3. Zero-Crossing Rate (ZCR) + Ngưỡng")
        axs[2].set_ylabel("Tỷ lệ ZCR")
        axs[2].legend(loc='upper right', fontsize=8)
        axs[2].grid(True, alpha=0.4)

        # 4) ACF R_max
        axs[3].plot(t_fr, r_max, color='#7C3AED', linewidth=1.2, label='ACF R_max')
        axs[3].axhline(T_R, color='black', linestyle='--', linewidth=1.0,
                       label=f'T_R = {T_R:.2f}')
        axs[3].set_title("4. Đỉnh Tự Tương Quan Chuẩn Hóa (ACF R_max) + Ngưỡng")
        axs[3].set_ylabel("R_max")
        axs[3].set_ylim(-0.1, 1.05)
        axs[3].legend(loc='upper right', fontsize=8)
        axs[3].grid(True, alpha=0.4)

        # 5) Kết quả phân loại
        label_map = {0: 'Silence', 1: 'Unvoice', 2: 'Voice'}
        color_map = {0: 'gray', 1: 'orange', 2: 'green'}
        for lbl, name in label_map.items():
            axs[4].fill_between(t_fr, 0, labels,
                                where=(labels == lbl),
                                color=color_map[lbl], alpha=0.5, label=name)
        axs[4].step(t_fr, labels, where='mid', color='black', linewidth=1.0)
        axs[4].set_title("5. Kết quả phân loại (0=Silence | 1=Unvoice | 2=Voice)")
        axs[4].set_yticks([0, 1, 2])
        axs[4].set_yticklabels(['Silence', 'Unvoice', 'Voice'])
        axs[4].set_xlabel("Thời gian (giây)")
        axs[4].legend(loc='upper right', fontsize=8)
        axs[4].grid(True, alpha=0.4)

        plt.tight_layout()
        filename = f"fig_graph-{time.time()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
