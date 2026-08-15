import os
import wave
import threading
from dsp import run_analysis
import scipy.io.wavfile as wav
import numpy as np
from datetime import datetime

from kivy.app import App
from kivy.core.audio import SoundLoader
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.button import Button


try:
    import sounddevice as sd
    import soundfile as sf

    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False

Builder.load_file("main.kv")

class MainLayout(BoxLayout):

    selected_file = None
    sound = None
    is_recording = False
    recorded_frames = []
    sample_rate = 44100
    channels = 1

    def choose_file(self):
        file_chooser = FileChooserIconView(
            filters=["*.wav", "*.mp3", "*.ogg", "*.flac"],
            path=os.getcwd()
        )

        select_button = Button(
            text="Chọn file này",
            size_hint_y=None,
            height=50
        )

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(file_chooser)
        layout.add_widget(select_button)

        popup = Popup(
            title="Chọn file âm thanh",
            content=layout,
            size_hint=(0.9, 0.9)
        )

        def select_audio_file(_instance):
            if file_chooser.selection:
                self.selected_file = file_chooser.selection[0]
                self.ids.file_label.text = f"File: {os.path.basename(self.selected_file)}"
                self.ids.status_label.text = "Đã chọn file âm thanh."
                popup.dismiss()
            else:
                self.ids.status_label.text = "Vui lòng chọn một file."

        select_button.bind(on_press=select_audio_file)
        popup.open()

    def toggle_recording(self):
        if not RECORDING_AVAILABLE:
            self.ids.status_label.text = (
                "Chưa thể thu âm. Cần cài thêm thư viện sounddevice và soundfile."
            )
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.recorded_frames = []

        self.ids.record_button.text = "Dừng thu âm"
        self.ids.status_label.text = "Đang thu âm..."

        thread = threading.Thread(target=self.record_audio, daemon=True)
        thread.start()

    def record_audio(self):
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback
            ):
                while self.is_recording:
                    sd.sleep(100)
        except Exception as error:
            self.is_recording = False
            self.ids.record_button.text = "Thu âm"
            self.ids.status_label.text = f"Lỗi thu âm: {error}"

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        self.recorded_frames.append(indata.copy())

    def stop_recording(self):
        self.is_recording = False
        self.ids.record_button.text = "Thu âm"

        if not self.recorded_frames:
            self.ids.status_label.text = "Không có dữ liệu thu âm."
            return

        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.wav")
        output_path = os.path.join(os.getcwd(), filename)

        try:
            import numpy as np

            audio_data = np.concatenate(self.recorded_frames, axis=0)
            sf.write(output_path, audio_data, self.sample_rate)

            self.selected_file = output_path
            self.ids.file_label.text = f"File: {os.path.basename(self.selected_file)}"
            self.ids.status_label.text = "Đã lưu file thu âm."
        except Exception as error:
            self.ids.status_label.text = f"Lỗi lưu file thu âm: {error}"

    def play_audio(self):
        if not self.selected_file:
            self.ids.status_label.text = "Chưa có file để phát."
            return

        if self.sound:
            self.sound.stop()

        self.sound = SoundLoader.load(self.selected_file)

        if not self.sound:
            self.ids.status_label.text = "Không thể phát file này."
            return

        self.sound.play()
        self.ids.status_label.text = "Đang phát âm thanh..."

    def stop_audio(self):
        if self.sound:
            self.sound.stop()
            self.ids.status_label.text = "Đã dừng phát âm thanh."
        else:
            self.ids.status_label.text = "Không có âm thanh đang phát."

    def analyze_audio(self):
        if not self.selected_file:
            self.ids.status_label.text = "Chưa có file để phân tích."
            return

        wav_path = self.selected_file
        file_extension = os.path.splitext(self.selected_file)[1].lower()
        analysis_text = []
        if file_extension == ".wav":
            try:
                sr, raw = wav.read(wav_path)
                raw = raw.astype(np.float64)
                if raw.ndim > 1:
                    raw = raw[:, 0]  # Stereo → Mono

                analysis_text = [
                    f"Tần số : {sr:,} Hz",
                    f"Độ dài : {len(raw) / sr:.2f} giây ({len(raw):,} mẫu)"
                ]

                res = run_analysis(raw, sr)
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

                # Thống kê
                total_frames = len(labels)
                p_voice = np.sum(labels == 2) / total_frames * 100
                p_unvoice = np.sum(labels == 1) / total_frames * 100
                p_silence = np.sum(labels == 0) / total_frames * 100
                analysis_text.append(f"\n Kết quả phân loại ({total_frames} khung):")
                analysis_text.append(f"   Voice   : {p_voice:.1f}%")
                analysis_text.append(f"   Unvoice : {p_unvoice:.1f}%")
                analysis_text.append(f"   Silence : {p_silence:.1f}%")
                analysis_text.append("\n Ngưỡng thích ứng:")
                analysis_text.append(f"   T_E   = {T_E:.6f}")
                analysis_text.append(f"   T_ZCR = {T_ZCR:.4f}")
                analysis_text.append(f"   T_R   = {T_R:.2f}")
                analysis_text.append(f"   T_SF  = {res['T_SF']:.4f}")
                analysis_text.append(f"   T_C   = {res['T_C']:.1f} Hz")
                analysis_text.append(f"   T_P   = {res['T_P']:.1f}")

            except Exception as error:
                analysis_text.append(f"Lỗi phân tích WAV: {error}")
        else:
            analysis_text.append(
                "Hiện tại ứng dụng phân tích chi tiết tốt nhất với file WAV."
            )

        self.ids.analysis_label.text = "\n".join(analysis_text)
        self.ids.status_label.text = "Đã phân tích file âm thanh."


class KivyApp(App):

    def build(self):
        return MainLayout()


if __name__ == "__main__":
    KivyApp().run()