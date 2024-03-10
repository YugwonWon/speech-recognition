
import glob
import os
import json
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from praat import Praat
from parselmouth.praat import call


class SpeechAnalysis(Praat):
    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.sound = parselmouth.Sound(wav_path)
        self.base_name = os.path.splitext(os.path.basename(self.wav_path))[0]
        self.intensity = self.get_intensity()
        # 'out/jpg' 디렉터리가 없으면 생성합니다.
        self.jpg_dir = 'out/jpg'
        if not os.path.exists(self.jpg_dir):
            os.makedirs(self.jpg_dir)

        # 초기화 시에 피치, 포먼트, 말하기 속도를 계산합니다.
        self.pitch = self.calculate_pitch()
        self.formants = self.calculate_formants()
        self.speech_rate = self.calculate_speech_rate()

    def calculate_pitch(self):
        # 피치를 계산하고 저장합니다.
        pitch = self.sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_times = pitch.xs()

        # 피치 값이 0이 아닌 경우만 필터링합니다.
        pitch_times = [time for time, value in zip(pitch_times, pitch_values) if value != 0]
        pitch_values = [value for value in pitch_values if value != 0]

        return pitch_times, pitch_values

    def calculate_formants(self, pitch_time_range=0.1):
        self.calculate_pitch()
        pitch_times, _ = self.pitch

        formant = self.sound.to_formant_burg()
        times = formant.t_grid()
        f1_values = [formant.get_value_at_time(1, time) for time in times]
        f2_values = [formant.get_value_at_time(2, time) for time in times]
        f3_values = [formant.get_value_at_time(3, time) for time in times]

        valid_data = []

        for pitch_time in pitch_times:
            # 피치 시간 주변의 포먼트 값을 추출합니다.
            time_window = [time for time in times if pitch_time - pitch_time_range <= time <= pitch_time + pitch_time_range]
            f1_window = [f1 for time, f1 in zip(times, f1_values) if time in time_window and not np.isnan(f1)]
            f2_window = [f2 for time, f2 in zip(times, f2_values) if time in time_window and not np.isnan(f2)]
            f3_window = [f3 for time, f3 in zip(times, f3_values) if time in time_window and not np.isnan(f3)]

            if f1_window and f2_window and f3_window:
                # 해당 시간대의 평균 포먼트 값을 계산합니다.
                avg_f1 = np.mean(f1_window)
                avg_f2 = np.mean(f2_window)
                avg_f3 = np.mean(f3_window)
                valid_data.append((pitch_time, avg_f1, avg_f2, avg_f3))

        valid_times, valid_f1, valid_f2, valid_f3 = zip(*valid_data) if valid_data else ([], [], [], [])

        return valid_times, valid_f1, valid_f2, valid_f3

    def calculate_speech_rate(self):
        """
        발화속도(초당 음절수)를 구한다.
        :param sound: parselmouth로 생성한 sound 객체
        :return: speech_rate
        """
        threshold, threshold2, threshold3 = self.get_threshold(self.intensity, silence_db=-25)
        textgrid = self.get_textgrid(self.intensity, threshold3=threshold3, min_pause=0.3)
        num_peaks, time, sound_from_intensity_matrix = self.get_num_peaks(self.intensity)
        time_peaks, peak_count, intensities = self.get_time_peaks(num_peaks, time, sound_from_intensity_matrix, threshold)
        valid_peak_count, current_time, current_int, valid_time = self.get_valid_peak_count(time_peaks, peak_count, self.intensity, intensities, min_dip = 2)

        speech_rate = valid_peak_count / self.sound.end_time
        return speech_rate

    def plot_spectrogram(self):
        plt.figure(figsize=(10, 4))
        self.draw_spectrogram(self.sound.to_spectrogram())
        plt.title('Spectrogram')
        plt.tight_layout()
        self.save_figure(suffix='_spectrogram')

    def plot_formants(self):
        plt.figure(figsize=(10, 4))
        # self.formants를 이용하여 포먼트 그래프를 그립니다.
        times, f1, f2, f3 = self.formants
        plt.scatter(times, f1, color='r', label='F1', s=8)
        plt.scatter(times, f2, color='g', label='F2', s=8)
        plt.scatter(times, f3, color='b', label='F3', s=8)
        plt.legend(loc='upper right')
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title('Formants')
        plt.tight_layout()
        self.save_figure(suffix='_formants')

    def plot_pitch(self):
        plt.figure(figsize=(10, 4))
        # self.pitch를 이용하여 피치 그래프를 그립니다.
        pitch_times, pitch_values = self.pitch
        plt.scatter(pitch_times, pitch_values, s=8)
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch [Hz]")
        plt.title("Pitch Curve")
        plt.ylim(0, self.sound.to_pitch().ceiling)
        plt.tight_layout()
        self.save_figure(suffix='_pitch')

    def draw_spectrogram(self, spectrogram):
        x, y = spectrogram.x_grid(), spectrogram.y_grid()
        sg_db = 10 * np.log10(spectrogram.values)
        plt.pcolormesh(x, y, sg_db, shading='auto')
        plt.ylim([spectrogram.ymin, spectrogram.ymax])
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")

    def save_figure(self, suffix):
        sub_dir = os.path.join(self.jpg_dir, self.base_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        figure_path = os.path.join(sub_dir, f"{self.base_name}{suffix}.jpg")
        plt.savefig(figure_path)
        plt.close()

    def save_features_to_json(self, json_dir='out/json/features'):
        # 결과를 저장할 디렉토리가 없으면 생성합니다.
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        # 데이터를 JSON 형식으로 준비합니다.
        data = {
            "pitch": list(zip(*self.pitch)),
            "formants": list(zip(*self.formants)),
            "speech_rate": self.speech_rate
        }

        # JSON 파일로 저장합니다.
        json_path = os.path.join(json_dir, f"{self.base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_average_features(json_dir='out/json/features'):
    pitch_values = []
    f1_values = []
    f2_values = []
    f3_values = []

    # JSON 파일들을 읽습니다.
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 피치 값들을 추가합니다.
            for time, pitch in data["pitch"]:
                if pitch is not None:
                    pitch_values.append(pitch)

            # 포먼트 값들을 추가합니다.
            for formants in data["formants"]:
                f1_values.append(formants[1])
                f2_values.append(formants[2])
                f3_values.append(formants[3])

    # 평균값을 계산합니다.
    avg_pitch = np.mean(pitch_values) if pitch_values else None
    avg_f1 = np.mean(f1_values) if f1_values else None
    avg_f2 = np.mean(f2_values) if f2_values else None
    avg_f3 = np.mean(f3_values) if f3_values else None

    return avg_pitch, avg_f1, avg_f2, avg_f3
    
    
if __name__ == '__main__':
    wave_files = glob.glob('out/split-wav/**/*.wav', recursive=True)  # 필요시 WAV 파일의 경로를 수정합니다.
    print("음성 분석 시작")
    for wave_file in tqdm(wave_files, desc='Total Wavs'):
        try:
            analyzer = SpeechAnalysis(wave_file)
            analyzer.plot_spectrogram()
            analyzer.plot_formants()
            analyzer.plot_pitch()
            analyzer.save_features_to_json()
            print(f"음성 분석 완료: {wave_file}")
        except Exception as e:
            print(f'Error SpeechAnalysis for {wave_file}: {e}')
            
    # 함수 호출 예시
    print("전체 음성에 대한 통계 산출")
    average_pitch, average_f1, average_f2, average_f3 = calculate_average_features(json_dir='out/json/features')
    print("Average Pitch:", average_pitch)
    print("Average F1:", average_f1)
    print("Average F2:", average_f2)
    print("Average F3:", average_f3)