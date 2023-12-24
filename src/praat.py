from typing import List, Tuple

import parselmouth
from parselmouth.praat import call

class Praat:
    def __init__(self) -> None:
        pass
    
    
    def get_speaking_time(self) -> float:
        """
        휴지를 제외한 발화 시간을 구한다
        :param sound: parselmouth로 생성한 sound 객체
        :return: speaking_time
        """
        threshold, threshold2, threshold3 = self.get_threshold(self.intensity, silence_db=-25)
        textgrid = self.get_textgrid(self.intensity, threshold3=threshold3, min_pause=0.3)
        silence_tier = self.get_silence_tier(textgrid)
        silence_table = self.get_silence_table(silence_tier)
        n_pauses = self.get_n_pauses(silence_table)
        
        speaking_time = 0
        for ipause in range(n_pauses):
            pause = ipause + 1
            begin_sound = call(silence_table, "Get value", pause, 1)
            end_sound = call(silence_table, "Get value", pause, 2)
            speaking_dur = end_sound - begin_sound
            speaking_time += speaking_dur
        return speaking_time
    
    def get_intensity(self, value=50) -> parselmouth.Intensity:
        """
        sound 객체로부터 intensity list를 계산한다.
        :param sound: signal, salplerate로부터 얻은 sound 객체
        :return: intensity
        """
        return self.sound.to_intensity(value)
    
    @staticmethod
    def get_threshold(intensity: parselmouth.Intensity, silence_db=-25) -> Tuple[float, float, float]:
        """
        intensity 임계값을 구한다.
        :param intensity: sound 객체의 intensity list
        :silence_db: 무음 감지를 위한 표준 설정
        :return: threshold, threshold2, threshold3
        """
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
        # 포만트 분위수를 가져온다. 0~1 사이의 값을 가지며 기본 분포의 중앙값 추정치를 얻으려면 0.5를 지정한다. 
        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

        threshold = max_99_intensity + silence_db
        threshold2 = max_intensity - max_99_intensity
        threshold3 = silence_db - threshold2
        if threshold < min_intensity:
            threshold = min_intensity
        return threshold, threshold2, threshold3

    @staticmethod
    def get_textgrid(intensity: parselmouth.Intensity, threshold3: float, min_pause=0.25) -> parselmouth.TextGrid:
        """
        sound의 무음 및 소리 간격이 표시되는 textgrid를 만든다.
        :param intensity: 강도
        :threshold3: 강도 임계값
        :min_pause: 최소 휴지(0.3초)
        :retrun: textgrid
        """
        textgrid = call(intensity, "To TextGrid (silences)", threshold3, min_pause, 0.1, "silent", "sounding")
        return textgrid

    @staticmethod
    def get_silence_tier(textgrid: parselmouth.TextGrid) -> parselmouth.Data:
        """
        praat의 textgrid tier 표시 기능을 이용하여 무음 구간을 표시한다.
        :param textgrid: textgrid 
        :return: praat "Extract tier" 함수 호출
        """
        return call(textgrid, "Extract tier", 1)

    @staticmethod
    def get_silence_table(silence_tier: parselmouth.Data) -> parselmouth.Data:
        """
        praat의 textgrid tier 표시 기능을 이용하여 무음 구간을 표시한다.
        :param silence_tier: silence_tier interval
        :return: praat "Down to TableOfReal", "sounding" 함수 호출
        """
        return call(silence_tier, "Down to TableOfReal", "sounding")

    @staticmethod
    def get_n_pauses(silence_table: parselmouth.Data) -> int:
        """
        sound에서 전체 pause의 개수를 구한다.
        :param silence_table: silence_table
        :retrun: praat "Get number of rows" 함수 호출 
        """
        return call(silence_table, "Get number of rows")

    @staticmethod
    def get_num_peaks(intensity: parselmouth.Intensity):
        """
        peak의 개수, 위치, 강도 매트릭스를 구한다.
        :param intensity: parselmouth sound 객체로부터 얻은 intensity
        :return num_peaks, time, sound_from_intensity_matrix
        """
        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
        num_peaks = call(point_process, "Get number of points")
        time = [call(point_process, "Get time from index", i + 1) for i in range(num_peaks)]
        
        return num_peaks, time, sound_from_intensity_matrix
    
    @staticmethod
    def get_time_peaks(num_peaks:int , time: List[float], sound_from_intensity_matrix: parselmouth.Sound, threshold: float):
        """
        시간 변화에 따른 peak의 배열을 구한다.
        :param num_peak: peak의 개수
        :time: 시간
        :sound_from_intensity_matrix: 강도 매트릭스
        :threshold: 강도 임계값
        :return: time_peaks, peak_count, intensities
        """
        time_peaks = []
        peak_count = 0
        intensities = []
        for i in range(num_peaks):
            value = call(sound_from_intensity_matrix, "Get value at time", time[i], "Cubic")
            if value > threshold:
                peak_count += 1
                intensities.append(value)
                time_peaks.append(time[i])
        return time_peaks, peak_count, intensities

    @staticmethod
    def get_valid_peak_count(time_peaks: List[float], peak_count: int, intensity: parselmouth.Intensity, intensities: List[float], min_dip = 1.5):
        """
        유효한 피크를 추출한다.
        :param time_peaks: peak의 시간
        :peak_count: peak의 개수
        :intensity: 강도
        :intensities: 강도 리스트
        :min_dip: 현재 intensity와 최소 intensity 차이 최소값
        :return: valid_peak_count, current_time, current_int, valid_time
        """
        valid_peak_count = 0
        current_time = time_peaks[0]
        current_int = intensities[0]
        valid_time = []
        for p in range(peak_count - 1):
            following = p + 1
            dip = call(intensity, "Get minimum", current_time, time_peaks[p + 1], "None")
            diff_int = abs(current_int - dip)
            if diff_int > min_dip:
                valid_peak_count += 1
                valid_time.append(time_peaks[p])  
            current_time = time_peaks[following]
            current_int = call(intensity, "Get value at time", time_peaks[following], "Cubic")

        return valid_peak_count, current_time, current_int, valid_time
    
    
