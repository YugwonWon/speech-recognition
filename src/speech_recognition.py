import os
import csv
import glob
import json

import torch
# import whisper
import whisper_timestamped as whisper
from tqdm import tqdm
from pydub import AudioSegment
from pyannote.audio import Pipeline

class SpeechRecognition:
    """
    화자 분리 및 STT 음성 인식을 수행합니다.
    """
    def __init__(self, access_token='Your_Huggingface_Access_Token'):
        # Pyannote 화자 분리 파이프라인 초기화
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=access_token)
        self.diarization_pipeline.to(torch.device("cuda"))
        # Whisper 모델 로드
        self.whisper_model = whisper.load_model("base")

    def separate_speakers(self, audio_path):
        # 화자 분리 수행
        diarization = self.diarization_pipeline(audio_path)
        return diarization

    def save_sep_dict(self, diarization):
        """
        화자 분리된 객체에서 필요한 정보를 dict 형식으로 저장
        :param diarization: pyannote.audio의 화자 분리 객체
        :return speakers_dict: 화자 분리 정보가 들어있는 사전
        """
        speakers_dict = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers_dict:
                speakers_dict[speaker] = []
            speakers_dict[speaker].append([turn.start, turn.end])

        return speakers_dict

    def transcribe_speech(self, audio_path):
        # Whisper를 사용한 음성 인식
        result = whisper.transcribe(self.whisper_model, audio_path, language='ko')
        return result

    def split_and_save_speakers(self, audio_path, speakers_dict, output_dir='out/split-wav'):
        """
        화자별로 오디오 파일을 분리하고 저장합니다.
        :param audio_path: 원본 오디오 파일 경로
        :param speakers_dict: 화자별 타임스탬프가 저장된 사전
        :param output_dir: 분리된 파일을 저장할 디렉토리
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio = AudioSegment.from_file(audio_path)
        basename = os.path.basename(audio_path).split('.')[0]
        for speaker in speakers_dict:
            for i, (start, end) in enumerate(speakers_dict[speaker]):
                segment_length = end - start  # 단위: 초
                if segment_length < 0.13:
                    continue  # 세그먼트 길이가 0.13초보다 짧으면 저장을 건너뜁니다.

                # 시간 정보를 초 단위로 반올림합니다.
                start_s = round(start, 1)
                end_s = round(end, 1)
                i_num = str(i).zfill(3)
                filename = f"{basename}_{speaker}_segment_{i_num}_{start_s}_{end_s}.wav"
                speaker_segment = audio[start_s*1000:end_s*1000]
                speaker_segment.export(f"{output_dir}/{filename}", format="wav")

    def save_speaker_diarization_to_csv(self, speaker_dict, audio_filename, diarization_csv_writer):
        """
        화자 분리 결과를 CSV 형태로 저장합니다.
        :param speaker_dict: 화자 분리 정보가 담긴 사전
        :param audio_filename: 처리중인 오디오 파일의 이름
        :param diarization_csv_writer: 화자 분리 정보 CSV 파일의 writer 객체
        """
        for speaker, segments in speaker_dict.items():
            for segment in segments:
                start, end = segment
                diarization_csv_writer.writerow([audio_filename, speaker, start, end])

    def save_to_csv(self, stt_result, audio_path, audio_filename, global_csv_writer):
        """
        STT 결과를 CSV 형태로 저장합니다. 여기서는 파일별 STT 결과를 전역 CSV에 추가합니다.
        :param stt_result: STT 결과
        :param audio_filename: 처리중인 오디오 파일의 이름
        :param global_csv_writer: 전역 CSV 파일의 writer 객체
        """
        for seg_index, seg in enumerate(stt_result['segments'], start=1):
            text = seg['text'].strip()  # 문장
            for word_index, word in enumerate(seg['words'], start=1):
                word_text = word['text'].strip()  # 어절 텍스트
                word_start = word['start']  # 어절 시작 시간
                word_end = word['end']  # 어절 종료 시간
                confidence = word['confidence']  # 어절 신뢰도
                word_id = f"{seg_index}-{word_index}"  # 어절 ID를 "문장번호-어절번호" 형식으로 변경
                global_csv_writer.writerow([audio_path, audio_filename, seg_index, text, word_id, word_text, word_start, word_end, confidence])

    def process_files(self, audio_files, output_dir='out/csv'):
        # output 폴더 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        global_stt_csv_path = os.path.join(output_dir, 'speech_transcription.csv')
        diarization_csv_path = os.path.join(output_dir, 'speaker_diarization.csv')

        # csv 파일 생성
        with open(global_stt_csv_path, 'w', newline='', encoding='utf-8') as stt_csvfile, \
             open(diarization_csv_path, 'w', newline='', encoding='utf-8') as diar_csvfile:
            global_csv_writer = csv.writer(stt_csvfile)
            diarization_csv_writer = csv.writer(diar_csvfile)

            stt_headers = ['위치', '파일명', '문장 번호', '문장', '어절 번호', '어절', '어절 시작', '어절 종료', '신뢰도']
            diar_headers = ['위치', '파일명', '화자 ID', '시작 시간', '종료 시간']

            global_csv_writer.writerow(stt_headers)
            diarization_csv_writer.writerow(diar_headers)

            for audio_path in tqdm(audio_files, desc='Total Wavs'):
                base_name = os.path.splitext(os.path.basename(audio_path))[0]

                # 화자 분리 실행
                try: # 예외 처리
                    diarization_result = self.separate_speakers(audio_path)
                except Exception as e:
                    print(f'Error in diarization for {audio_path}: {e}')
                    continue
                speaker_dict = self.save_sep_dict(diarization_result)

                for speaker, segments in speaker_dict.items():
                    for segment in segments:
                        start, end = segment
                        diarization_csv_writer.writerow([audio_path, base_name, speaker, start, end])

                split_output_dir = os.path.join(output_dir, 'split-wav', base_name)

                # 화자분리된 wav파일 저장
                try:
                    self.split_and_save_speakers(audio_path, speaker_dict)
                except Exception as e:
                    print(f'Error in split save wavfile for {audio_path}: {e}')
                    continue

                # STT 처리
                try:
                    stt_result = self.transcribe_speech(audio_path)
                    self.save_to_csv(stt_result, audio_path, base_name, global_csv_writer)
                except Exception as e:
                    print(f'Error in split save wavfile for {audio_path}: {e}')
                    continue

                print(f'{audio_path}: 화자 분리 및 STT 처리 완료.')

if __name__ == '__main__':
    # config 파일 불러오기
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 사용 예시
    recognition = SpeechRecognition(access_token=config['hf_access_key'])
    audio_files = glob.glob("data/wav-files/**/*.wav", recursive=True)
    audio_files = sorted(audio_files)
    print(f'audio files: {audio_files[:10]}')
    recognition.process_files(audio_files, output_dir='out/csv')