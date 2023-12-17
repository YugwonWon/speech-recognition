import os
import glob
import json
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline

class SpeechRecognition:
    """
    화자 분리 및 STT 음성 인식을 수행합니다.
    """
    def __init__(self, acces_token='Your_Huggingface_Access_Token'):
        # Pyannote 화자 분리 파이프라인 초기화
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                             use_auth_token=acces_token)
        
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
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def split_and_save_speakers(self, audio_path, speakers_dict, output_dir='out/wav'):
        """
        화자별로 오디오 파일을 분리하고 저장합니다.
        :param audio_path: 원본 오디오 파일 경로
        :param speakers_dict: 화자별 타임스탬프가 저장된 사전
        :param output_dir: 분리된 파일을 저장할 디렉토리
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        audio = AudioSegment.from_wav(audio_path)
        for speaker in speakers_dict:
            for i, (start, end) in enumerate(speakers_dict[speaker]):
                # 시간 정보를 초 단위로 반올림합니다.
                start_s = round(start, 1)
                end_s = round(end, 1)
                filename = f"speaker_{speaker}_segment_{i}_{start_s}_{end_s}.wav"
                speaker_segment = audio[start_s * 1000:end_s * 1000]
                speaker_segment.export(f"{output_dir}/{filename}", format="wav")
    
    def save_transcriptions(self, transcriptions, out_name, output_dir='out/txt'):
        """
        STT 결과를 텍스트 파일로 저장합니다.
        :param transcriptions: STT 결과가 담긴 리스트
        :param output_file: 저장할 텍스트 파일 경로
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{out_name}.txt", 'w', encoding='utf-8') as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")
    
    def save_text(self, text, out_name='result_text', output_dir='out/txt'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{out_name}.txt", 'w', encoding='utf-8') as f:
            f.write(text)
            
    def save_json(self, dict, out_name='result_json', output_dir='out/json'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{out_name}.json", 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)

    def process_files(self, audio_files, output_dir='out'):
        """
        여러 오디오 파일을 처리합니다.
        :param audio_files: 처리할 오디오 파일 목록
        :param output_dir: 결과를 저장할 기본 디렉토리
        """
        for audio_path in audio_files:
          base_name = os.path.splitext(os.path.basename(audio_path))[0]
          # 화자 분리
          # 허깅페이스 access_token 발급받아서 허깅페이스 pyannote에 등록해서 사용
          diarization_result = recognition.separate_speakers(audio_path)
          speaker_dict = recognition.save_sep_dict(diarization_result)
          # 화자 분리정보 저장
          recognition.save_json(speaker_dict, out_name=f'{base_name}_diarization_result')
          print(f'{audio_path}: 화자 분리 저장 완료')

          # speakers_dict의 음성 타임스탬프로부터 음성 잘라서 저장하기("out/wav"에 저장)
          recognition.split_and_save_speakers(audio_path, speaker_dict, output_dir=f'{output_dir}/wav/{base_name}')
          print(f'{audio_path}: 음성 분할하여 WAV파일 저장 완료')
          # 음성 전체에 대해서 STT 실행(data/sample_sound.wav)
          full_text = recognition.transcribe_speech(audio_path)
          print(f"{audio_path}: Full Audio Transcription: ", full_text)

          # 텍스트 파일로 저장
          recognition.save_text(full_text, out_name = f'{base_name}_full_text')
          print(f'{audio_path}: 텍스트 파일 저장 완료')

          # out/wav 디렉토리에서 모든 음성 파일 불러오기
          speaker_files = glob.glob(f"out/wav/{base_name}/*.wav")
          # 파일 이름에 포함된 시간 정보를 기반으로 정렬
          speaker_files.sort(key=lambda x: round(float(x.split('_')[-2]), 1))  # 파일 이름 형식에 따라 조정 필요

          # 음성 잘린 파일에 대해서 STT 실행
          transcriptions = []
          print(f'{audio_path}: 잘린 음성 파일 STT 실행')
          for speaker_audio_path in speaker_files:
              text = recognition.transcribe_speech(speaker_audio_path)
              transcription = f"Transcription of {speaker_audio_path}: {text}"
              print(transcription)
              transcriptions.append(transcription)

          # STT 결과를 텍스트 파일로 저장
          recognition.save_transcriptions(transcriptions, out_name=f'{base_name}_speaker_transcriptions')
    
    
    
if __name__ == '__main__':
    # config 파일 불러오기
    with open('config.json', 'r') as f:
        config = json.load(f)
    # 사용 예시                                          
    recognition = SpeechRecognition(acces_token=config['hf_access_key'])
    audio_files = glob.glob("./data/wav-files/*.wav")
    print(audio_files)
    recognition.process_files(audio_files)