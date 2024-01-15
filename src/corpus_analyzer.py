import os

import json
import glob

from tqdm import tqdm

from speech_analysis import SpeechAnalysis

class CorpusAnalyzer(SpeechAnalysis):
    """
    각 코퍼스에서의 발화를 분석하여 기초 자료를 추출합니다.
    나이, 성별, 학력, 질환, 음성피쳐 등등
    """
    def __init__(self, corpus_list, out_name):
        """
        wav file 경로를 미리 생성해놓고, 각 json을 돌면서 wav를 찾아서 분석합니다.
        """
        self.corpus_list = corpus_list
        self.out_name = out_name
        self.corpus_json_dict = {}
        self.corpus_wav_dict = {}

        # 디렉토리에서 모든 WAV 파일과 JSON 파일 찾기
        for idx, corpus in enumerate(corpus_list):
            wavs = glob.glob(os.path.join(corpus, '**/*.wav'), recursive=True) # file 이름 규칙은 변경될 수 있다.
            json_files = glob.glob(os.path.join(corpus, '**/*.json'), recursive=True)
            self.corpus_json_dict[f'{str(idx)}_json'] = json_files # 요청된 corpus 리스트 순서대로 번호를 부여한다.
            self.corpus_wav_dict[f'{str(idx)}_wavs'] = self.mapping_name(wavs)

        self.results = []

    def mapping_name(self, target_dir_files):
        wav_id_dict = {}
        for wav in target_dir_files:
            bn =  os.path.basename(wav).split('.')[0]
            wav_id_dict[bn] = wav

        return wav_id_dict

    def load_user_info(self):
        with open(self.user_info_file, 'r') as file:
            data = json.load(file)
            return data

    def save_analysis_to_json(self, out_name):
        if not os.path.isdir('out/json/corpus'):
            os.mkdir('out/json/corpus')
        with open(f'out/json/corpus/{out_name}.json', 'w', encoding='utf-8') as file:
            json.dump(self.results, file, ensure_ascii=False, indent=4)

    def get_metadata(self, meta_file:str, corpus_id:int):
        base_name = os.path.basename(meta_file).split('.')[0]

        # 한국어방언데이터(충청,전라,제주)
        if corpus_id == 0:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            if len(data['speaker']) > 1:
                print('speaker가 2명 이상입니다.')
            gender = data['speaker'][0]['gender']
            age = 2023 - int(data['speaker'][0]['birthYear'])
            meta_dict = {
                "speaker_id" : data['speaker'],
                "file_path" : meta_file,
                "file_name" : base_name,
                "gender" : gender,
                "age" : age,
                "segment_id": '',
                "disease" : ''
            }
        elif corpus_id == 1: # 다른 json구조를 가진 코퍼스를 추가하여 같이 처리할 수 있음
            pass

        elif corpus_id == 2:
            pass

        return meta_dict

    def analyze_speech(self, json_files, corpus_id=0):
        """
        메인 분석 함수
        :param json_files: corpus json 파일, 메타데이터가 들어있다.
        :param corpus_id: 코퍼스 인덱스, 인덱스에 따라 전처리한다.
        :return extract_features: 음성 feature 추출 실행
        """
        # 낭송 데이터의 경우 json 파일 하나에
        # 모든 정보가 다 들어있으므로 따로 처리해야 한다.
        for json_file in tqdm(json_files):
            try:
                base_name = os.path.basename(json_file).split('.')[0]
                user_meta = self.get_metadata(json_file, corpus_id)

                self.extract_features(user_meta=user_meta, corpus_id=corpus_id)
            except:
                print(f'error {json_file}')
                continue

    def extract_features(self, user_meta, corpus_id=0):
        """
        음성 특성 추출 함수
        :param user_meta: user meta data
        :return analysis_results: 분석 결과 저장
        """
        analyzer = SpeechAnalysis(self.corpus_wav_dict[f'{str(corpus_id)}_wavs'][user_meta['file_name']])
        analysis_results = {
            "meta_data": user_meta,
            "pitch": list(zip(*analyzer.pitch)),
            "formants": list(zip(*analyzer.formants)),
            "speech_rate": analyzer.speech_rate
        }
        self.results.append(analysis_results)

    def run(self):
        """
        json에 있는 id의 wav를 찾아서 음성 분석 시행
        """
        for idx, (corpus, k) in tqdm(enumerate(zip(self.corpus_list, self.corpus_json_dict.keys()))):
            print(f'\nStart Analysis index {idx}')
            print(f'target_corpus_directory: {corpus}\nindex: {k}')
            self.analyze_speech(self.corpus_json_dict[k], corpus_id=idx)
            print(f'Finish')
            print(f'---------------------------------------------')
            self.save_analysis_to_json(f'{self.out_name}_{idx}')
            print(f'save results out/json/corpus/{self.out_name}_{idx}')
            self.results = [] # 변수 초기화

if __name__ == '__main__':
    # 0: 중.노년층 한국어 방언 데이터
    corpus_list = ['./data/corpus']
    analyzer = CorpusAnalyzer(corpus_list, out_name='speech_analysis')
    analyzer.run()

