# Speech-Recognition

* Python을 사용하여 음성 인식과 화자 분리를 수행합니다. 먼저 필요한 라이브러리를 설치해야 합니다.
* pyannote, whisper, praat 라이브러리 사용

## 사전 준비사항1(가상환경 설정)

* 현재 프로젝트에 파이썬 가상환경 생성
* 가상환경 활성화
* pip upgrade
* 라이브러리 설치

```

python3 -m venv venv 

source venv/bin/activate 

pip install --upgrade pip 

pip install -r requirements.txt 

```

* 파이썬 인터프리터(단축키: 컨트롤+쉬프트+P) 선택 -> venv 선택

## 사전 준비사항2(HuggingFace Access Key 등록)

- [허깅페이스 홈페이지](https://huggingface.co/) 접속
- 이메일과 비밀번호(대문자, 소문자, 숫자 포함)를 입력하고 회원가입을 합니다.
- https://hf.co/pyannote/segmentation
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization
- https://huggingface.co/pyannote/speaker-diarization-3.1
- 위의 사이트에 접속하여 login -> `대학, 대학교 주소` 정보를 입력하고 동의합니다.
- https://huggingface.co/settings/tokens 에 접속하여 `New Token`을 눌러 access key를 생성합니다.
- 파일 목록에 있는 `config.json`에 해당 키를 입력하고 `ctrl+s`를 눌러 파일을 저장합니다.

## 1. SpeechRecognition 클래스 정의

./src/speech_recognition.py 에서는

`SpeechRecognition` 클래스는 음성 인식과 화자 분리를 수행하는 기능을 제공합니다. 이 클래스는 두 가지 주요 기능을 포함합니다: 화자 분리를 위한 Pyannote 라이브러리와 음성 텍스트 변환을 위한 Whisper 모델입니다.

### 화자별 오디오 분리 및 저장

`split_and_save_speakers` 메서드는 화자별로 오디오 파일을 분리하고 각각의 파일을 저장합니다. 이 과정에서 오디오 파일은 화자별로 나누어지며, 각 파일은 해당 화자의 말하는 부분만 포함합니다.

### 전체 음성 인식 결과 저장

`save_transcriptions` 및 `save_text` 메서드는 음성 인식 결과를 텍스트 파일로 저장합니다. `save_transcriptions`는 분리된 각 화자의 음성 인식 결과를 저장하고, `save_text`는 전체 오디오 파일의 음성 인식 결과를 저장합니다.

### 음성 파일 처리

`process_files` 메서드는 여러 오디오 파일을 처리하는 데 사용됩니다. 이 메서드는 각 오디오 파일에 대해 화자 분리, 오디오 파일 분리, 음성 인식을 수행하고 결과를 저장합니다.

## 2. Praat-parselmouth를 사용한 음성 분석

./src/speech_analysis.yp에서는

`praat-parselmouth` 라이브러리를 사용하여 음성 분석을 진행합니다. 이 라이브러리는 음성학 및 언어학에서 음성의 음향적 및 음성학적 특성을 분석하는 데 널리 사용되는 Praat 소프트웨어를 파이썬에 연결합니다. 스펙트로그램, 포먼트, 피치를 그리고, 말하기 속도를 측정할 것입니다.

### Praat 클래스 정의

`Praat` 클래스는 음성 분석을 위한 여러 함수를 포함하고 있으며, Praat 소프트웨어의 기능을 활용합니다. 이 클래스의 메서드들은 발화 시간 측정, 강도 분석, 무음 구간 탐지 등 다양한 음성 분석 작업을 수행합니다.

### SpeechAnalysis 클래스 정의

`SpeechAnalysis` 클래스를 정의하여 음성 파일 분석을 수행합니다. 생성자에서는 WAV 파일의 경로를 입력받아 `parselmouth.Sound` 객체를 초기화하고, 필요한 디렉토리를 생성합니다. `plot_formants_and_spectrogram`, `draw_spectrogram`, `draw_formants`, `measure_speech_rate` 메서드를 통해 스펙트로그램, 포먼트, 말하기 속도를 분석하고 그래프로 나타냅니다.

## 3. CorpusAnalyzer 클래스

./src/corpus_analyzer.py 에서는

`CorpusAnalyzer` 클래스는 다양한 코퍼스의 음성 데이터를 분석합니다. 이 클래스는 `SpeechAnalysis` 클래스를 상속받아 음성 데이터의 다양한 특성(피치, 포먼트, 발화 속도 등)을 추출하고, 각 코퍼스의 메타데이터(나이, 성별, 학력, 질환 등)와 함께 저장합니다.
