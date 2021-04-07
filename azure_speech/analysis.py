import azure.cognitiveservices.speech as speechsdk
import json

SPEECH_KEY = "09205fb8bb7a4d8bb5069d592c09596d"
SERVICE_REGION = "westeurope"


class Analysis:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.json_path = f"F:\\LangBot\\{chat_id}\\personal\\azure_analysis.json"
        self.speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)

    def analyse_and_save(self, voice_path, reference_text):

        audio_config = speechsdk.audio.AudioConfig(filename=voice_path)

        pronunciation_assessment_config = \
            speechsdk.PronunciationAssessmentConfig(reference_text=reference_text,
                                                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                                                    granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config)

        # apply the pronunciation assessment configuration to the speech recognizer
        pronunciation_assessment_config.apply_to(speech_recognizer)
        result = speech_recognizer.recognize_once()
        pronunciation_assessment_result = speechsdk.PronunciationAssessmentResult(result)
        pronunciation_score = pronunciation_assessment_result.pronunciation_score

        # save data to json
        try:
            with open(self.json_path, 'r') as handle:
                data = json.load(handle)
            data.append(json.loads(result.json)['NBest'])
            with open(self.json_path, 'w') as handle:
                json.dump(data, handle, indent=4)
        except FileNotFoundError:
            with open(self.json_path, 'w+') as handle:
                data = json.loads(result.json)['NBest']
                json.dump(data, handle, indent=4)

        phonemes = []
        for word in pronunciation_assessment_result.words:
            for _phoneme in word.phonemes:
                phonemes.append(_phoneme.phoneme.upper())

        return {
            'pronunciation_score': pronunciation_score,
            'accuracy_score': pronunciation_assessment_result.accuracy_score,
            'completeness_score': pronunciation_assessment_result.completeness_score,
            'fluency_score': pronunciation_assessment_result.fluency_score,
            'words': [word.word for word in pronunciation_assessment_result.words],
            'phonemes': phonemes,
            'error_type': [word.error_type for word in pronunciation_assessment_result.words]
        }

    def get_phonemes(self, recognized_word):
        try:
            with open(self.json_path, 'r') as handle:
                data = json.load(handle)
                for entry in data:
                    if entry['Words'][0]['Word'] == recognized_word:
                        phonemes = [a['Phoneme'].upper() for a in entry['Words'][0]['Phonemes']]
                        return phonemes
        except FileNotFoundError:
            return []