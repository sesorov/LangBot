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
            data = []
            with open(self.json_path, 'r') as handle:
                data = json.load(handle)
            data.append(json.loads(result.json)['NBest'][0])
            with open(self.json_path, 'w') as handle:
                json.dump(data, handle, indent=4)
        except FileNotFoundError:
            with open(self.json_path, 'w+') as handle:
                data.append(json.loads(result.json)['NBest'][0])
                json.dump(data, handle, indent=4)

        phonemes = []
        for word in pronunciation_assessment_result.words:
            for _phoneme in word.phonemes:
                phone = _phoneme.phoneme
                # Normalize phonemes
                if phone.upper() == 'AX':
                    phone = 'AH'

                phonemes.append(phone.upper())

        return {
            'pronunciation_score': pronunciation_score,
            'accuracy_score': pronunciation_assessment_result.accuracy_score,
            'completeness_score': pronunciation_assessment_result.completeness_score,
            'fluency_score': pronunciation_assessment_result.fluency_score,
            'words': [word.word for word in pronunciation_assessment_result.words],
            'phonemes': phonemes,
            'phonemes_scores': self.get_phonemes_scores(reference_text),
            'error_type': [word.error_type for word in pronunciation_assessment_result.words]
        }

    def get_phonemes(self, recognized_word):
        try:
            with open(self.json_path, 'r') as handle:
                data = json.load(handle)
                for entry in data:
                    if entry['Words'][0]['Word'] == recognized_word:
                        phonemes = [a['Phoneme'].upper() for a in entry['Words'][0]['Phonemes']]
                        for i, phone in enumerate(phonemes):
                            # Normalize phonemes
                            if phone.upper() == 'AX':
                                phonemes[i] = 'AH'
                        return phonemes
        except FileNotFoundError:
            return []

    def get_phonemes_scores(self, recognized_word):
        try:
            with open(self.json_path, 'r') as handle:
                data = json.load(handle)
                for entry in data:
                    if entry['Words'][0]['Word'] == recognized_word:
                        phonemes_scores = {a['Phoneme'].upper().replace('AX', 'AH'):
                                           a['PronunciationAssessment']['AccuracyScore']
                                           for a in entry['Words'][0]['Phonemes']}
                        return phonemes_scores
        except FileNotFoundError:
            return False

    def get_recognized_word(self, voice_path):
        """performs one-shot speech recognition with input from an audio file"""
        # <SpeechRecognitionWithFile>
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
        audio_config = speechsdk.audio.AudioConfig(filename=voice_path)
        # Creates a speech recognizer using a file as audio input, also specify the speech language
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, language="de-DE", audio_config=audio_config)

        # Starts speech recognition, and returns after a single utterance is recognized. The end of a
        # single utterance is determined by listening for silence at the end or until a maximum of 15
        # seconds of audio is processed. It returns the recognition text as result.
        # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
        # shot recognition like command or query.
        # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
        result = speech_recognizer.recognize_once()

        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
        return None
