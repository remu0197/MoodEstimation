from watson_developer_cloud import SpeechToTextV1
import watson_developer_cloud
import json
import MeCab
import csv, os, glob, sys, copy

class SpeechToText:
    def __init__(self):
        self.__API_KEY = "cmovZ3SbW34kbhsqumH2lYeo-AOy-lMBiSqapGLZnl3x"
        self.__IBM_URL = "https://api.jp-tok.speech-to-text.watson.cloud.ibm.com/instances/4985f968-7799-4849-abff-934822d4dfd4"
        self.__LANG = "ja-JP_BroadbandModel"

    def to_text(self, sound_path, sound_type="wav", is_update=False):
        dir_name = os.path.basename(os.path.dirname(sound_path))
        export_dir = "../data/speech_to_text/" + dir_name + "/"
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)

        file_id = os.path.splitext(os.path.basename(sound_path))[0]
        export_path = export_dir + "/" + file_id + ".csv"

        if os.path.exists(export_path) and not is_update:
            return

        print("CURRENT EXPORT: " + export_path)

        audio_file = open(sound_path, "rb")

        stt = SpeechToTextV1(iam_apikey=self.__API_KEY, url=self.__IBM_URL)

        try: 
            result_json = stt.recognize(audio=audio_file, content_type="audio/"+sound_type, model=self.__LANG, timestamp=False, speaker_labels=True)
        except watson_developer_cloud.watson_service.WatsonApiException:
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(["no_outputs", "0", "0"])
            return
            
        stt_result = result_json.get_result()

        timestamps = []
        for temp in stt_result["results"]:
            for timestamp in temp["alternatives"][0]["timestamps"]:
                timestamps.append(timestamp)

        with open(export_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(timestamps)

    def adjust_VAD(self, sound_path, is_update=False):
        dir_name = os.path.basename(os.path.dirname(sound_path))
        timestamp_dir = "../data/speech_to_text/" + dir_name + "/"
        file_id = os.path.splitext(os.path.basename(sound_path))[0]
        timestamp_path = timestamp_dir + "/" + file_id + ".csv"

        export_dir = "../data/edit_texts/" + dir_name + "/"
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
        export_path = export_dir + file_id + ".csv"

        if os.path.exists(export_path) and not is_update:
            return True  

        print("ADJUST VAD:  " + timestamp_path)

        if not os.path.exists(timestamp_path):
            print(" ERROR:   " + timestamp_path + " is not exist.")
            return False

        timestamps = []

        with open(timestamp_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                timestamps.append(row)

        section_id = file_id.split("_")
        vad_filename = str(int(section_id[1]) + 1) + dir_name.split("_")[int(section_id[0])].rjust(3, "0") + ".csv"
        vad_filepath = "../../VAD/data/csv/" + vad_filename

        if not os.path.exists(vad_filepath):
            print(" ERROR:   " + vad_filepath + " is not exist.")
            return False

        active_sections = []
        start, end = 0.0, 0.0

        with open(vad_filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                end = float(row[0])
                if start != 0.0 and end != 0.0:
                    active_sections.append([start, end])
                start = float(row[1])

        exports = []

        for section in active_sections:
            text = ""
            while len(timestamps) > 0:
                if float(timestamps[0][2]) < section[0]:
                    timestamps.pop(0)
                elif float(timestamps[0][1]) > section[1]:
                    break
                else:
                    text += timestamps[0][0]
                    timestamps.pop(0)

            temp = []
            temp.append(str(section[0]))
            temp.append(str(section[1]))
            temp.append(text)
            exports.append(temp)   
   
        with open(export_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(exports)

    def divide_filler(self, text_path):
        pos_id = "C:\Program Files\MeCab\dic\ipadic\pos-id.def"
        return


if __name__ == "__main__":
    hoge = SpeechToText()
    pathes = glob.glob("../data/dataset/**/*_*.wav")
    
    for path in pathes:
        hoge.to_text(sound_path=path)
        hoge.adjust_VAD(sound_path=path)


# from watson_developer_cloud import SpeechToTextV1
# import json

# # define
# apikey = "[API鍵]"
# url = "[URL]"
# audio_file = open("voice.wav", "rb")
# cont_type = "audio/wav"
# lang = "ja-JP_BroadbandModel"

# # watson connection
# stt = SpeechToTextV1(iam_apikey=apikey,url=url)
# result_json = stt.recognize(audio=audio_file, content_type=cont_type, model=lang, timestamps = False)

# #print
# sttResult = result_json.get_result()
# print(sttResult)

# # json file save
# result = json.dumps(result_json, indent=2)
# f = open("result.json", "w")
# f.write(result)
# f.close()