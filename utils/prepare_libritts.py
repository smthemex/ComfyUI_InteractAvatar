import os
import sys
import json
import jieba
from pypinyin import Style, lazy_pinyin
from tqdm import tqdm
import multiprocessing
import pandas as pd
# convert_char_to_pinyin: convert char to pinyin
def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )
    # convert char to pinyin
    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list
# run: run
def run(base_dir, raw_data, start_index, end_index, thread):
    output_data = []
    for idx in range(start_index, end_index):
        if (idx-start_index) % 100 == 0:
            print('processed {:d}/{:d}, ranging from [{:d}, {:d}], thread number {:d}'.format(
                idx+1-start_index, end_index-start_index, start_index, end_index, thread))
        try:
            data = raw_data[idx]
            text = open(f"{base_dir}/{data['text_file']}", "r").readline()
            out_pinyin = convert_char_to_pinyin([text])
            out_pinyin = out_pinyin[0]
            output_data.append(
                {
                    "base_dir": base_dir, 
                    "audio_file": data["audio_file"], 
                    "text_file": data["text_file"], 
                    "pinyin": out_pinyin, 
                }
            )
        except:
            pass
    
    return output_data
# main: main
def main(base_dir, raw_data, output_file, num_proc=32):
    
    processor_list = []
    pool = multiprocessing.Pool(processes=num_proc)
    
    for thr in range(num_proc):
        start, end = len(raw_data) // num_proc * thr, len(raw_data) // num_proc * (thr + 1)
        if thr == num_proc - 1:
            end = len(raw_data)
        
        processor_list.append(pool.apply_async(run, (base_dir, raw_data, start, end, thr, )))
            
    pool.close()
    pool.join()
    
    total_file_info = []
    for proc in processor_list:
        total_file_info += proc.get()
        
    df = pd.DataFrame(total_file_info)
    df.to_csv(output_file, index=False)
# prepare_libritts: prepare libritts
if __name__ == "__main__":
    input_json = "/apdcephfs_cq8/share_1367250/zixiangzhou/projects/VideoChat/data_pipeline/LibriTTS/LibriTTS_test.json"
    with open(input_json, "r") as f:
        raw_data = json.load(f)
    
    main("/apdcephfs_jn2/share_302243908", raw_data, "LibriTTS_test.csv", num_proc=64)
