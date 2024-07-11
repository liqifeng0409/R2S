# coding=utf-8
import argparse
import json
import random
import re
from tqdm import tqdm


def compute_score(input_path, output_path):
    informa_all, underst_all, helpful_all, Loyal_all, Flex_all = 0, 0, 0, 0, 0
    consist_all, cohere_all, emo_all = 0, 0, 0
    Count_loy, Count_flx = 0, 0
    multi_dia_num = 0
    err_num = 0
    with open(file=input_path, mode='r', encoding='utf-8') as f:
        alls = f.read().split("/**/")
    pattern = r':\s*(.*?)]'
    with open(output_path, mode='w', encoding='utf-8') as w1:
        score_list = []
        for i, text in enumerate(tqdm(alls)):
            # if i < 209:
            #     continue
            if "x.x" in text:  # or "\"Example\""
                continue
            if "\"Example\"" in text:
                continue
            while True:
                try:
                    data = eval(text)
                except Exception as e:
                    # print(text)
                    print("发生错误:", e)
                    break
                break
            multi_dia_num += 1
            # data = json.loads(text)
            informa, underst, helpful, Loyal, Flex = 0, 0, 0, 0, 0
            consist, cohere, emo = 0, 0, 0
            Loyal_count, Flex_count = 0, 0
            if len(data["Multi-round Dialogue Evaluation"]) == 0:
                continue
            for dialogue in data["Multi-round Dialogue Evaluation"]:
                if "\'A\': \'\'" in str(dialogue) or "\'B\': \'\'" in str(dialogue):
                    continue
                if 'A-score' in dialogue:
                    if dialogue['A-score'].split(': ')[1][0] == "N":
                        continue
                    else:
                        informa += float(dialogue['A-score'].split(': ')[1][0])
                    # matches = re.findall(': (\d+|\d+/NA)', dialogue['B-score'])
                if 'B-score' in dialogue:
                    matches = re.findall(pattern, dialogue['B-score'])
                    if len(matches) != 4:
                        continue
                    if matches[0] == 'N/A':
                        underst += 0
                    else:
                        underst += float(matches[0])
                    if matches[1] == 'N/A':
                        helpful += 0
                    else:
                        helpful += float(matches[1].replace(" acurate", ""))
                    if matches[2] == 'N/A':
                        Loyal += 0
                    else:
                        Loyal_count += 1
                        Loyal += float(matches[2])
                    if matches[3] == 'N/A':
                        Flex += 0
                    else:
                        Flex_count += 1
                        Flex += float(matches[3])
            num = len(data['Multi-round Dialogue Evaluation'])  # Overall Dialogue Evaluation  Multi-round Dialogue Evaluation

            while True:
                try:
                    ttt = data['Overall Dialogue Evaluation']
                except Exception as e:
                    # print(text)
                    err_num += 1
                    print("发生错误:", e)
                    break
                break

            general_scores = re.findall(pattern, ttt)
            if len(general_scores) != 3:
                continue
            if general_scores[0] == 'N/A':
                consist += 0
            else:
                consist += float(general_scores[0])
            if general_scores[1] == 'N/A':
                cohere += 0
            else:
                cohere += float(general_scores[1])
            if general_scores[2] == 'N/A':
                emo += 0
            else:
                emo += float(general_scores[2])
            consist_all += consist
            cohere_all += cohere
            emo_all += emo

            if Loyal_count == 0 or Loyal == 0:
                loyal_compute = 'N/A'
                Count_loy += 1
            else:
                loyal_compute = '%.2f' % (Loyal / Loyal_count)
            if Flex_count == 0 or Flex == 0:
                flex_compute = 'N/A'
                Count_flx += 1
            else:
                flex_compute = '%.2f' % (Flex / Flex_count)

            single = {'id': i,
                      'Informativeness': '%.2f' % (informa / num),
                      'Understanding': '%.2f' % (underst / num),
                      'Helpfulness': '%.2f' % (helpful / num),
                      'Loyalty': loyal_compute,
                      'Flexibility': flex_compute,
                      'Consistency': '%.2f' % consist,
                      'Coherence': '%.2f' % cohere,
                      'emo': '%.2f' % emo,
                      }
            informa_all += informa / num
            underst_all += underst / num
            helpful_all += helpful / num
            if Loyal != 0:
                Loyal_all += Loyal / Loyal_count
            if Flex != 0:
                Flex_all += Flex / Flex_count
            score_list.append(single)
            w1.write("/**/" + str(single) + '\n')
        w1.write('/**/\n' + "Multi_dia_num: " + str(multi_dia_num) + '\n' +
                 "Information_avg: " + '%.2f' % (informa_all / (multi_dia_num)) + '\n' +
                 "Understand_avg: " + '%.2f' % (underst_all / (multi_dia_num)) + '\n' +
                 "Helpful_avg: " + '%.2f' % (helpful_all / (multi_dia_num)) + '\n' +
                 "Loyal_avg: " + '%.2f' % (Loyal_all / (multi_dia_num - Count_loy)) + '\n' +
                 "Flex_avg: " + '%.2f' % (Flex_all / (multi_dia_num - Count_flx)) + '\n' +
                 "Consistency_avg: " + '%.2f' % (consist_all / (multi_dia_num)) + '\n' +
                 "Coherence_avg: " + '%.2f' % (cohere_all / (multi_dia_num)) + '\n' +
                 "emo_avg: " + '%.2f' % (emo_all / (multi_dia_num)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='/home/build_data/dia/eval/eval_llama_gllm_eng.json', help='')
    parser.add_argument('--output_path', type=str, default="/home/build_data/dia/score/newscore_llama_gllm_eng.json", help='')
    args = parser.parse_args()
    compute_score(
        input_path=args.source_path,
        output_path=args.output_path,
    )
