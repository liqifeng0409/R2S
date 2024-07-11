# coding=utf-8
import json
import re

import jieba  # jieba分词
import difflib  # 方法一：Python自带标准库计算相似度的方法，可直接用
import numpy as np
from collections import Counter

from tqdm import tqdm


def xgx():
    from scipy import stats
    a = np.array([0, 0, 0, 1, 2, 0, 0])
    b = np.arange(7)
    c, p = stats.pearsonr(a, b)
    print('Pearson correlation：%s' % c)
    print('p-value：%s' % p)
    # Pearson correlation：0.8660254037844385
    # p-value：0.0117248110039547

    from scipy import stats
    s, p1 = stats.spearmanr([1, 2, 3, 7, 5], [5, 6, 5, 8, 9])
    print('Spearman correlation：%s' % s)
    print('p-value：%s' % p1)

    from scipy import stats
    x1 = [1, 4, 13, 2, 1]
    x2 = [1, 4, 7, 1, 0]
    k, p2 = stats.kendalltau(x1, x2)
    print('Kendallta correlatio：%s' % k)
    print('p-value：%s' % p2)


def fugai_ww(input, output, label):
    from bert4vec import Bert4Vec

    with open(input, mode='r', encoding='utf-8') as f1:
        alls = f1.read().split('/**/')
    all_data = []
    single_data, single_text, find_text = [], [], []
    asd = r'Find\[([^}]*)\]'
    for i, text in enumerate(tqdm(alls)):
        # if i < 50:
        #     continue
        if "Reference" not in text:
            continue
        while True:
            try:
                text = text.replace(" \"[A: ", " {\"A\": \"").replace("  B: ", "\", \"B\": \"") \
                    .replace("]\",", "\"},")
                data = json.loads(json.dumps(eval(text)))
                # ref = data['Reference'].replace("\n", "").split('】').split('。')[:-1]
                ref_ = data['Reference'].replace("\n", "").split('】')
                if len(ref_) > 1:
                    ref = ref_[1].split('。')[:-1]
                else:
                    ref = ref_[0].split('。')[:-1]  # ref_.split('。')[:-1]
                answer = {
                    "id": "",
                    "content": "",
                }
                for j, dia in enumerate(data['Multi-round Dialogue Content']):
                    answer['id'] = dia['id']
                    single_text.append(dia['B'])
                    # if len(dia) == 3:
                    #     single_text.append(dia['B'])
                    # if len(dia) == 5:
                    #     if '无直接信息' in dia['Thought']:
                    #         answer['content'] = "null"
                    #     else:
                    #         part = re.search(asd, dia['Thought']).group(1)
                    #         answer['content'] = dia['B']
                    #         single_text.append(dia['B'])
                    #         find_text.append(part)
                    #     single_data.append(answer)
                    # if label == 'base':
                    #     single_text.append(dia['B'])
                    # if label == 'my':
                    #     if '无直接信息' in dia['Thought']:
                    #         answer['content'] = "null"
                    #     else:
                    #         part = re.search(asd, dia['Thought']).group(1)
                    #         answer['content'] = dia['B']
                    #         single_text.append(dia['B'])
                    #         find_text.append(part)
                    #     single_data.append(answer)
                    answer = {
                        "id": "",
                        "content": "",
                    }
                # if label == 'base':
                #     all_data.append([ref, single_text, single_text])
                # else:
                #     all_data.append([ref, single_data, single_text, find_text])
                all_data.append([ref, single_text, single_text])
                ref, single_data, single_text, find_text = [], [], [], []
            except Exception as e:
                # print("发生错误:", e)
                break
            break
    model = Bert4Vec(model_name_or_path='/home/pre_model/paraphrase-multilingual-MiniLM-L12-v2')
    # model = Bert4Vec(mode='simbert-base', model_name_or_path='WangZeJun/simbert-base-chinese')
    # model = Bert4Vec(mode='roformer-sim-base', model_name_or_path='WangZeJun/roformer-sim-base-chinese')
    # model = Bert4Vec(mode='roformer-sim-small', model_name_or_path='WangZeJun/roformer-sim-small-chinese')
    # model = Bert4Vec(mode='paraphrase-multilingual-minilm',model_name_or_path='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    with open(output, mode='w', encoding='utf-8') as w1:
        result = []
        all_fg_lv = 0
        count = 0
        for i, data in enumerate(tqdm(all_data)):
            label_chongfu = False
            mid = ''
            # if len(data[0]) == len(data[2]):
            #     continue
            zhong = ""
            fg = []
            labeled_list = [[0, x] for x in data[0]]
            w1.write(str(data[0]) + '\n\n')
            for j, dataa in enumerate(data[2]):

                if '。' in dataa:
                    query = dataa.split("。")[:-1]
                else:
                    query = [dataa]
                # query = dataa
                while True:
                    try:
                        similarity = model.similarity(data[0], query, return_matrix=True)
                        # similarity函数支持的默认输入参数有：batch_size=64, return_matrix=False
                        # print(similarity)
                    except Exception as e:
                        print("发生错误:", e, "\nnumber:", i, query, data[0])
                        break
                    break
                # for k in range(len(query)):
                for n, da in enumerate(data[0]):
                    for m in range(len(query)):
                        if similarity[n][m] > 0.85:
                            labeled_list[n][0] = 1
                w1.write(str(dataa) + '\n')
                # fg.append([dataa, zhong])
            ref_num, zhong_num = 0, 0
            for i, lab in enumerate(labeled_list):
                ref_num += len(lab[1])
                if lab[0] == 1:
                    zhong += str(lab[1]) + '。'
                    zhong_num += len(lab[1])
            fg_lv = '%.4f' % (zhong_num / ref_num)
            if zhong_num / ref_num > 0:
                count += 1
            all_fg_lv += zhong_num / ref_num
            result.append(fg)
            w1.write("\n/**/" + zhong + '\n')
            w1.write('覆盖率：' + str(fg_lv))
            w1.write('\n\n==========\n\n')
        w1.write('\nuseful_example:' + str(count) + '\n' + str('%.4f' % (all_fg_lv / count)))


def split_string_by_length(input_string, length):
    return [input_string[i:i+length] for i in range(0, len(input_string), length)]


def fugai_ref(input, output):
    from bert4vec import Bert4Vec

    with open(input, mode='r', encoding='utf-8') as f1:
        alls = f1.read().split('/**/')
    all_data = []
    single_data, single_text, find_text = [], [], []
    asd = r'Find\[([^}]*)\]'
    for i, text in enumerate(tqdm(alls)):
        # if i < 50:
        #     continue
        if "Reference" not in text:
            continue
        while True:
            try:
                text = text.replace(" \"[A: ", " {\"A\": \"").replace("  B: ", "\", \"B\": \"") \
                    .replace("]\",", "\"},")
                data = json.loads(json.dumps(eval(text)))
                # ref = data['Reference'].replace("\n", "").split('】').split('。')[:-1]
                ref_ = data['Reference'].replace("\n", "").split('】')
                if len(ref_) > 1:
                    if '。' in ref_[1]:
                        ref = ref_[1].split('。')[:-1]
                    else:
                        ref = split_string_by_length(ref_[1], 30)
                else:
                    ref = split_string_by_length(ref_[0], 30)
                answer = {
                    "id": "",
                    "content": "",
                }
                for j, dia in enumerate(data['Multi-round Dialogue Evaluation']):
                    answer['id'] = dia['id']
                    single_text.append(dia['B'])
                    answer = {
                        "id": "",
                        "content": "",
                    }
                all_data.append([ref, single_text, single_text])
                ref, single_data, single_text, find_text = [], [], [], []
            except Exception as e:
                # print("发生错误:", e)
                break
            break
    model = Bert4Vec(model_name_or_path='/home/pre_model/paraphrase-multilingual-MiniLM-L12-v2')
    with open(output, mode='w', encoding='utf-8') as w1:
        result = []
        all_fg_lv = 0
        count = 0
        for i, data in enumerate(tqdm(all_data)):
            # if len(data[0]) == len(data[2]):
            #     continue
            zhong = ""
            fg = []
            labeled_list = [[0, x] for x in data[0]]
            w1.write(str(data[0]) + '\n\n')
            for j, dataa in enumerate(data[2]):
                if '。' in dataa:
                    query = dataa.split("。")[:-1]
                else:
                    query = [dataa]
                # query = dataa
                while True:
                    try:
                        simi = model.similarity(data[0], query, return_matrix=True)
                        # similarity函数支持的默认输入参数有：batch_size=64, return_matrix=False
                        # print(similarity)
                    except Exception as e:
                        print("发生错误:", e, "\nnumber:", i, query, data[0])
                        break
                    break
                # for k in range(len(query)):
                for n, da in enumerate(data[0]):
                    for m in range(len(query)):
                        if simi[n][m] > 0.85:
                            labeled_list[n][0] = 1
                w1.write(str(dataa) + '\n')
                # fg.append([dataa, zhong])
            ref_num, zhong_num = 0, 0
            for i, lab in enumerate(labeled_list):
                ref_num += len(lab[1])
                if lab[0] == 1:
                    zhong += str(lab[1]) + '。'
                    zhong_num += len(lab[1])
            fg_lv = '%.4f' % (zhong_num / ref_num)
            count += 1
            all_fg_lv += zhong_num / ref_num
            result.append(fg)
            w1.write("\n/**/" + zhong + '\n')
            w1.write('覆盖率：' + str(fg_lv))
            w1.write('\n\n==========\n\n')
        w1.write('\nuseful_example:' + str(count) + '\n' + str('%.4f' % (all_fg_lv / count)))


def fugai_eng(input, output):
    from bert4vec import Bert4Vec

    with open(input, mode='r', encoding='utf-8') as f1:
        alls = f1.read().split('/**/')
    all_data = []
    single_data, single_text, find_text = [], [], []
    asd = r'Find\[([^}]*)\]'

    # with open("/home/build_data/squad_v2/squad_v2_train.json", mode='r', encoding='utf-8') as f3:
    #     all_list = json.load(f3)['data']
    #     doc_list = []
    #     for i, single in enumerate(all_list):
    #         doc = single['title'] + '.'
    #         content = ""
    #         for j, para in enumerate(single['paragraphs']):
    #             content += para['context'].encode('ASCII', 'ignore').decode(
    #                 'ASCII')  # 解决Python print 输出文本显示 gbk 编码错误问题
    #             # print(content)
    #             if len(content) > 800:
    #                 doc_list.append(doc + content)
    #                 content = ""
    #                 continue
    #     print("eng data length:", len(doc_list))
    # data_refs = doc_list

    for i, text in enumerate(tqdm(alls)):
        # if i < 50:
        #     continue
        if "Reference" not in text:
            continue
        while True:
            try:
                # text = text.replace(" \"[A: ", " {\"A\": \"").replace("  B: ", "\", \"B\": \"").replace("]\",", "\"},")
                text = text.replace(" \"[A: ", " {\"A\": \"").replace(" B: ", "\", \"B\": \"").replace("]\",", "\"},")
                data = json.loads(json.dumps(eval(text)))
                # ref = data['Reference'].replace("\n", "").split('】').split('。')[:-1]
                # ref_ = data['Reference'].replace("\n", "").split('】')
                # if len(ref_) > 1:
                #     if '。' in ref_[1]:
                #         ref = ref_[1].split('。')[:-1]
                #     else:
                #         ref = split_string_by_length(ref_[1], 30)
                # else:
                #     ref = split_string_by_length(ref_[0], 30)
                ref_ = data['Reference'].replace("\n", "")
                # ref_ = data_refs[i]
                if len(ref_.split(". ")) < 5:
                    ref = split_string_by_length(ref_, 30)
                else:
                    ref = ref_.split(". ")
                answer = {
                    "id": "",
                    "content": "",
                }
                for j, dia in enumerate(data['Multi-round Dialogue Content']):
                    answer['id'] = dia['id']
                    single_text.append(dia['A'])
                    single_text.append(dia['B'])
                    answer = {
                        "id": "",
                        "content": "",
                    }
                all_data.append([ref, single_text, single_text])
                ref, single_data, single_text, find_text = [], [], [], []
            except Exception as e:
                # print("发生错误:", e)
                break
            break
    model = Bert4Vec(model_name_or_path='/home/pre_model/paraphrase-multilingual-MiniLM-L12-v2')
    with open(output, mode='w', encoding='utf-8') as w1:
        result = []
        all_fg_lv = 0
        count = 0
        for i, data in enumerate(tqdm(all_data)):
            # if len(data[0]) == len(data[2]):
            #     continue
            zhong = ""
            fg = []
            labeled_list = [[0, x] for x in data[0]]
            w1.write(str(data[0]) + '\n\n')
            for j, dataa in enumerate(data[2]):
                if '. ' in dataa:
                    query = dataa.split(". ")[:-1]
                else:
                    query = [dataa]
                # query = dataa
                while True:
                    try:
                        simi = model.similarity(data[0], query, return_matrix=True)
                        # similarity函数支持的默认输入参数有：batch_size=64, return_matrix=False
                        # print(similarity)
                    except Exception as e:
                        print("发生错误:", e, "\nnumber:", i, query, data[0])
                        break
                    break
                # for k in range(len(query)):
                for n, da in enumerate(data[0]):
                    for m in range(len(query)):
                        if simi[n][m] > 0.75:
                            labeled_list[n][0] = 1
                w1.write(str(dataa) + '\n')
                # fg.append([dataa, zhong])
            ref_num, zhong_num = 0, 0
            for i, lab in enumerate(labeled_list):
                ref_num += len(lab[1])
                if lab[0] == 1:
                    zhong += str(lab[1]) + '。'
                    zhong_num += len(lab[1])
            fg_lv = '%.4f' % (zhong_num / ref_num)
            count += 1
            all_fg_lv += zhong_num / ref_num
            result.append(fg)
            w1.write("\n/**/" + zhong + '\n')
            w1.write('覆盖率：' + str(fg_lv))
            w1.write('\n\n==========\n\n')
        w1.write('\nuseful_example:' + str(count) + '\n' + str('%.4f' % (all_fg_lv / count)))


if __name__ == '__main__':
    # xgx()
    doc2dia_paht = "/home/build_data/more/gpt_cot_eng_300.json"   # "/home/build_data/refGPT/doc2dia_ref_my(500).json"
    output_path = "/home/build_data/more/fg75_gpt_cot_eng_300.json"
    # fugai_ww(input=doc2dia_paht, output=output_path, label='my1')
    # fugai_ref(input=doc2dia_paht, output=output_path)
    fugai_eng(input=doc2dia_paht, output=output_path)