# coding=utf-8
import argparse
import multiprocessing as mp
import traceback
import json
import os
import time

import jsonlines as jsonlines
from func_timeout import func_set_timeout
from openai import OpenAI
from tqdm import tqdm
import requests

system = ""
os.environ[
    "MIT_SPIDER_TOKEN"] = ""
os.environ["M6_TENANT"] = ""
os.environ["MIT_SPIDER_URL"] = ""
MAX_API_RETRY = 3
LLM_MIT_RETRY_SLEEP = 5


USE_MIT = True
# os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


def mit_openai_api(**kwargs):
    tenant = None
    if USE_MIT:
        if not os.environ.get('MIT_SPIDER_TOKEN', None):
            print("NO MIT_SPIDER_TOKEN FOUND，please set export MIT_SPIDER_TOKEN=<YOUR TOKEN>")
        if not os.environ.get('MIT_SPIDER_URL', None):
            print("NO MIT_SPIDER_URL FOUND，please set export MIT_SPIDER_URL=<YOUR URL>")
        mit_spider_config = {
            "url": os.environ.get("MIT_SPIDER_URL", None),
            "header": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('MIT_SPIDER_TOKEN', None)}"
            }
        }
        if kwargs['model'].startswith('gpt-4') and os.environ.get("M6_TENANT", None):
            tenant = os.environ.get("M6_TENANT")
    else:
        if not os.environ.get('OPENAI_API_KEY', None):
            print("NO OPENAI_API_KEY FOUND，please set export OPENAI_API_KEY=<YOUR TOKEN>")
        if not os.environ.get('OPENAI_API_BASE', None):
            print("NO OPENAI_API_BASE FOUND，please set export OPENAI_API_BASE=<YOUR URL>")
        mit_spider_config = {
            "url": os.environ.get("OPENAI_API_BASE", None),
            "header": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', None)}"
            }
        }
    response = None
    for i in range(MAX_API_RETRY):
        try:
            if tenant:
                payload = {'tenant': tenant}
            else:
                payload = dict()
            for k, w in kwargs.items():
                payload[f"{k}"] = w
            response = requests.post(mit_spider_config['url'], json=payload, headers=mit_spider_config['header']).json()
        except Exception as e:
            print(response, e)
            time.sleep(LLM_MIT_RETRY_SLEEP)
            continue
        if USE_MIT and response['code'] == 200:
            return response
        elif not USE_MIT and "choices" in response:
            return response
        else:
            time.sleep(LLM_MIT_RETRY_SLEEP)
            print(response)

    return None


@func_set_timeout(520000)
def get_result_by_request(**kwargs):
    response = mit_openai_api(**kwargs)
    if USE_MIT and response['code'] == 200:
        bc = response['data']['response'].get('choices')
        result = bc[0]["message"]["content"]
        return result
    elif not USE_MIT and "choices" in response:
        result = response["choices"][0]["message"]["content"]
        return result
    else:
        raise Exception(response['messages']) if USE_MIT else Exception(response['message'])


def read_jsonl_file(file_name, max_sentence=None):
    data = []
    with jsonlines.open(file_name, "r") as r:
        for i, obj in enumerate(r):
            if max_sentence is not None and i >= max_sentence:
                return data
            data.append(obj)
    return data


class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args):
        return mp.get_logger().error(msg, *args)

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def doc2dia_task(objs, prompt, worker_id=0, check_func=None, output_path=""):
    # output_objs = []
    with open(f"{output_path}.worker-{worker_id}", mode='w', encoding='utf-8') as w:
        for i, dia in enumerate(tqdm(objs, position=worker_id, desc=f"Worker {worker_id}")):
            if check_func is not None and check_func():
                continue
            while True:
                try:
                    content = prompt + dia + '\n\nOutput:\n'
                    # print(len(prompt), len(dia), len(content))
                    if len(content) > 25000:
                        print(content)
                        break
                    # if len(prompt.split()) < 2048:
                    #     #model = "gpt-4-32k-0613"
                    #     model = "gpt-4-1106-preview"
                    # else:
                    # model = "gpt-4-1106-preview"
                    model = args.model_name
                    # model = "gpt-4-32k-0613"
                    # if len(content) < 2048:
                    #     model = "gpt-4-1106-preview"
                    # else:
                    #     model = "gpt-4-1106-preview"
                    # model = "gpt-3.5-turbo-0125"
                    openai_args = {
                        "model": model,
                        # model='gpt-3.5-turbo-0613',  gpt-3.5-turbo-16k-0613 model='gpt-4' gpt-3.5-turbo-16k
                        "temperature": 0,
                        # "max_tokens": 4096,
                        "messages": [
                            {'role': 'system', 'content': "You are a helpful assistant."},
                            {'role': 'user', 'content': content},
                        ]
                    }
                    answer = get_result_by_request(**openai_args)
                    if i > 0:
                        w.write("\n/**/" + answer.replace("```", "").replace("json\n", ""))
                    else:
                        w.write("/**/" + answer.replace("```", "").replace("json\n", ""))
                    w.flush()
                    time.sleep(5)
                    break
                except Exception as e:
                    # output_obj["messages"][2]["content"] = ""
                    print("Request Error:", e)
                    time.sleep(5)
            # output_objs.append(output_obj)
    print(f"worker {worker_id} finished...")
    # return output_objs
    return f"worker {worker_id} finished..."


def multi_tasks(objs, base_prompt, workers=10, path="/cpfs01/shared/Group-m6/yj411294/data/system_role/log_gpt.jsonl",
                task=None):
    p = mp.Pool(workers)
    lens = len(objs)
    chunk_size = lens // workers
    results = []
    for worker_id in range(workers):
        results.append(p.apply_async(MPLogExceptions(task), args=(
        objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], base_prompt, worker_id, None, path)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    # write_jsonl_file(output_objs, path)
    return output_objs


def write_jsonl_file(objs, path):
    with jsonlines.open(path, "w") as w:
        for obj in objs:
            w.write(obj)
    print(f"Successfully saving to {path}: {len(objs)}")


def parse_args():
    parser = argparse.ArgumentParser(description="llm gen")
    parser.add_argument("-n", "--workers", type=int, default=1)
    parser.add_argument("-m", "--model-name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--uuid", type=str, default='uuid')
    args = parser.parse_args()
    return args


def load_cached_objs(output_path):
    result_name = os.path.basename(output_path)
    root_dir = os.path.dirname(output_path)
    file_names = [f for f in os.listdir(root_dir) if f.startswith(f"{result_name}.")]
    cached_objs = {}
    for file_name in file_names:
        objs = read_jsonl_file(f"{root_dir}/{file_name}")
        for obj in objs:
            cached_objs[obj["id"]] = obj
    return cached_objs


def dia2eval(input_path, doc2dia_prompt_path, output_path, test, newdata, workers=32, max_sentence=-1):
    # merge(path=output_path, output_path=output_path, worker=workers)
    # exit()
    # _objs = read_jsonl_file(input_path)
    data1, _objs = [], []
    with open(input_path, mode='r', encoding='utf-8') as f1:
        if test == 1:
            a = f1.readlines()
            if newdata:
                # for i, exam in enumerate(a):
                #     zxc = json.loads(exam)
                #     reff = zxc["messages"][0]['content'].split('{')[1].strip().split('}')[0]
                #     final = zxc['output'].replace("Character Information", "Reference\": \"" + reff + '\",\"Character Information')
                #     data1.append(final)
                nmp = 0
                for i, exam in enumerate(a):
                    zxc = json.loads(exam)
                    reff = zxc["messages"][0]['content'].split('{')[1].strip().split('}')[0]
                    final = zxc['raw_generation'][0].replace("Character Information", "Reference\": \"" + reff + '\",\"Character Information')
                    if "Question:" and "Answer:" in final:
                        continue
                    else:
                        data1.append(final)
            else:
                for i, exam in enumerate(a):
                    zxc = json.loads(exam)
                    ref_source = zxc["messages"][0]['content'].split('{')[1].strip().split('}')[0]
                    muti_dia = zxc["messages"][1]['content'].replace("Character Information", "Reference\": \"" + ref_source + '\",\"Character Information')
                    data1.append(muti_dia)
            data = data1[2000:]  # [0-615]  [615-1570]  [1570-]
        else:
            a = f1.read()
            data = a.split('/**/')

    if max_sentence > 0:
        _objs = data[:max_sentence]

    print("Total amount of data:", len(_objs))

    prompt = ""
    with open(doc2dia_prompt_path, mode='r', encoding='gbk') as f2:  # squad exp 10:30
        prompt = f2.read()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(prompt)
    output_objs = multi_tasks(_objs, base_prompt=prompt, workers=workers, path=output_path,
                              task=doc2dia_task)
    print(''.join(output_objs))
    time.sleep(5)
    merge(path=output_path, output_path=output_path, worker=workers)  # 合并文件


def convert_log2chatml(input_path="/cpfs01/shared/Group-m6/lukeming.lkm/sft_data/badcase_after_sep_code.jsonl"):
    _objs = read_jsonl_file(input_path)


def merge(path, output_path, worker=1):
    # 创建一个空列表，用于存储所有的JSON数据
    count = 0

    with open(output_path, "w", encoding='utf-8') as file2:
        # 遍历每个JSON文件
        lab = 0
        for worker_id in range(0, worker):
            filename = f"{path}.worker-{worker_id}"
            with open(filename, "r", encoding='utf=8') as file:
                data_i = file.read().split('/**/')[1:]
                for i, doc in enumerate(data_i):
                    # al = json.loads(doc)  # 读取JSON数据
                    # json.dump(doc, file2, ensure_ascii=False)
                    while doc.endswith("\n"):
                        doc = doc.rstrip("\n")
                    if lab > 0:
                        file2.write('\n/**/' + doc)
                    else:
                        file2.write(doc)
                    lab = 1
                    count += 1
            os.remove(filename)
    os.rename(output_path, args.output_path)

    print("merge data length:", str(count))

    # 将合并后的数据写入新的JSON文件
    # with open(output_path, "w", encoding='utf-8') as file:
    #     json.dump(merged_data, file, ensure_ascii=False)


if __name__ == "__main__":
    # 基本参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default="/home/build_data/dia/qwen_cod_gllm_ref_952.json", help='')
    parser.add_argument('--prompt', type=str, default="/home/SDXX/build_data/use/chinese_data/dia_chinese_eval_prompt.json", help='')  # new_eval_prompt.json  chinese_eval_prompt.json
    parser.add_argument('--output_path', type=str, default="/home/build_data/dia/eval/eval_qwen_gllm_ref.json", help='')
    parser.add_argument('--model_name', type=str, default='gpt-4-0125-preview', help='')
    parser.add_argument('--mid_path', type=str, default="/home/build_data/mid_dia_cod_zh.json", help='')
    parser.add_argument('--name', type=int, default=6, help='')
    # gpt-3.5-turbo-0125 gpt-4-1106-preview gpt-3.5-turbo-16k   new: gpt-4-0125-preview
    args = parser.parse_args()
    dia2eval(
        input_path=args.source_path,
        doc2dia_prompt_path=args.prompt,
        output_path=args.mid_path,
        test=0,
        newdata=True,
        workers=64,
        max_sentence=10000
    )
    time.sleep(3)
