from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
import openai # For GPT-3.5 and GPT-4 API ...
from openai import OpenAI # For open LLM API ...
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import torchtext
import re
import random
import time
import datetime
import pandas as pd

# https://review-of-my-life.blogspot.com/2017/11/python-dict-shuffle.html
def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)
  
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y%m%d_%H_%M_%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input, max_length, i, k):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)
    time.sleep(args.api_time_interval)
    
    # https://beta.openai.com/account/api-keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #print(openai.api_key)
    
    # Specify engine ...
    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    else:
        raise ValueError("model is not properly defined ...")
        
    response = openai.Completion.create(
      engine=engine,
      prompt=input,
      max_tokens=max_length,
      temperature=0,
      stop=None
    )
    
    return response["choices"][0]["text"]

def decoder_for_gpt4(args, input, max_length, i, k):
    
    time.sleep(args.api_time_interval)

    openai.api_type = "azure"
    openai.api_base = "" # fill your API base at here
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "" # fill your API key at here
    deployment_name = "gpt-35-turbo-0613" # gpt-35-turbo-0613, gpt-4

    if args.model == "gpt-4":
        deployment_name = "gpt-4"
    elif args.model == "chatgpt":
        deployment_name = "gpt-35-turbo-0613"
    else:
        raise ValueError("model is not properly defined ...")
    
    print("Model name:", deployment_name)

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[{"role": "system",
                    "content": "You are" + deployment_name + ", a large language model trained by OpenAI.\nCurrent date: 2024-02-22"},
                    {"role": "user", "content": input}],
            temperature=0.0,
            max_tokens=max_length,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)
    except:
        return ".........."

    if "content" in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return ".........."

def decoder_for_openLLM(args, input, max_length, i, k):
    # https://docs.llama-api.com/quickstart
    
    time.sleep(args.api_time_interval)

    client = OpenAI(
        api_key = "", # fill your API key at here
        base_url = "https://api.llama-api.com"
    )
    print("Model name:", args.model)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Assistant is a large language model."},
                {"role": "user", "content": input}
            ]

        )
        result = response.choices[0].message.content
    except:
        return ".........."

    #print(response)
    #print(response.model_dump_json(indent=2))
    #print(response.choices[0].message.content)
    return result

class Decoder():
    def __init__(self, args):
        print_now()
 
    def decode(self, args, input, max_length, i, k):
        #response = decoder_for_gpt3(args, input, max_length, i, k)
        if args.model in ["chatgpt", "gpt-4"]:
            response = decoder_for_gpt4(args, input, max_length, i, k)
        elif args.model in ["llama3-8b", "llama3-70b", "mistral-7b", "gemma-7b", "gemma-2b", "Qwen1.5-72B-Chat"]:
            response = decoder_for_openLLM(args, input, max_length, i, k)
        else:
            response = ".........."
        return response

def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset == "commonsensqa":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(args.dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "svamp":
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)            
          
    elif args.dataset in ("coin_flip", "last_letters"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html

    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args)
    
    # dataloader = torch.utils.data.DataLoader(dataset,
    #               shuffle=True,
    #               batch_size=args.minibatch_size,
    #               drop_last=False,
    #               num_workers=dataloader_num_workers,
    #               worker_init_fn=seed_worker,
    #               generator=g,
    #               pin_memory=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=False,
                  batch_size=args.minibatch_size,
                  num_workers=dataloader_num_workers)
    return dataloader

# ver 0.2
def answer_cleansing(args, pred):

    print("pred_before : " + pred)
    
    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot", "zero_shot_cp", "few_shot_cp", "few_shot_cot_cp"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred

def create_demo_text(args, cot_flag):
    x, z, y = [], [], []
    
    # example sentences ...    
    if args.dataset in ("multiarith", "gsm8k", "svamp"):
        
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")

    elif args.dataset in ("aqua"):
        
        x.append("John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64")
        z.append("If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.")
        y.append("A")

        x.append("If a / b = 3/4 and 8a + 5b = 22, then find the value of a. Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append("If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")

        x.append("A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append("The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.")
        y.append("E")

        x.append("How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append("There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.")
        y.append("B")

    elif args.dataset in ("strategyqa"):
        
        x.append("Do hamsters provide food for any animals?")
        z.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
        y.append("yes")

        x.append("Could Brooke Shields succeed at University of Pennsylvania?")
        z.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
        y.append("yes")

        x.append("Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?")
        z.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5.")
        y.append("no")

        x.append("Yes or no: Is it common to see frost during some college commencements?")
        z.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
        y.append("yes")

        x.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
        z.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
        y.append("no")

        x.append("Yes or no: Would a pear sink in water?")
        z.append("The density of a pear is about 0.6g/cm3 , which is less than water. Objects less dense than water float. Thus, a pear would float.")
        y.append("no")

    else:
        raise ValueError("dataset is not properly defined ...")
        
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text
