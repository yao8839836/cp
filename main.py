import argparse
import logging
import torch
import random
import time
import datetime
import os
from utils import *

os.environ["WANDB_DISABLED"] = "true"

def main():
    args = parse_arguments()
    now = print_now(return_flag=1)
    logging.basicConfig(filename=args.log_dir + args.dataset + "_" + args.method + "_" + args.model + "_" + now + ".log", level=logging.INFO)
    print('*****************************')
    logging.info('*****************************')
    print(args)
    logging.info(args)
    print('*****************************')
    logging.info('*****************************')
    
    fix_seed(args.random_seed)
    
    #print("OPENAI_API_KEY:")
    #print(os.getenv("OPENAI_API_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
    elif args.method == "few_shot_cp":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot_cp":
        demo = create_demo_text(args, cot_flag=True)        
    else:
        pass
    
    total = 0
    correct_list = []        
    for i, data in enumerate(dataloader):

        print('*************************')
        logging.info('*************************')
        print("{}st data".format(i+1))
        logging.info("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        
        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "zero_shot_cp":
            x = x + " " + args.cp_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            x = demo + x
        elif args.method == "few_shot_cp":
            x = demo + x + " " + args.cp_trigger
        elif args.method == "few_shot_cot_cp":
            x = demo + x + " " + args.cp_trigger
        else:
            raise ValueError("method is not properly defined ...")
        
        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z = decoder.decode(args, x, max_length, i, 1)

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            print(z2 + " " + pred)
            logging.info(z2 + " " + pred)

        elif args.method == "zero_shot_cp" or args.method == "few_shot_cp" or args.method == "few_shot_cot_cp":
            z2 = x + " " + z + " " + args.direct_answer_trigger_cp
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)
            print(z2 + " " + pred)
            logging.info(z2 + " " + pred)

        else:
            pred = z
            print(x + " " + pred)
            logging.info(x + " " + pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        logging.info("pred : {}".format(pred))
        print("GT : " + y)
        logging.info("GT : " + y)
        print('*************************')
        logging.info('*************************')
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    logging.info("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CP")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=1, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt-4", choices=["chatgpt", "gpt-4", "llama3-8b", "llama3-70b", "mistral-7b", "gemma-7b", "gemma-2b", "Qwen1.5-72B-Chat"], help="model used for decoding."
        #"gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl",  Note that 'gpt3' are the smallest models.
        # Qwen1.5-72B-Chat ( replace 72B with 32B / 14B / 7B / 4B / 1.8B / 0.5B)
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cp", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "zero_shot_cp", "few_shot_cp", "few_shot_cot_cp"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--cp_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute contrastive prompting"
    )    
    parser.add_argument(
        "--max_length_cot", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_cp", type=int, default=512, help="maximum length of output tokens by model for reasoning extraction"
    )    
    parser.add_argument(
        "--max_length_direct", type=int, default=512, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=0.1, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.direct_answer_trigger_cp = "\nTherefore, among A through E, the correct answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.direct_answer_trigger_cp = "\nTherefore, among A through E, the correct answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (Yes or No) is"        
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (arabic numerals) is"        
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (arabic numerals) is"        
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
        args.direct_answer_trigger_cp = "\nTherefore, among A through F, the correct answer is"        
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
        args.direct_answer_trigger_cp = "\nTherefore, among A through C, the correct answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
        args.direct_answer_trigger_cp = "\nTherefore, the correct answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")


    if args.cp_trigger_no == 1:
        args.cp_trigger = "Let's give a correct and a wrong answer."
    elif args.cp_trigger_no == 2:
        args.cp_trigger = "Please give a correct and a wrong answer."
    elif args.cp_trigger_no == 3:
        args.cp_trigger = "Let's give a correct and a wrong answer. Let's also think step by step for the correct and the wrong answer."
    elif args.cp_trigger_no == 4:
        args.cp_trigger = "Let's think step by step and give both a correct answer and a wrong answer."
    elif args.cp_trigger_no == 5:
        args.cp_trigger = "Let's first give a wrong answer, then give the correct answer." 
    elif args.cp_trigger_no == 6:
        args.cp_trigger = "Let's first give the correct answer, then give a wrong answer."
    elif args.cp_trigger_no == 7:
        args.cp_trigger = "Let's give a correct and an incorrect answer."
    elif args.cp_trigger_no == 8:
        args.cp_trigger = "Let's give a correct and two wrong answers."
    elif args.cp_trigger_no == 9:
        args.cp_trigger = "Let's give a correct and three wrong answers."
    elif args.cp_trigger_no == 10:
        args.cp_trigger = "Let's give a correct and four wrong answers."
    elif args.cp_trigger_no == 11:
        args.cp_trigger = "Let's give a correct answer."                                               
    else:
        raise ValueError("cp_trigger_no is not properly defined ...")

    return args

if __name__ == "__main__":
    main()