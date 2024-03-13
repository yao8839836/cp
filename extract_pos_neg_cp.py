import os
import numpy as np

file_list = os.listdir("./log/")
# print(file_list)

all_examples = []

for f_name in file_list:

    if f_name.find("zero_shot_cp_gpt-4") != -1:
        
        with open("./log/" + f_name, "r", encoding="gbk") as f:
            all_str = f.read()

        if all_str.find("Let's give a correct and a wrong answer.") == -1:
            continue
        if all_str.find("Let's also think step by step for the correct and the wrong answer") != -1:
            continue
        print(f_name)
        qa_pairs = all_str.split("INFO:root:Q")
        print(len(qa_pairs))

        pos_examples = []
        neg_examples = []

        for qa_pair in qa_pairs[1:]:
            end_idx = qa_pair.find("INFO:root:*************************")
            qa_pair = qa_pair[:end_idx].strip()

            pred_begin_idx = qa_pair.find("INFO:root:pred :")
            pred_end_idx = qa_pair.rfind("\n")
            pred = qa_pair[pred_begin_idx + 16: pred_end_idx].strip()
            
            gt_begin_idx = qa_pair.find("INFO:root:GT :")
            gt = qa_pair[gt_begin_idx + 14:].strip()

            qa_pair = "Q" + qa_pair 
            # print(qa_pair)
            # print(pred, gt)
            if pred == gt:
                pos_examples.append(qa_pair)
            else:
                neg_examples.append(qa_pair)

        sample_pos_examples = np.random.choice(pos_examples, 10)
        sample_neg_examples = np.random.choice(neg_examples, 10)

        for x in sample_pos_examples:
            all_examples.append(x)
        for x in sample_neg_examples:
            all_examples.append(x)

with open("results/zero_shot_cp_gpt4_240_examples.txt", "w", encoding="gbk") as f:
    f.write("\n\n\n\n".join(all_examples))