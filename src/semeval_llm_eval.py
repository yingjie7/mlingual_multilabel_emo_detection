import argparse
import json
import re
import traceback 
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig, MT5ForConditionalGeneration, MT5Tokenizer
from trl import set_seed as trl_seed
from trl.models.utils import FORMAT_MAPPING
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import pipeline, set_seed as transf_seed
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np 
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
import random, os, glob
from lightning import seed_everything 

from reformat_data_ft_llm import semeval_process, semeval_split_data
import sys, pandas as pd 

sys.path.append('libs/SemEval2025-Task11-Evaluation/')
from reformat_data_ft_llm import json_to_dataframe, json_to_dataframe_taskb, pairwise_prompting_json_to_dataframe
from utils import evaluate as semeval11_evaluate
 
 
if __name__=="__main__":
    path_result = '/home/s2220429/per_erc/finetuned_llm/semeval_track_a/alllanguage_Qwen2.5-32B-Instruct_ep6_step-1_lrs-linear0.0003_r32_default_ED600_idRun-/result_swe_valid_step-2748.json'
    file_result_name = path_result.split("/")[-1]
    d_type = file_result_name.split("_")[-2]
    folder_result = path_result[:-len(file_result_name)]+ f"/{d_type}/"
    all_data_eval = json.load(open(path_result))
    
    c_filter = "swe"
    org_all_data_eval = all_data_eval
    new_detail_pred = [(x,y) for x, y in zip(all_data_eval['detail_pred'], all_data_eval['all_ids']) if c_filter not in y]
    all_data_eval['detail_pred'] = [e[0] for e in new_detail_pred]
    all_data_eval['all_ids'] = [e[1] for e in new_detail_pred]
    
    # for i, e in enumerate(all_data_eval['detail_pred']):
    #     pred = e[2].split("<｜Assistant｜>")[1].split("<｜end▁of▁sentence｜>")[0]
    #     all_data_eval['detail_pred'][i][0] = pred
    
    task_id = 'a' if 'track_a' in path_result else 'b'
    publicvalid_folder = f"data/semeval11/{d_type}/track_{task_id}/"
    all_e_checks = [None]*len(all_data_eval['detail_pred'])
    
    if '_default_' in path_result:
        process_fn_json_to_dataframe = json_to_dataframe if task_id == 'a' else json_to_dataframe_taskb
    elif '_pairwise_' in path_result:
        process_fn_json_to_dataframe = pairwise_prompting_json_to_dataframe
        if task_id == "b":
            # the intensity of the emotion joy expressed in it?
            all_e_checks = [e[2].split("the intensity of the emotion ")[1].split(" expressed in it?")[0] for e in all_data_eval['detail_pred']]
        else:
            # of the emotion joy expressed in it?
            all_e_checks = [e[2].split(", is the emotion ")[1].split(" expressed in it?")[0] for e in all_data_eval['detail_pred']]
    all_data_eval['e_check'] = all_e_checks
    label_str = "anger, disgust, fear, joy, sadness, surprise".replace(" ", "")
    group_data_eval  = {'alllanguage': all_data_eval}
    for e, e_id, e_check in zip(all_data_eval['detail_pred'], all_data_eval['all_ids'], all_e_checks):
        lang_id = e_id.split("_")[0]
        if lang_id not in group_data_eval:
            group_data_eval[lang_id] = {"detail_pred": [], "all_ids": [], 'e_check':[]}
        group_data_eval[lang_id]['detail_pred'].append(e)
        group_data_eval[lang_id]['all_ids'].append(e_id)
        group_data_eval[lang_id]['e_check'].append(e_check)
        group_data_eval[lang_id]['labels_set'] = ['id'] + list(pd.read_csv(f"{publicvalid_folder}/{lang_id}.csv").columns[2:])
        
    all_results = []
    for lang_id, data_eval in group_data_eval.items():
        df_pred = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=0, all_ids=data_eval['all_ids'], all_e_checks=data_eval['e_check'])
        if lang_id!="alllanguage":
            df_pred = df_pred[data_eval['labels_set']]
        pred_lines = df_pred.to_csv(sep=',', index=False).strip('\n').split('\n')
        
        if lang_id!="alllanguage":
            # df_pred['id'] = data_eval['all_ids']
            print(f"- Write file {folder_result}/track_{task_id}/pred_{lang_id}.csv")
            if not os.path.exists(f"{folder_result}/track_{task_id}/"):
                os.makedirs(f"{folder_result}/track_{task_id}/")
            df_pred.to_csv(f"{folder_result}/track_{task_id}/pred_{lang_id}.csv", sep=',', index=False) 
        
        df_gold = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=1, all_ids=data_eval['all_ids'], all_e_checks=data_eval['e_check'])
        if lang_id!="alllanguage":
            df_gold = df_gold[data_eval['labels_set']]
        gold_lines = df_gold.to_csv(sep=',', index=False).strip('\n').split('\n')
        
        scores = semeval11_evaluate(gold_lines, pred_lines, task=task_id)
        eval_f1_score = scores['macro']['f1'] if task_id=="a" else sum(scores.values()) / len(scores) 
        
        if task_id=="b":
            scores = dict([(k,float(v)) for k, v in scores.items() ])
        
        all_results.append({
            'lang_id': lang_id,
            'official_score': eval_f1_score,
            'scores': scores
        })
        print(lang_id, eval_f1_score)
        print(lang_id, scores)
    json.dump(all_results, open(f"{folder_result}/track_{task_id}/log_result_all.json", "wt"), indent=1)
