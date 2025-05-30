
import glob
import json
import os
import random
import re
from sentence_transformers import SentenceTransformer
import torch, numpy as np
from tqdm import tqdm
import pandas as pd

def semeval_label_id_to_string(row, all_e_labels, task_id):
    
    gold_lb = ""
    for e_check in all_e_labels:
        if task_id=="a":
            if row[e_check] == 1:
                gold_lb = e_check if len(gold_lb) == 0 else gold_lb+", " + e_check
        elif task_id=="b":
            if row[e_check] == 1:
                gold_lb = f"low degree of {e_check}" if len(gold_lb) == 0 else gold_lb+", " + f"low degree of {e_check}"
            elif row[e_check] == 2:
                gold_lb = f"moderate degree of {e_check}" if len(gold_lb) == 0 else gold_lb+", " + f"moderate degree of {e_check}"
            elif row[e_check] == 3:
                gold_lb = f"high degree of {e_check}" if len(gold_lb) == 0 else gold_lb+", " + f"high degree of {e_check}"
 
    if len(gold_lb) == 0:
        gold_lb = "<none>"
        
    return gold_lb

def semeval_default_prompting(org_data, args):
    new_format = []
    all_e_labels = [str(e) for e in org_data.columns[2:]]
    label_str = "{" + (", ".join(all_e_labels)) + "}"
    for index, row in org_data.iterrows():
        task_id = "a" if "track_a" in args.data_name else "b" if "track_b" in args.data_name else None 
        if task_id is None:
            raise Exception(f"{args.data_name} does not contain track information: 'track_a' or 'track_b'")
        gold_lb = semeval_label_id_to_string(row, all_e_labels, task_id=task_id) 
                
        system_msg = f'You are an expert in analyzing the emotions expressed in a natural sentence. The emotional label set includes {label_str}. Each sentence may have one or more emotional labels, or none at all.'
        if "track_b" in args.data_name:
            system_msg = f'You are an expert in analyzing the emotions expressed in a natural sentence. The emotional label set includes {label_str}, with three levels of intensity: low, moderate, and high. Each sentence may have one or more emotional labels, or none at all.'
        q_msg = f"Given the sentence: \"{row['text']}\", which emotions and their corresponding intensities are expressed in it?"
        new_format.append({
            "id":  row['id'],
            "e_check":  all_e_labels,
            "messages":  [
                {'role': "system", 'content': system_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': gold_lb},
            ]
        })   
    return new_format

def semeval_pairwise_prompting(org_data, args):
    new_format = []
    all_e_labels = [str(e) for e in org_data.columns[2:]]
    label_str = "{" + (", ".join(all_e_labels)) + "}"
    for index, row in org_data.iterrows():
        task_id = "a" if ("track_a" in args.data_name or "track_c" in args.data_name) else "b" if "track_b" in args.data_name else None 
        if task_id is None:
            raise Exception(f"{args.data_name} does not contain track information: 'track_a' or 'track_b'")
        
        system_msg = f'You are an expert in analyzing the emotions expressed in a natural sentence. The emotional label set includes {label_str}. Each sentence may have one or more emotional labels, or none at all.'
        if task_id=="b":
            system_msg = f'You are an expert in analyzing the emotions expressed in a natural sentence. The emotional label set includes {label_str}, with three levels of intensity: low, moderate, and high. Each sentence may have one or more emotional labels, or none at all.'

        gold_lb = ""
        for e_check in all_e_labels:
            if task_id=="a":
                q_msg = f"Given the sentence: \"{row['text']}\", is the emotion {e_check} expressed in it?"
                if row[e_check] == 1:
                    gold_lb = "Yes"
                else:
                    gold_lb = "No"
            elif task_id=="b":
                q_msg = f"Given the sentence: \"{row['text']}\", what is the intensity of the emotion {e_check} expressed in it?"
                if row[e_check] == 1:
                    gold_lb = f"low"
                elif row[e_check] == 2:
                    gold_lb = f"moderate"
                elif row[e_check] == 3:
                    gold_lb = f"high"
                elif row[e_check] == 0:
                    gold_lb = "none"
                    
            new_format.append({
                "id":  row['id'],
                "e_check":  e_check,
                "messages":  [
                    {'role': "system", 'content': system_msg},
                    {'role': "user", 'content': q_msg},
                    {'role': "assistant", 'content': gold_lb},
                ]
            })   
    return new_format

def pairwise_prompting_json_to_dataframe(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", 
                                         prefixid="eng_dev_track_a_", idx_convert=0, 
                                         all_ids=None, all_e_checks=None):
    e_labels = e_labels_str.split(",")
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
    mapping_values = {} 
    for e_id, e_check, e_pred in zip(all_ids, all_e_checks, pred_data_['detail_pred']):
        if e_id not in mapping_values:
            mapping_values[e_id] = {}
        
    mapping_values = dict([((e_id, e_check), e_pred) for e_id, e_check, e_pred in zip(all_ids, all_e_checks, pred_data_['detail_pred'])])
    checking_set = set()
    for e_id, e_check, e_preds in zip(all_ids, all_e_checks, pred_data_['detail_pred']):
        e_pred = e_preds[idx_convert]
        if e_id not in checking_set:
            data_submit['id'].append(e_id)
            checking_set.add(e_id)
        value = 0 if (e_pred == "none" or e_pred == "No") \
            else 1 if (e_pred == "low" or e_pred == "Yes") \
            else 2 if e_pred == "moderate"  \
            else 3 if e_pred == "high" else 0
        data_submit[e_check].append(value)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit

def json_to_dataframe(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", prefixid="eng_dev_track_a_", idx_convert=0, all_ids=None,  **kwargs):
    e_labels = e_labels_str.split(",")
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
        
    for idx, e in enumerate(pred_data_['detail_pred']):
        data_submit ['id'].append(prefixid + "{0:05d}".format(idx+1) if all_ids==None else all_ids[idx])
        for e_check in e_labels:
            data_submit[e_check].append(1 if e_check in e[idx_convert] else 0)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit

def json_to_dataframe_taskb(pred_data_, e_labels_str ="anger,fear,joy,sadness,surprise", prefixid="eng_dev_track_a_", idx_convert=0, all_ids=None, **kwargs):
    e_labels = e_labels_str.split(",")
    data_submit = {
        "id": []
    }
    for e in e_labels:
        data_submit[e] = []
        
    for idx, e in enumerate(pred_data_['detail_pred']):
        data_submit ['id'].append(prefixid + "{0:05d}".format(idx+1) if all_ids==None else all_ids[idx])
        all_preds = e[idx_convert].split(",")
        no_predicted_e =  set(e_labels)
        for e_pred in all_preds:
            for e_check in e_labels:
                if e_check not in e_pred: # e_pred = high degree of Happy ... , e_check=Happy
                    continue
                else:
                    no_predicted_e.remove(e_check)
                    intensity = 3 if 'high' in e_pred else 2 if 'moderate' in e_pred else 1 
                    data_submit[e_check].append(intensity)
                    break # no need to check other label if found 
        
        for e in no_predicted_e:
            data_submit[e].append(0)
        
    df_submit = pd.DataFrame(data_submit)
    return df_submit
    
    
def semeval_process(paths_folder_preprocessed_data, args, split_train_valid=True):
    
    for path_folder_preprocessed_data in paths_folder_preprocessed_data:
        
        d_type = 'train' if '.train.' in path_folder_preprocessed_data else \
                'publicvalid' if '.publicvalid.' in path_folder_preprocessed_data else \
                'valid' if '.valid.' in path_folder_preprocessed_data else \
                'test' if '.test.' in path_folder_preprocessed_data else None  
        # if d_type == "valid":
        #     # because internal valid data is processed with training data (10% of training data)
        #     continue
        
        folder_data = args.data_folder
        data_name = args.data_name
        path_data_out = path_folder_preprocessed_data
        prompting_type = args.prompting_type
        
        raw_data = f'{folder_data}/{d_type}/{data_name}.csv'
        org_data = pd.read_csv(raw_data) # ; org_data = dict([(k,v) for k,v in org_data.items()][:10])
        
        if prompting_type == "pairwise":
            new_format = semeval_pairwise_prompting(org_data, args)
        else:
            new_format = semeval_default_prompting(org_data, args)

        path_processed_data = raw_data.replace(".json", f".0shot_{prompting_type}.jsonl") if path_data_out is None else path_data_out
        
        if split_train_valid and d_type == 'train':
            random.shuffle(new_format)
            new_format_dev = new_format[-int(0.1*len(new_format)):]
            with open(f"{path_processed_data.replace('train', 'valid')}", 'wt') as f:
                f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in new_format_dev]))
                
            new_format_train = new_format[:-int(0.1*len(new_format))] #new_format[:] # 
            with open(f'{path_processed_data}', 'wt') as f:
                f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in new_format_train]))
            

        else:
            with open(f'{path_processed_data}', 'wt') as f:
                f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in new_format]))


def semeval_split_data(args, training_rate=0.9):
    folder_data = args.data_folder
    if not os.path.exists(f"{folder_data}/publictrain"):
        print(f"- Do not found  {folder_data}/publictrain!! No process is ran.")
        return
    
    if not os.path.exists(f"{folder_data}/valid"):
        os.makedirs(f"{folder_data}/valid")
    if not os.path.exists(f"{folder_data}/train"):
        os.makedirs(f"{folder_data}/train")
    if not os.path.exists(f"{folder_data}/test"):
        os.makedirs(f"{folder_data}/test")
        
    for track_id in ['a', 'b']:
        all_valid = []
        all_train = []
        if not os.path.exists(f"{folder_data}/valid/track_{track_id}"):
            os.makedirs(f"{folder_data}/valid/track_{track_id}")
        if not os.path.exists(f"{folder_data}/train/track_{track_id}"):
            os.makedirs(f"{folder_data}/train/track_{track_id}")
        for path_data in glob.glob(f'{folder_data}/publictrain/track_{track_id}/*.csv'):
            print( f"- process {path_data}")
            origin_df = pd.read_csv(path_data)
            lang_id = path_data.split("/")[-1].replace(".csv", "")
            
            origin_df = origin_df.sample(frac=1, random_state=1)
            train_part = origin_df.iloc[:int(training_rate*len(origin_df))]
            dev_part = origin_df.iloc[int(training_rate*len(origin_df)):]
            
            all_train.append(train_part)
            all_valid.append(dev_part)
            
            train_part.to_csv(f'{folder_data}/train/track_{track_id}/{lang_id}.csv', sep=',', index=False)
            dev_part.to_csv(f'{folder_data}/valid/track_{track_id}/{lang_id}.csv', sep=',', index=False)
    
        all_train_df = pd.concat(all_train, axis=0, ignore_index=True)
        all_train_df.to_csv(f'{folder_data}/train/track_{track_id}/alllanguage.csv', sep=',', index=False)
        all_valid_df = pd.concat(all_valid, axis=0, ignore_index=True)
        all_valid_df.to_csv(f'{folder_data}/valid/track_{track_id}/alllanguage.csv', sep=',', index=False)
             
    for track_id in ['a', 'b']:
        for d_type in ["publicvalid", "test"]:
            all_valid = []
            for path_data in glob.glob(f'{folder_data}/{d_type}/track_{track_id}/*.csv'):
                if "alllanguage.csv" in path_data:
                    continue
                print( f"- process {path_data}")
                all_valid.append(pd.read_csv(path_data))
            all_valid_df = pd.concat(all_valid, axis=0, ignore_index=True)
            all_valid_df.to_csv(f'{folder_data}/{d_type}/track_{track_id}/alllanguage.csv', sep=',', index=False) 
             