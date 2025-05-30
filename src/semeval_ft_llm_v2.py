import argparse
import json
import re
import shutil
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
from reformat_data_ft_llm import json_to_dataframe, json_to_dataframe_taskb, pairwise_prompting_json_to_dataframe
import sys

sys.path.append('libs/SemEval2025-Task11-Evaluation/')
from utils import evaluate as semeval11_evaluate

# FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}

def my_setup_chat_format(
    model,
    tokenizer,
    format = "chatml",
    resize_to_multiple_of= None,
): 
    # check if model already had a chat template
    if tokenizer.chat_template is not None:
        raise ValueError(
            "Chat template is already added to the tokenizer. If you want to overwrite it, please set it to None"
        )

    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None, mean_resizing=False
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
 
 
def formatting_prompts_func(samples):
    prompt_texts = [tokenizer.apply_chat_template(
             sample[:-1], tokenize=False, add_generation_prompt=True) for sample in samples["messages"]]
    
    print("=="*50)
    print(prompt_texts[-1])
    print("=="*50)
    return prompt_texts

def split_label(sample, tokenizer_):
    tokenized_lb = tokenizer_.encode(sample['messages'][-1]['content'], padding='max_length',max_length=30 )
    sample['labels'] = tokenized_lb 
    return sample
 
class LLMErcTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        eval_dataset = kwargs.get('eval_dataset')
        kwargs = kwargs.get('args')
        
        self.data_process_args = argparse.Namespace(
            packing=False,
            dataset_text_field=None,
            max_seq_length=kwargs.max_seq_length,
            formatting_func=formatting_prompts_func,
            num_of_sequences=kwargs.num_of_sequences,
            chars_per_token=kwargs.chars_per_token,
            remove_unused_columns=kwargs.remove_unused_columns,
            dataset_kwargs=kwargs.dataset_kwargs
        )
        self.eval_dataset = self._process_raw_data(eval_dataset)  
        print("len(eval dataset) = ",  len(self.eval_dataset))
    
    def _process_raw_data(self, dataset):
        dataset2 = dataset.map(split_label, fn_kwargs={"tokenizer_": self.processing_class})
        dataset = self._prepare_dataset(
                dataset=dataset,
                processing_class=self.processing_class,
                packing=False,
                dataset_text_field=None,
                max_seq_length=self.data_process_args.max_seq_length,
                formatting_func=self.data_process_args.formatting_func,
                num_of_sequences=self.data_process_args.num_of_sequences,
                chars_per_token=self.data_process_args.chars_per_token,
                remove_unused_columns=self.data_process_args.remove_unused_columns,
                **self.data_process_args.dataset_kwargs, 
            )
        dataset = dataset.add_column('labels', dataset2['labels']) 
        return dataset 
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if "input_ids" not in eval_dataset.column_names and "labels" not in eval_dataset.column_names:
            # this is raw data which need to preprocess
            eval_dataset = self._process_raw_data(eval_dataset)
            
        return super().get_eval_dataloader(eval_dataset)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only= None,
        ignore_keys = None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        model = self.model
        # model_dtype = next(model.parameters()).dtype
        # model = model.to(dtype=torch.bfloat16)
        
        model.eval()
            
        # losses/preds/labels on CPU (final containers)
        all_preds = []
        all_labels = []
        all_raw_decoded = []
         
        def post_process(str_out):
            try:
                gen_text = str_out.split("assistant\n")[-1].split(self.processing_class.eos_token)[0]
            except:
                gen_text = "error"
            return gen_text
        
        # all id
        all_ids = None
        all_e_checks = None
        example_msg_question=""
        lang_id = "unknown"
        _dataset = dataloader.dataset
        if 'id' in dataloader.dataset.column_names:
            all_ids = dataloader.dataset['id']
            lang_id = all_ids[0].split("_")[0]
            _dataset = _dataset.remove_columns(["id"])
        if 'e_check' in dataloader.dataset.column_names:
            all_e_checks = dataloader.dataset['e_check']
            _dataset = _dataset.remove_columns(["e_check"])
        if 'messages' in dataloader.dataset.column_names:
            example_msg = dataloader.dataset['messages'][0][0]['content']
            example_msg_question = dataloader.dataset['messages'][0][1]['content']
            _dataset = _dataset.remove_columns(["messages"])
        dataloader = super().get_eval_dataloader(_dataset)
        
        # Main evaluation loop
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader)):
                inputs = self._prepare_inputs(inputs)
                gen_kwargs = {'max_new_tokens': 30, 
                              'do_sample': False, 
                              'eos_token_id': self.processing_class.eos_token_id, 
                              'pad_token_id': self.processing_class.pad_token_id,
                              "temperature": 0.1,
                              }
                generated_tokens = model.to(torch.float).generate(inputs["input_ids"],attention_mask=inputs["attention_mask"],**gen_kwargs,)
                labels = inputs.pop("labels")
                str_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)
                
                raw_decoded = [e for e in self.processing_class.batch_decode(generated_tokens, skip_special_tokens=False)]
                str_decoded = [post_process(e) for e in raw_decoded]
                all_preds += str_decoded
                all_labels += str_labels 
                all_raw_decoded += raw_decoded
        num_samples = len(dataloader)
           
        task_id = 'a' if ('track_a' in all_ids[0] or 'track_c' in all_ids[0]) else "b"
        
        if "label set includes {" in example_msg and "}. Each sentence may " in example_msg:
            label_str = example_msg.split("emotional label set includes {")[1].split("}. Each sentence may have")[0].replace(" ", "")
        elif "emotional label set is {" in example_msg and "}. Each sentence may " in example_msg:
            label_str = example_msg.split("emotional label set is {")[1].split("}. Each sentence may have")[0].replace(" ", "")
        elif "emotional label set includes {" in example_msg and "}, with three levels of intensity" in example_msg:
            label_str = example_msg.split("emotional label set includes {")[1].split("}, with three levels of intensity")[0].replace(" ", "")
        
        # checking template default or pairwise
        if ", is the emotion" in example_msg_question or ", what is the intensity of the emotion" in example_msg_question: 
            # pairwise
            process_fn_json_to_dataframe = pairwise_prompting_json_to_dataframe
        else:
            # default
            if task_id == 'b':
                process_fn_json_to_dataframe = json_to_dataframe_taskb
            else:
                process_fn_json_to_dataframe = json_to_dataframe
        if "sadness" not in label_str or "anger" not in label_str:
            print(f"[W] label string have problem need check: {label_str}")
        data_eval = {"detail_pred": list(zip(all_preds, all_labels))}
        
        
        df_pred = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=0, all_ids=all_ids, all_e_checks=all_e_checks)
        pred_lines = df_pred.to_csv(sep=',', index=False).strip('\n').split('\n')
        df_gold = process_fn_json_to_dataframe(data_eval, e_labels_str=label_str, idx_convert=1, all_ids=all_ids, all_e_checks=all_e_checks)
        gold_lines = df_gold.to_csv(sep=',', index=False).strip('\n').split('\n')
        
        scores = semeval11_evaluate(gold_lines, pred_lines, task=task_id)
        print(scores)
        eval_f1_score = scores['macro']['f1'] if task_id=="a" else sum(scores.values()) / len(scores) 
        metrics = { f"{metric_key_prefix}_weighted-f1": eval_f1_score  }
        
        if task_id=="b":
            scores = dict([(k,float(v)) for k, v in scores.items() ])
        
        # dump submission file with ids
        # if all_ids is not None: 
        #     df_pred['id'] = all_ids
        df_pred.to_csv(f"{self.args.output_dir}/result_{lang_id}_{metric_key_prefix}_step-{self.state.global_step}.csv", sep=',', index=False)
        
        json.dump({"metrics": { f"{metric_key_prefix}_weighted-f1": eval_f1_score, 'other_scores': scores }, 
                   "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded)),
                   "all_ids": all_ids
                   }, 
                   open(f"{self.args.output_dir}/result_{lang_id}_{metric_key_prefix}_step-{self.state.global_step}.json", "wt"), indent=1, ensure_ascii=False)
        
        # free the memory again
        del model
        torch.cuda.empty_cache()
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        
    
if __name__=='__main__':
        
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--do_train', action="store_true", help='fine tuning a LLM model with LoRA', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--do_eval_publicvalid', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--do_eval_train', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument('--prompting_type', type=str, default='spdescV2', help='prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--base_model_id', type=str, default='meta-llama/Llama-2-7b-hf', help='base llm model id')
    parser.add_argument('--extract_prompting_llm_id', type=str, default='Llama-2-7b-chat-hf', help='base llm model id')
    parser.add_argument('--epoch', type=int, default=None, help='training epoch')
    parser.add_argument('--max_steps', type=int, default=None, help='training steps')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate value')
    parser.add_argument('--seed', type=int, default=1, help='random seed value')
    parser.add_argument('--kshot', type=int, default=0, help='k shot examples for llm')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--eval_delay', type=int, default=200, help='eval delay')
    parser.add_argument('--log_step', type=int, default=50, help='eval delay')
    parser.add_argument('--window', type=int, default=5, help='local context window size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length for chunking/packing')
    parser.add_argument('--re_gen_data', action="store_true", help='re generate data', default=False)
    parser.add_argument('--data_name', type=str,  help='data name in {iemocap, meld, emorynlp}', default='iemocap')
    parser.add_argument('--data_folder', type=str,  help='path folder save all data', default='/home/s2220429/per_erc/data/all_raw_data')
    parser.add_argument('--output_folder', type=str,  help='path folder save all data', default='./finetuned_llm/')
    parser.add_argument('--path_emotional_first_prediction',  type=str, default=None, help='path of emotional first prediction') 
    parser.add_argument('--predefined_label_in_prompt',  action="store_true", help='predefined label in prompting', default=False) 
    parser.add_argument('--split_all_language', action="store_true", help='split data', default=False)
    parser.add_argument('--train_pipeline_all_language', action="store_true", help='split data', default=False)
    parser.add_argument('--filter_path',  type=str,  help='path folder save all data', default=None)
    parser.add_argument('--synthesized_conv_path',  type=str,  help='path save synthesized conversational data', default='/home/s2220429/per_erc/data/semeval11/synthesized_conv/seval_context_outputs_{}_Qwen2.5-14B-Instruct_first.jsonl')
    
    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.prompting_type == 'zeroshot':
        args.kshot = 0
    print(args)
    
    set_random_seed(args.seed)
    if args.split_all_language:
        semeval_split_data(args)
        exit()
    
    # correct the prompting type name :
    prompting_type_jsonl_data = f"{args.prompting_type}-emotional-prediction-support" if args.path_emotional_first_prediction is not None else args.prompting_type
    prompting_type_jsonl_data = f"{prompting_type_jsonl_data}-withpredefinedlabel" if args.predefined_label_in_prompt else prompting_type_jsonl_data
    
    ft_model_id = f"semeval_{args.data_name}_{args.base_model_id.split('/')[-1]}_ep{args.epoch}_step{args.max_steps}_lrs-{args.lr_scheduler}{args.lr}"+\
                f"_r{args.lora_r}_{prompting_type_jsonl_data}"+\
                f"_ED{args.eval_delay}_idRun-{args.ft_model_id}"
                
    
    if args.train_pipeline_all_language:
        lang_id = args.data_name.split("/")[-1]
        args.lang_id = lang_id
        share_lang_id = "/".join(args.data_name.split("/")[:-1]+["pipeline-crosslingual"])
        ft_model_id = ft_model_id.replace(f"_{args.data_name}_", f"_{share_lang_id}_")
        if os.path.exists(f'{args.output_folder}/{ft_model_id}/run_config.json'):
            args.ft_model_path = f'{args.output_folder}/{ft_model_id}/'
    
    output_dir=f'{args.output_folder}/{ft_model_id}'
    print(output_dir)
    
    if args.do_train:
        print(f"- Mkdir {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if (os.path.exists(f'{output_dir}/run_config.json')) and not args.train_pipeline_all_language:
            # checking condition 
            exit()
    else:
        if args.ft_model_path is not None:
            output_dir=args.ft_model_path
            
    
    # generate training data for LLM  
    args.prompting_type_jsonl_data = prompting_type_jsonl_data
    all_type_data = ['train', 'valid'] 
    if args.do_eval_publicvalid: 
        all_type_data.append("publicvalid")
    if args.do_eval_test: 
        all_type_data.append("test")
    all_path_folder_preprocessed_data = dict([(d_type, f"{output_dir}/{args.data_name.replace('/', '_')}.{d_type}.{args.kshot}shot_w{args.window}_{prompting_type_jsonl_data}.jsonl") \
        for d_type in all_type_data])
    if args.re_gen_data:
        semeval_process([all_path_folder_preprocessed_data[d_type] for d_type in all_type_data], args, split_train_valid=False) 
                    
    
    # Load jsonl data from disk
    dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data['train'], split="train", cache_dir=output_dir) if 'train' in all_path_folder_preprocessed_data else None
    valid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data['valid'], split="train", cache_dir=output_dir) if 'valid' in all_path_folder_preprocessed_data else None
    publicvalid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data['publicvalid'], split="train", cache_dir=output_dir) if 'publicvalid' in all_path_folder_preprocessed_data else None
    test_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data['test'], split="train", cache_dir=output_dir) if 'test' in all_path_folder_preprocessed_data else None
    
    # Load model and tokenizer
    TokenizerClass = AutoTokenizer if "t5" not in args.base_model_id else MT5Tokenizer
    LLMClass = AutoModelForCausalLM if "t5" not in args.base_model_id else MT5ForConditionalGeneration
    
    model_id = args.base_model_id # "codellama/CodeLlama-7b-hf" # or `mistralai/Mistral-7B-v0.1`
    tokenizer = TokenizerClass.from_pretrained(model_id if args.ft_model_path  is None else args.ft_model_path, use_fast="t5" not in args.base_model_id)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    if args.do_train:
        model = LLMClass.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            # torch_dtype=tensor_data_type,
            quantization_config=quantization_config
        )
    else:
        if args.do_eval_publicvalid or  args.do_eval_dev or args.do_eval_test or args.do_eval_train:
            tensor_data_type = torch.float32 # for reduce the miss matching of ouputs of batch inference
            if args.ft_model_path is not None:
                output_dir = args.ft_model_path
                model = AutoPeftModelForCausalLM.from_pretrained(
                    args.ft_model_path,
                    device_map="auto",
                    # torch_dtype=tensor_data_type,
                    quantization_config=quantization_config,
                )
            else: 
                tensor_data_type = torch.float32  
                model = LLMClass.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=tensor_data_type,
                    load_in_8bit=True
                )
        
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left' 

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    if "Qwen" not in model_id:
        if tokenizer.chat_template is not None:
            tokenizer.chat_template = None
        model, tokenizer = my_setup_chat_format(model, tokenizer) 
    
    # training config 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=args.lora_r,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
    )
 
    training_args = SFTConfig(
        max_seq_length=args.max_seq_len,
        packing=True,
        output_dir=output_dir,                  # directory to save and repository id
        num_train_epochs= args.epoch,                     # number of training epochs
        max_steps=args.max_steps,
        per_device_train_batch_size=4,          # batch size per device during training
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        save_total_limit=1,
        optim="adamw_torch_fused",              # use fused adamw optimizer
        eval_delay=args.eval_delay,                       # log every 10 steps meld:200
        logging_steps=args.log_step,                       # log every 10 steps
        eval_steps=0.1,
        save_steps=0.1,
        load_best_model_at_end=True if not args.train_pipeline_all_language else False,
        metric_for_best_model='weighted-f1',
        # metric_for_best_model='loss',
        # greater_is_better=False,
        greater_is_better=True,
        evaluation_strategy='steps',
        save_strategy="steps",                  # save checkpoint every epoch
        learning_rate=args.lr,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type=args.lr_scheduler,           # use constant learning rate scheduler
        push_to_hub=False,                      # push model to hub ##########################
        group_by_length=False,
        remove_unused_columns=False,
        report_to="tensorboard",                # report metrics to tensorboard
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        },
        neftune_noise_alpha=5,
    )
    trainer = LLMErcTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config if args.do_train else None,
        processing_class=tokenizer,
        
    )
    
    # n_trainable_pr, total_pr = get_peft_model(model, peft_config).get_nb_trainable_parameters()
    # print(f"total params: {n_trainable_pr}, trainable params {total_pr}, percentage={n_trainable_pr/total_pr*100}")
    config_file = f"{output_dir}/run_config.json"
    all_params = vars(args) 
    all_params['training_args'] = dict([(k, str(v))for k, v in vars(training_args).items() ])
    if args.do_train:
        print("training .... ")
        json.dump(all_params, open(config_file.replace(".json", "_tmp.json"), "wt"), indent=2)
        
        # start training, the model will be automatically saved to the hub and the output directory
        print(f"- train with: resume_from_checkpoint={args.ft_model_path}")
        if args.train_pipeline_all_language and args.ft_model_path is not None:
            print(f"- copy: /home/phuongnm/per_erc/finetuned_llm/trainer_state.json -> {args.ft_model_path}/trainer_state.json")
            shutil.copyfile('/home/phuongnm/per_erc/finetuned_llm/trainer_state.json', f"{args.ft_model_path}/trainer_state.json")
        trainer.train(resume_from_checkpoint=args.ft_model_path)

        # save model 
        trainer.save_model()
        trainer.save_state()
        json.dump(all_params, open(config_file, "wt"), indent=2)
        os.remove(config_file.replace(".json", "_tmp.json"))
    else:
        if not os.path.exists(config_file):
            json.dump(all_params, open(config_file, "wt"), indent=2)

    ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        
    if args.do_eval_test:
        print("eval test .... ")
        result = trainer.evaluate(test_dataset, metric_key_prefix='test')
        print(f"Test result = {result}")
        
    if args.do_eval_publicvalid:
        print("eval publicvalid .... ")
        result = trainer.evaluate(publicvalid_dataset, metric_key_prefix='publicvalid')
        print(f"Publicvalid result = {result}")
        
    if args.do_eval_dev:
        print("eval dev .... ")
        result = trainer.evaluate(valid_dataset, metric_key_prefix='valid')
        print(f"Valid result = {result}")
        
    if args.do_eval_train:
        print("eval train .... ")
        result = trainer.evaluate(dataset, metric_key_prefix='train')
        print(f"train result = {result}")