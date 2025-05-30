import argparse
import os
import pickle, torch, numpy as np
from torch import nn
from transformers import DebertaV2Config, DebertaV2Tokenizer, DebertaV2Model, BertForSequenceClassification

from simpletransformers.custom_models.models import RobertaForMultiLabelSequenceClassification,RobertaClassificationHead,BertForMultiLabelSequenceClassification
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs, ClassificationModel
)
import re 
import pandas as pd
import logging
import random, sys
import sklearn
import sklearn.metrics 
from sklearn.preprocessing import minmax_scale
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, DebertaV2Embeddings, DebertaV2Encoder, DebertaV2ForSequenceClassification, DebertaV2Model, DebertaV2PreTrainedModel

sys.path.append('libs/SemEval2025-Task11-Evaluation/')
from reformat_data_ft_llm import json_to_dataframe
from utils import evaluate as semeval11_evaluate

manual_seed = 1
random.seed(manual_seed)
label_str = "Anger,Disgust,Fear,Joy,Sadness,Surprise"
lang="amh"

def load_data(raw_data, training_rate = 0.9, shuffle=True):
    org_data = pd.read_csv(raw_data) # ; org_data = dict([(k,v) for k,v in org_data.items()][:10])
    
    all_data = []
    all_e_labels = [str(e) for e in org_data.columns[2:]]
    global label_str
    label_str = ",".join(all_e_labels)
    for index, row in org_data.iterrows():
        gold_lb = [row[e_check] if row[e_check]> 0 else 0 for e_check in all_e_labels]
        text_input = row['text'] # f"Which emotions in set [{','.join(all_e_labels)}] expressed in this sentence: {row['text']}?"
        id_sample = row['id'] # f"Which emotions in set [{','.join(all_e_labels)}] expressed in this sentence: {row['text']}?"
        all_data.append([
            id_sample, text_input, gold_lb
        ]) 
    if shuffle:
        random.shuffle(all_data)
    
    train_df = pd.DataFrame(all_data[:int(training_rate*len(all_data))])
    train_df.columns = ["id", "text", "labels"]
    valid_df = None
    if training_rate < 1.0:
        valid_df = pd.DataFrame(all_data[int(training_rate*len(all_data)):])
        valid_df.columns = ["id", "text", "labels"]
    return train_df, valid_df, all_e_labels

# BertForSequenceClassification
# BertForMultiLabelSequenceClassification
# DebertaV2ForSequenceClassification
     
     
class DebertaForMultiLabelSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config.problem_type = "multi_label_classification"
        self.config.max_length = 256
        
    def forward(self, *args, **kwargs):
        if len(args) >= 6:
            args[5] = args[5].float()
        if 'labels' in kwargs:
            kwargs['labels'] = kwargs['labels'].float()
        return super().forward(*args, **kwargs)
    
class MyMultiLabelClassificationModel(MultiLabelClassificationModel):
    def __init__(
        self,
        model_type,
        model_name,
        num_labels=None,
        pos_weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):
        """
        Initializes a MultiLabelClassification model.

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "deberta": (
                DebertaV2Config,
                DebertaForMultiLabelSequenceClassification,
                DebertaV2Tokenizer,
            )
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MultiLabelClassificationArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:
            self.args.fp16 = False

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels
        self.pos_weight = pos_weight
        self.loss_fct = None

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if not self.args.quantized_model:
            if self.pos_weight:
                self.model = model_class.from_pretrained(
                    model_name,
                    config=self.config,
                    pos_weight=torch.Tensor(self.pos_weight).to(self.device),
                    **kwargs,
                )
            else:
                self.model = model_class.from_pretrained(
                    model_name, config=self.config, **kwargs
                )
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )
            if self.pos_weight:
                self.model = model_class.from_pretrained(
                    None,
                    config=self.config,
                    state_dict=quantized_weights,
                    weight=torch.Tensor(self.pos_weight).to(self.device),
                )
            else:
                self.model = model_class.from_pretrained(
                    None, config=self.config, state_dict=quantized_weights
                )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        self.results = {}

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        self.weight = None  # Not implemented for multilabel

 
    
def manual_fscore_b(gold_labels, model_logit_outputs): 
    scores =  manual_fscore(gold_labels, model_logit_outputs, task_id='b') 
    return sum(scores.values()) / len(scores) 
    
def manual_fscore_a(gold_labels, model_logit_outputs):  
    return manual_fscore(gold_labels, model_logit_outputs, task_id='a')['macro']['f1']
    
def manual_fscore(gold_labels, model_logit_outputs, task_id='a'): 
    # model_logit_outputs = minmax_scale(model_logit_outputs, feature_range=(0.2, 1), axis=1, copy=True) 
    model_outputs = (1*(model_logit_outputs>0.5)).tolist()
    gold_labels = (1*(gold_labels ==1)).tolist()
    global label_str
    model_outputs = ["id,"+label_str]+[str(idx)+"," + ",".join([str(k) for k in e]) for idx, e in enumerate(model_outputs)]
    gold_lb = ["id,"+label_str]+[str(idx)+","+",".join([str(k) for k in e]) for idx, e in enumerate(gold_labels)]
    scores = semeval11_evaluate(gold_lb, model_outputs, task=task_id)
    print(scores)
    return scores 
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--lang', type=str,  help='language', default='eng')
    parser.add_argument('--do_train', action="store_true", help='fine tuning a PLM model', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--do_eval_publicvalid', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--model_type', type=str, default="xlmroberta", help='model_type') 
    parser.add_argument('--model_id', type=str, default="FacebookAI/xlm-roberta-large", help='model_id') 
    parser.add_argument('--task_id', type=str, default="a", help='task id') 

    args, unknown__ = parser.parse_known_args()
    lang = args.lang
    
    # test()
    # model_info = ("xlmroberta", "FacebookAI/xlm-roberta-large")
    # model_info = ("bert", "google-bert/bert-base-multilingual-cased")
    model_info = (args.model_type, args.model_id)
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    train_df, _, all_e_labels = load_data(raw_data = f'./data/semeval11/train/track_{args.task_id}/{lang}.csv', training_rate=1.0)
    eval_df, _, _ = load_data(raw_data = f'./data/semeval11/valid/track_{args.task_id}/{lang}.csv', training_rate=1.0)
    output_dir = f'./finetuned_llm/semeval_track_{args.task_id}/{model_info[1].replace("/", "_")}__{lang}'
    if not os.path.exists(output_dir) and args.do_train:
        os.makedirs(output_dir)
        

    # Optional model configuration
    model_args = MultiLabelClassificationArgs(use_multiprocessing=False,
                                              use_multiprocessing_for_evaluation=False)
    model_args.sliding_window = False
    model_args.labels_list = all_e_labels
    model_args.gradient_accumulation_steps = 2
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.save_best_model = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps=-1
    model_args.save_model_every_epoch = False
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 2
    model_args.num_train_epochs = 6
    model_args.max_seq_length = 256
    model_args.manual_seed = manual_seed
    model_args.tensorboard_dir = f"{output_dir}/tensorboard"
    model_args.best_model_dir = f"{output_dir}/best_model"
    model_args.early_stopping_metric = "manual_fscore"
    model_args.early_stopping_metric_minimize = False
    model_args.output_dir=output_dir 
    model_args.evaluate_each_epoch = True
    model_args.evaluate_during_training_steps = 2000

    # Create a MultiLabelClassificationModel
    # MultiLabelClassificationModel.MODEL_CLASSES["deberta"] =  (
    #     DebertaConfig,
    #     DebertaForMultiLabelSequenceClassification,
    #     DebertaTokenizer,
    # )
    # model = MyMultiLabelClassificationModel(
    #     "deberta",
    #     "microsoft/mdeberta-v3-base",
    #     num_labels=len(all_e_labels),
    #     args=model_args,
    #     ignore_mismatched_sizes=True
    # ) 
    
    if args.ft_model_path is None:
        model = MultiLabelClassificationModel(
            model_info[0],
            model_info[1],
            num_labels=len(all_e_labels),
            args=model_args,
            ignore_mismatched_sizes=True
        ) 
    else:
        model_args = MultiLabelClassificationArgs()
        model_args.load(args.ft_model_path)
        model = MultiLabelClassificationModel(
            model_args.model_type, args.ft_model_path, args=model_args, use_cuda=True
        )  
        
    eval_func = manual_fscore_b if args.task_id == "b" else manual_fscore_a
    
    # Train the model
    if args.do_train:
        model.train_model(train_df,  eval_df=eval_df,  manual_fscore=eval_func)
        
        # load best model at end 
        model = MultiLabelClassificationModel(
            model_args.model_type, model_args.best_model_dir, args=model_args, use_cuda=True
        )  
 
        
    # Evaluate the model
    def predict_data(d_type):
        publicvalid_df, _, all_data_labels = load_data(
            raw_data = f'./data/semeval11/{d_type}/track_{args.task_id}/{lang}.csv',
            training_rate=1.0,
            shuffle=False
        )
        all_model_labels = model_args.labels_list
        results = model.predict(
            publicvalid_df['text'].tolist()
        )
        predictions = results[0]
        predictions = [[publicvalid_df['id'][idx]]+preds for idx, preds in enumerate(predictions) ]
        out_df = pd.DataFrame(predictions)
        
        out_df.columns = ['id']+all_model_labels
        out_folder = args.ft_model_path if args.ft_model_path is not None else model_args.best_model_dir
        out_folder = f"{out_folder}/{d_type}/track_{args.task_id}"
        print(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        
        # check labels 
        remove_columns = [e for e in all_model_labels if e not in all_data_labels]
        if len(remove_columns) > 0:
            out_df = out_df.drop(columns=remove_columns)
        
        out_df.to_csv(f"{out_folder}/pred_{args.lang}.csv", sep=',', index=False)
        
    # Evaluate the model
    if args.do_eval_dev:
        predict_data('valid')
    
    # Evaluate the model
    if args.do_eval_publicvalid:
        predict_data('publicvalid')
        
    # Evaluate the model
    if args.do_eval_test:
        predict_data('test')
    
    # pickle.dump([result, model_outputs, wrong_predictions, all_e_labels], open(f'{output_dir}/model_outputs.pkl', 'wb'))
    # print(result)
     