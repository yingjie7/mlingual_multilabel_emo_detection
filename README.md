# JNLP at SemEval-2025 Task 11: Cross-Lingual Multi-Label Emotion Detection Using Generative Models
> **Note**: This project is built upon our previous work [(BiosERC)](https://github.com/yingjie7/BiosERC). Please refer to this project for more details. 
## Abstract 

With the rapid advancement of global digitalization, users from different countries increasingly rely on social media for information exchange. In this context, multilingual multi-label emotion detection has emerged as a critical research area.
This study addresses SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. Our paper focuses on two sub-tracks of this task: (1) Track A: Multi-label emotion detection, and (2) Track B: Emotion intensity.
To tackle multilingual challenges, we leverage pre-trained multilingual models and focus on two architectures: (1) a fine-tuned BERT-based classification model and (2) an instruction-tuned generative LLM. Additionally, we propose two methods for handling multi-label classification: the \textit{base} method, which maps an input directly to all its corresponding emotion labels, and the \textit{pairwise} method, which models the relationship between the input text and each emotion category individually.
Experimental results demonstrate the strong generalization ability of our approach in multilingual emotion recognition. In Track A, our method achieved Top 4 performance across 10 languages, ranking 1st in Hindi. In Track B, our approach also secured Top 5 performance in 7 languages, highlighting its simplicity and effectiveness.

## Data structure 
```
./data/semeval11/
├── train
│   └── track_a
│   │   ├── chn.csv
│   │   ├── deu.csv
│   │   ├── eng.csv 
│   │   ├── .... (other languages)
│   │   └── ptbr.csv
│   └── track_b
│       └── eng.csv 
│       ├── .... (other languages)
│       └── ptbr.csv
├── publicvalid -> ... (similar to train data)
├── test -> ... (similar to train data)
└── valid -> ... (similar to train data)
```
##  Python ENV 
Init python environment 
```cmd
    conda create --prefix=./env_semeval_py39  python=3.9
    conda activate ./env_semeval_py39 
    pip install -r requirements.txt
```

## Run 

```bash 
python ./src/semeval_ft_llm_v2.py \
--do_eval_dev --do_eval_test --do_train \
--max_seq_len 1024 \
--base_model_id meta-llama/Llama-2-7b-hf \
--ft_model_id debug \
--lr_scheduler linear --lr 3e-4 --epoch 3 --lora_r 32 \
--data_name track_a/eng --max_steps -1 \
--data_folder ./data/semeval11/ \
--prompting_type synthezied-conv-default --log_step 50 --eval_delay 50 \
--re_gen_data 
```

## Citation 
```bibtex
@inproceedings{xue-etal-2025-jnlp,
  title = "{JNLP at SemEval-2025 Task 11: Cross-Lingual Multi-Label Emotion Detection Using Generative Models}",
  author = "Xue, Jieying and Nguyen, Phuong Minh and Nguyen, Minh Le and Liu, Xin",
  booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics"
}
```

```bibtex
@InProceedings{10.1007/978-3-031-72344-5_19,
    author="Xue, Jieying
    and Nguyen, Minh-Phuong
    and Matheny, Blake
    and Nguyen, Le-Minh",
    editor="Wand, Michael
    and Malinovsk{\'a}, Krist{\'i}na
    and Schmidhuber, J{\"u}rgen
    and Tetko, Igor V.",
    title="BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks",
    booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2024",
    year="2024",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="277--292",
    abstract="In the Emotion Recognition in Conversation task, recent investigations have utilized attention mechanisms exploring relationships among utterances from intra- and inter-speakers for modeling emotional interaction between them. However, attributes such as speaker personality traits remain unexplored and present challenges in terms of their applicability to other tasks or compatibility with diverse model architectures. Therefore, this work introduces a novel framework named BiosERC, which investigates speaker characteristics in a conversation. By employing Large Language Models (LLMs), we extract the ``biographical information'' of the speaker within a conversation as supplementary knowledge injected into the model to classify emotional labels for each utterance. Our proposed method achieved state-of-the-art (SOTA) results on three famous benchmark datasets: IEMOCAP, MELD, and EmoryNLP, demonstrating the effectiveness and generalization of our model and showcasing its potential for adaptation to various conversation analysis tasks. Our source code is available at https://github.com/yingjie7/BiosERC.",
    isbn="978-3-031-72344-5"
}

```