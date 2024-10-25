import sys
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, load_metric
from transformers import (
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
)
from utils import *
import re
import json
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Log Few-shot Learning')

    # Logistics
    parser.add_argument('--template', '-t', type=str, default='dataset/template.txt', help='path to the raw template file')
    parser.add_argument('--gen_data', '-d', type=str, default='dataset/dataset.json', help='path to the generated dataset json file')
    parser.add_argument('--data_name', '-n', type=str, default='hdfs', choices=['AIT', 'BGL', 'HDFS'], help='Dataset name')
    parser.add_argument('--output_dir', '-o', type=str, default='dataset/lfe', help='path to the generated files directory')
    parser.add_argument('--ckpt_dir', type=str, default='results/BART_seq2seq/10-shot-0', help='checkpoint directory')
    parser.add_argument('--strategy', '-s', type=int, default=0, choices=[0, 1], help='strategy to generate prompt template')
    parser.add_argument('--n_grams', type=int, default=8, help='how many grams for generating entity phrases')
    parser.add_argument('--neg_rate', type=float, default=1.5, help='negative rate for sampling negative entity phrases')
    parser.add_argument('--n_shots', type=int, default=10, help='how many shots for each class to generate few-shot datasets')
    parser.add_argument('--labeling_technique', type=str, default='prompt', choices=['prompt', 'regex'], help='use prompt seq2seq or regular expression to recognize named entities')

    # Training args
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--max_source_length', type=int, default=1024)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default='facebook/bart-large')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--pad_to_max_length', action='store_true', default=False, help="whether to pad all samples to model maximum sentence length")
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help="whether to ignore the tokens corresponding to padded labels in the loss computation or not")
    parser.add_argument('--preprocessing_num_workers', type=int, default=None, help="the number of processes to use for the preprocessing")
    parser.add_argument('--overwrite_cache', action='store_true', default=False, help="overwrite the cached training and evaluation sets")
    return parser.parse_args()

def load_data(gen_data):
    with open(gen_data, 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_dict({
        "log:msg": [item["log:msg"] for item in data],
        "log:hasParameterList": [item["log:hasParameterList"] for item in data],
        "log:hastag": [item["log:hastag"] for item in data]
    })
    return data, dataset

def initialize_entity_sets(dataset, ENTITY_COLUMN_NAME, TAG_COLUMN_NAME):
    entity_set = defaultdict(set)
    entity_count = defaultdict(list)
    for i, instance in enumerate(dataset):
        entity_list = instance.get(ENTITY_COLUMN_NAME, [])
        tag_list = instance.get(TAG_COLUMN_NAME, [])
        for ent, tag in zip(entity_list, tag_list):
            entity_set[tag].add(ent)
            entity_count[tag].append(i)
    return entity_set, entity_count

def split_data(data, entity_count, seed, n_shots):
    n_shot_ids = []
    ten_shot_ids = []
    random.seed(seed)
    for tag in entity_count:
        tag_ids = random.choices(entity_count[tag], k=10)
        ten_shot_ids.extend(tag_ids)
        n_shot_ids.extend(tag_ids[:n_shots])
    ten_shot_ids = list(set(ten_shot_ids))
    max_ten_shot_length = len(data) // 2
    if len(ten_shot_ids) > max_ten_shot_length:
        ten_shot_ids = random.sample(ten_shot_ids, max_ten_shot_length)
    n_shot_data = [data[i] for i in n_shot_ids]
    random.shuffle(n_shot_data)
    remain_ids = list(set(range(len(data))) - set(ten_shot_ids))
    val_ids = random.sample(remain_ids, int(len(remain_ids) * 0.5))
    test_ids = list(set(remain_ids) - set(val_ids))
    return n_shot_data, val_ids, test_ids

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def preprocess_data(train_set, val_set, tokenizer, max_source_length, max_target_length, pad_to_max_length, ignore_pad_token_for_loss, preprocessing_num_workers, overwrite_cache):
    INPUT_COLUMN_NAME = 'log'
    TARGET_COLUMN_NAME = 'prompt'
    #LABEL_COLUMN_NAME = 'entity_tags'
    padding = "max_length" if pad_to_max_length else False
    column_names = train_set.column_names
    print("Train set columns:", train_set.column_names)
    print("Validation set columns:", val_set.column_names)


    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples[INPUT_COLUMN_NAME],
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[TARGET_COLUMN_NAME],
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_data = train_set.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )
    tokenized_val_data = val_set.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )
    return tokenized_train_data, tokenized_val_data

def template_entity(words, input_TXT, start, tokenizer, model, device=DEVICE, strategy=0):
    '''
    tokenizer: huggingface transformer pre-trained tokenizer.
    model: huggingface transformer pre-trained language model.
    words (list): list of all enumerated word phrases starting from 'start' index. 
    '''
    # input text -> template
    num_words = len(words)
    num_labels = len(LABEL2TEMPLATE) + 1
    input_TXT = [input_TXT]*(num_labels*num_words)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [v[strategy] for v in LABEL2TEMPLATE.values()] + [NONE2TEMPLATE[strategy]]
    entity_dict = {i:k for i,k in enumerate(LABEL2TEMPLATE.keys())}
    entity_dict[len(LABEL2TEMPLATE)] = 'O'
    
    temp_list = []
    for i in range(num_words):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids'] # num_words*num_labels X T
    output_ids[:, 0] = tokenizer.eos_token_id # num_words*num_labels X T
    output_length_list = [0]*num_labels*num_words 

    for i in range(len(temp_list)//num_labels): # word phrase + is (+ not)
        base_length = ((tokenizer(temp_list[i * num_labels], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*num_labels:i*num_labels+num_labels] = [base_length]*num_labels
        output_length_list[i*num_labels+num_labels-1] += 1 # negative ones

    score = [1 for _ in range(num_labels*num_words)] # placeholder for template scores
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0] # 2 means "entity ."
        for i in range(output_ids.shape[1] - 3): # 2 + 1
            logits = output[:, i, :] # num_words*num_labels X V
            logits = logits.softmax(dim=1) # num_words*num_labels X V
            logits = logits.to('cpu').numpy()
            for j in range(num_labels*num_words):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    largest_idx = score.index(max(score))
    # score_temp = [(s, t) for s, t in zip(score, temp_list)]
    # for i, (s, t) in enumerate(score_temp):
    #     if i == largest_idx:
    #         print('(best) score: {}, temp: {}, entity: {}, label: {}'.format(s, t, words[i%num_words], entity_dict[i%num_labels]))
    #     else:
    #         print('score: {}, temp: {}, entity: {}, label: {}'.format(s, t, words[i%num_words], entity_dict[i%num_labels]))
    end = start+(largest_idx//num_labels)
    return [start, end, entity_dict[(largest_idx%num_labels)], max(score)] # [start_index, end_index, label, score]

def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def prediction(input_TXT, model, tokenizer, strategy=0, n=8, device=DEVICE):
    input_TXT_list = list(filter(None, re.split(TOKENIZE_PATTERN, input_TXT)))
    num_tok = len(input_TXT_list) # number of tokens

    entity_list = []
    for i in range(num_tok): # i: start index
        words = []
        # Enumerate all word phrases starting from i
        for j in range(1, min(n+1, num_tok - i + 1)): # j: offset index (w.r.t. i)
            word = (' ').join(input_TXT_list[i:i+j]) # words[i:i+j]
            words.append(word) 

        entity = template_entity(words, input_TXT, i, tokenizer, model, device, strategy) # [start_index, end_index, label, score]
        if entity[1] >= num_tok:
            entity[1] = num_tok-1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * num_tok

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
        
    return label_list



def compute_metrics(eval_preds, tokenizer, ignore_pad_token_for_loss, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train_model(trainer, checkpoint, tokenized_train_data, max_train_samples):
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()
    max_train_samples = max_train_samples if max_train_samples is not None else len(tokenized_train_data)
    metrics["train_samples"] = min(max_train_samples, len(tokenized_train_data))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def evaluate_model(trainer, test_set, model, tokenizer, strategy, LABEL_COLUMN_NAME, entity_count, output_dir, n_shots, labeling_technique):
    model.eval()
    model.config.use_cache = False
    preds_list, trues_list = [], []
    for instance in tqdm(test_set):
        log = instance['log']
        pred = prediction(log, model, tokenizer, strategy)
        preds_list.append(pred)

        print("###################")
        print("Instance:", instance)

        trues_list.append(instance[LABEL_COLUMN_NAME])
        print('Pred:', pred)
        print('Gold:', instance[LABEL_COLUMN_NAME])
    true_entities = [get_entities_bio(true_list) for true_list in trues_list]
    pred_entities = [get_entities_bio(pred_list) for pred_list in preds_list]
    results = {
        "precision": compute_precision(true_entities, pred_entities),
        "recall": compute_recall(true_entities, pred_entities),
        "f1": compute_f1_score(true_entities, pred_entities),
    }
    eval_dict = evaluation_report(true_entities, pred_entities, entity_count)
    print(eval_dict)
    savePath = os.path.join(output_dir, f"preds-prompt-{n_shots}-shot-{strategy}.json")
    save_preds(test_set, pred_entities, true_entities, savePath, results, eval_dict, labeling_technique)

def regex_entity_recognition(test_set, output_dir, entity_count, labeling_technique):
    preds_list, trues_list = [], []
    for i, instance in enumerate(test_set):
        preds, truths = set(), set()
        text = test_set[i]['log']
        for tag, pat in REGEX_PATTERN.items():
            ans = re.findall(pat, text + ' ')
            if ans:
                for phrase in ans:
                    if isinstance(phrase, str):
                        preds.add((tag, phrase))
                    elif isinstance(phrase, tuple):
                        phrase = min(list(phrase), key=len)
                        preds.add((tag, phrase))
                    else:
                        raise TypeError()
        for ent, tag in zip(instance['entities'], instance['tags']):
            truths.add((tag, ent))
        trues_list.append(truths)
        preds_list.append(preds)

    true_entities = trues_list
    pred_entities = preds_list
    results = {
        "precision": precision_score(true_entities, pred_entities),
        "recall": recall_score(true_entities, pred_entities),
        "f1": f1_score(true_entities, pred_entities),
    }

    eval_dict = evalutation_report(true_entities, pred_entities, entity_count)
    print(eval_dict)

    savePath = os.path.join(output_dir, f"preds-regex.json")
    save_preds(test_set, pred_entities, true_entities, savePath, results, eval_dict, labeling_technique)


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data, dataset = load_data(args.gen_data)

    ENTITY_COLUMN_NAME = 'log:hasParameterList'
    TAG_COLUMN_NAME = 'log:hastag'
    LOG_COLUMN_NAME = 'log:msg'

    entity_set, entity_count = initialize_entity_sets(dataset, ENTITY_COLUMN_NAME, TAG_COLUMN_NAME)

    n_shot_data, val_ids, test_ids = split_data(data, entity_count, args.seed, args.n_shots)

    val_data = dataset.select(val_ids)
    test_data = dataset.select(test_ids)
    #print("Validation IDs:", val_ids)

    #print("Dataset length:", len(data))

    #print("val_data!!!!!!!!!!!!!!!",val_data)
    trainPath = os.path.join(args.output_dir, f"train-{args.n_shots}-shot-{args.strategy}.json")
    valPath = os.path.join(args.output_dir, f"val-{args.strategy}.json")
    testPath = os.path.join(args.output_dir, "test.json")
    gen_train_prompt(n_shot_data, trainPath, args.strategy, n=args.n_grams, negrate=args.neg_rate, seed=args.seed)
    gen_train_prompt(val_data, valPath, args.strategy, n=args.n_grams, negrate=args.neg_rate, seed=args.seed)

    df = pd.read_json(trainPath, lines=True)
    train_set = Dataset.from_pandas(df)

    df = pd.read_json(valPath, lines=True)
    val_set = Dataset.from_pandas(df)

    gen_test_labels(test_data, testPath)
    df = pd.read_json(testPath, lines=True)
    test_set = Dataset.from_pandas(df)

    if args.labeling_technique == 'prompt':
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)

        tokenized_train_data, tokenized_val_data = preprocess_data(
            train_set, val_set, tokenizer, args.max_source_length, args.max_target_length,
            args.pad_to_max_length, args.ignore_pad_token_for_loss, args.preprocessing_num_workers, args.overwrite_cache
        )

        metric = load_metric("sacrebleu")

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.ckpt_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
        )

        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data if training_args.do_train else None,
            eval_dataset=tokenized_val_data if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, args.ignore_pad_token_for_loss, metric),
        )

        if args.do_train:
            train_model(trainer, args.checkpoint, tokenized_train_data, args.max_train_samples)

        if args.do_eval:
            evaluate_model(trainer, test_set, model, tokenizer, args.strategy, LABEL_COLUMN_NAME, entity_count, args.output_dir, args.n_shots, args.labeling_technique)
        else:
            regex_entity_recognition(test_set, args.output_dir, entity_count, args.labeling_technique)

if __name__ == '__main__':
    main()
