import random
import re
import ast
import json
import numpy as np
import torch
from torch_scatter import scatter
from sklearn.metrics import roc_curve, auc, average_precision_score, classification_report
#from sentence_transformers import SentenceTransformer, util

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#EMBED_SIZE = 1024  # Model hidden size

LOG_COLUMN_NAME = 'log:msg'
ENTITY_COLUMN_NAME = 'log:hasParameterList'
TAG_COLUMN_NAME = 'log:hastag'
LABEL_COLUMN_NAME = 'ner_tags'
INPUT_COLUMN_NAME ='log'
TARGET_COLUMN_NAME ='prompt'

TOKENIZE_PATTERN = r' |(=)|(:) |([()])|(,) |([\[\]])|([{}])|([<>])|(\.) |(\.$)'

# Templates and Regex patterns for NER task
LABEL2TEMPLATE = {
    'status code': [' is a status code entity .', '=statuscode .'],
    'operation result': [' is a operation result entity .', '=operesylt .'],
    'exit code': [' is a exit code entity .', '=exitcode .'],
    'error code': [' is a error code entity .', '=errcode .'],
    'application': [' is an application entity .', '=application .'],
    'size': [' is an size entity .', '=size .'],
    'version': [' is a version entity .', '=version .'],
    'domain': [' is a domain entity .', '=domain .'],
    'duration': [' is a duration entity .', '=duration .'],
    'path': [' is a path entity .', '=path .'],
    'ip': [' is an ip entity .', '=ip .'],
    'pid': [' is a pid entity .', '=pid .'],
    'port': [' is a port entity .', '=port .'],
    'session': [' is a session entity .', '=session .'],
    'time': [' is a time entity .', '=time .'],
    'uid': [' is a uid entity .', '=uid .'],
    'url': [' is a url entity .', '=url .'],
    'user': [' is a user entity .', '=user .'],
    'server': [' is a server entity .', '=server .'],
    'email': [' is an email entity .', '=email .'],
    'id': [' is an id entity .', '=id .'],
}

NONE2TEMPLATE = [' is not a named entity .', '=none .']


# Helper functions
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def tokenize(text):
    return list(filter(None, re.split(TOKENIZE_PATTERN, text)))


# Template generation functions
def gen_templates(text, entities, tags, strategy=0, n=8, negrate=1.5, seed=0):
    templates = [ent + LABEL2TEMPLATE[tag][strategy] for ent, tag in zip(entities, tags)]
    words = tokenize(text)

    # Generate negative samples
    ngrams = {ng for i in range(1, n+1) for ng in generate_ngrams(words, i)} - set(entities)
    neg_num = max(1, int(0.1 * len(words))) if not entities else min(int(len(entities) * negrate), len(ngrams) - 1)
    
    random.seed(seed)
    #templates += [ng + NONE2TEMPLATE[strategy] for ng in random.sample(ngrams, neg_num)]
   
    templates += [ng + NONE2TEMPLATE[strategy] for ng in random.sample(list(ngrams), neg_num)] 
    return templates


def process_corpus(data, save_path):
    with open(save_path, 'w') as f:
        value = {}
        for i, row in enumerate(data):
            exp = re.sub(';', '', row.strip())
            kv = exp.split(None, 1)
            if len(kv) == 1:
                if i > 0:
                    json.dump(value, f)
                    f.write('\n')
                value = {'eventID': kv[0]}
            else:
                key, val = kv[0], kv[1]
                if key in ['logex:hasAnnotation', 'logex:keyword']:
                    value[key] = [v.replace('"', '').strip() for v in val.split(',')]
                elif key in ['logex:hasParameterList', 'logex:hasNERtag']:
                    value[key] = ast.literal_eval(val)
                else:
                    value[key] = val.replace('"', '').strip()
        json.dump(value, f)


def gen_train_prompt(data, train_path, strategy=0, n=8, negrate=1.5, seed=0):
    with open(train_path, 'w') as f:
        for instance in data:
            templates = gen_templates(
                instance[LOG_COLUMN_NAME], 
                instance[ENTITY_COLUMN_NAME], 
                instance[TAG_COLUMN_NAME], 
                strategy=strategy, 
                n=n, 
                negrate=negrate, 
                seed=seed
            )
            for template in templates:
                json.dump({'log': instance[LOG_COLUMN_NAME], 'prompt': template}, f)
                f.write('\n')


def gen_test_labels(data, test_path):
    with open(test_path, 'w') as f:
        for instance in data:
            text, entities, tags = instance[LOG_COLUMN_NAME], instance[ENTITY_COLUMN_NAME], instance[TAG_COLUMN_NAME]
            words = tokenize(text)
            labels = ['O'] * len(words)

            for ent, tag in zip(entities, tags):
                subwords = tokenize(ent)
                n = len(subwords)
                phrase = ' '.join(subwords)

                for j in range(len(words) - n + 1):
                    if phrase == ' '.join(words[j:j+n]):
                        labels[j] = 'B-' + tag
                        for k in range(j+1, j+n):
                            labels[k] = 'I-' + tag
                        break
            json.dump({'log': text, 'tokens': words, 'ner_tags': labels, 'entities': entities, 'tags': tags}, f)
            f.write('\n')


def save_preds(test_data, pred_entities, true_entities, save_path, results, eval_dict, labeling_technique='prompt'):
    with open(save_path, 'w') as f:
        f.write(json.dumps(results) + '\n')
        for tag, res in eval_dict.items():
            f.write(f'{tag}: {json.dumps(res)}\n')

        for i, instance in enumerate(test_data):
            log = instance['log']
            tokens = instance['tokens']
            preds = [(tag, ' '.join(tokens[start:end+1])) for tag, start, end in sorted(pred_entities[i], key=lambda x: x[1])]
            labels = [(tag, ' '.join(tokens[start:end+1])) for tag, start, end in sorted(true_entities[i], key=lambda x: x[1])]

            json.dump({'log': log, 'tokens': tokens, 'preds': preds, 'labels': labels}, f)
            f.write('\n')


# Evaluation Metrics
def cal_auc_score(labels, preds):
    """Calculate AUC (Area Under the Curve) score."""
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)

def cal_aupr_score(labels, preds):
    """Calculate AUPR (Area Under the Precision-Recall Curve) score."""
    return average_precision_score(labels, preds)

def cal_accuracy(labels, preds, threshold=1):
    """Calculate accuracy given true labels and predicted values."""
    scores = np.int32(preds > threshold)
    return sum(scores == labels) / len(labels) if len(labels) else 0

def cal_cls_report(labels, preds, threshold=1, output_dict=True):
    """Generate classification report."""
    scores = np.int32(preds > threshold)
    report = classification_report(labels, scores, output_dict=output_dict)
    return scores, report

def compute_f1_score(true_entities, pred_entities):
    """Compute the F1 score for named entities."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_pred = sum(len(pred_ent) for pred_ent in pred_entities)
    nb_true = sum(len(true_ent) for true_ent in true_entities)

    precision = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1

def compute_precision(true_entities, pred_entities):
    """Compute the precision score for named entities."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_pred = sum(len(pred_ent) for pred_ent in pred_entities)

    return nb_correct / nb_pred if nb_pred > 0 else 0

def compute_recall(true_entities, pred_entities):
    """Compute the recall score for named entities."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_true = sum(len(true_ent) for true_ent in true_entities)

    return nb_correct / nb_true if nb_true > 0 else 0

def evaluation_report(true_entities, pred_entities, entity_count):
    """Generate a detailed evaluation report for each entity."""
    eval_dict = {tag: {'T': 0, 'P': 0, 'TP': 0, 'pre': 0, 'rec': 0, 'f1': 0} for tag in entity_count}

    for true_ent, pred_ent in zip(true_entities, pred_entities):
        for tup in true_ent:
            if tup[0] in eval_dict:
                eval_dict[tup[0]]['T'] += 1
        for tup in pred_ent:
            if tup[0] in eval_dict:
                eval_dict[tup[0]]['P'] += 1
        for tup in true_ent & pred_ent:
            if tup[0] in eval_dict:
                eval_dict[tup[0]]['TP'] += 1

    for tag, value in eval_dict.items():
        value['pre'] = value['TP'] / value['P'] if value['P'] else 0
        value['rec'] = value['TP'] / value['T'] if value['T'] else 0
        value['f1'] = 2 * value['pre'] * value['rec'] / (value['pre'] + value['rec']) if value['pre'] + value['rec'] else 0

    return eval_dict


# class SentenceEncoder:
#     def __init__(self, device='cuda'):
#         """Initialize a sentence encoder using the paraphrase-distilroberta-base-v1 model."""
#         self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', device=device)

#     def encode(self, sentences):
#         """Encode sentences into embeddings. Supports both string and list input."""
#         if isinstance(sentences, str):
#             sentences = [sentences]
#         return self.model.encode(sentences, convert_to_tensor=True)

#     def get_similarity(self, sentence1, sentence2):
#         """Compute cosine similarity between two sentences."""
#         embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
#         return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

#     def find_best_sim(self, original_sentence, adversarial_sentences, find_min=False):
#         """Find the adversarial sentence with the highest or lowest similarity to the original sentence."""
#         ori_embedding = self.model.encode(original_sentence, convert_to_tensor=True)
#         adv_embeddings = self.model.encode(adversarial_sentences, convert_to_tensor=True)

#         best_sim = 10 if find_min else -10
#         best_adv, best_index = None, None

#         for i, adv_embedding in enumerate(adv_embeddings):
#             sim = util.pytorch_cos_sim(ori_embedding, adv_embedding).item()
#             if (find_min and sim < best_sim) or (not find_min and sim > best_sim):
#                 best_sim = sim
#                 best_adv = adversarial_sentences[i]
#                 best_index = i

#         return best_adv, best_index, best_sim

