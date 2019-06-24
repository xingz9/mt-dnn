import os
import json
import tqdm
import pickle
import re
import collections
import argparse
from sys import path
path.append(os.getcwd())
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import NER_LabelMapper
from data_utils.glue_utils import *

is_cased = False
if is_cased:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
else:
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

logger = create_logger(__name__, to_disk=True, log_file='bert_ner_data_proc_512_cased.log')

MAX_SEQ_LEN = 512

def load_conll_ner(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label= []
    with open(file, encoding="utf8") as f:
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    sample = {'uid': cnt, 'premise': sentence, 'label': label}
                    rows.append(sample)
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1][:-1])
            cnt += 1
        if len(sentence) > 0:
            sample = {'uid': cnt, 'premise': sentence, 'label': label}
    return rows


def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True):
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            mylabels = sample['label']
            tokens = []
            labels = []
            for i, word in enumerate(premise):
                subwords = bert_tokenizer.tokenize(word)
                tokens.extend(subwords)
                for j in range(len(subwords)):
                    if j == 0:
                        labels.append(mylabels[i])
                    else:
                        labels.append('X')

            if len(premise) >  max_seq_len - 2:
                tokens = tokens[:max_seq_len - 2]
                labels = labels[:max_seq_len - 2]
            labels = ['[CLS]'] + labels[:max_seq_len - 2] + ['[SEP]']
            label = [NER_LabelMapper[lab] for lab in labels]
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            assert len(label) == len(input_ids)
            type_ids = [0] * ( len(tokens) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing English NER dataset.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    data_dir = args.data_dir
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    train_path = os.path.join(data_dir, 'train.txt')
    dev_path = os.path.join(data_dir, 'valid.txt')
    test_path = os.path.join(data_dir, 'test.txt')

    train_data = load_conll_ner(train_path)
    dev_data = load_conll_ner(dev_path)
    test_data = load_conll_ner(test_path)
    logger.info('Loaded {} NER train samples'.format(len(train_data)))
    logger.info('Loaded {} NER dev samples'.format(len(dev_data)))
    logger.info('Loaded {} NER test samples'.format(len(test_data)))

    bert_root = os.path.join(data_dir, 'bert_cased')
    if not os.path.isdir(bert_root):
        os.mkdir(bert_root)

    my_cased = 'cased' if is_cased else 'uncased'
    train_fout = os.path.join(bert_root, 'ener_{}_train.json'.format(my_cased))
    dev_fout = os.path.join(bert_root, 'ener_{}_dev.json'.format(my_cased))
    test_fout = os.path.join(bert_root, 'ener_{}_test.json'.format(my_cased))

    build_data(train_data, train_fout)
    build_data(dev_data, dev_fout)
    build_data(test_data, test_fout)
    logger.info('done with NER')


if __name__ == '__main__':
    args = parse_args()
    main(args)
