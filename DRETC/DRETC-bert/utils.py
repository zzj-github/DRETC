
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import logging
from collections import Counter, OrderedDict, defaultdict
from transformers import BertTokenizer
import tqdm

RELATION_NUM = 97
dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


def get_logger(pathname):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'

    def __init__(self, rel2id, ner2id, frequency=0):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}
        self.token2count = {self.PAD: 1000, self.UNK: 1000}
        self.frequency = frequency

        self.rel2id = rel2id
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.ner2id = ner2id

    def add_token(self, token):
        token = token.lower()
        if token in self.token2id:
            self.token2count[token] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token
            self.token2count[token] = 1

        assert token == self.id2token[self.token2id[token]]

    def remove_low_frequency_token(self):
        new_token2id = {self.PAD: 0, self.UNK: 1}
        new_id2token = {0: self.PAD, 1: self.UNK}

        for token in self.token2id:
            if self.token2count[token] > self.frequency and token not in new_token2id:
                new_token2id[token] = len(new_token2id)
                new_id2token[new_token2id[token]] = token

        self.token2id = new_token2id
        self.id2token = new_id2token

    def __len__(self):
        return len(self.token2id)

    def encode(self, text):
        return [self.token2id.get(x.lower(), 1) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


def collate_fn(data):
    doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask, \
    doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask, title = map(list, zip(*data))

    batch_size = len(doc_inputs)
    max_tok = np.max([x.shape[0] for x in doc_inputs]) - 2

    doc_inputs = pad_sequence(doc_inputs, True)
    psn_inputs = pad_sequence(psn_inputs, True)
    ner_inputs = pad_sequence(ner_inputs, True)

    ent_num = [x.shape[0] for x in doc2ent_mask]
    men_num = [x.shape[0] for x in doc2men_mask]

    max_ent = np.max(ent_num)
    max_men = np.max(men_num)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_men, max_men), dtype=torch.long)
    dis_inputs = fill(dis_inputs, dis_mat)
    d2e_mat = torch.zeros((batch_size, max_ent, max_tok), dtype=torch.bool)
    doc2ent_mask = fill(doc2ent_mask, d2e_mat)
    d2m_mat = torch.zeros((batch_size, max_men, max_tok), dtype=torch.bool)
    doc2men_mask = fill(doc2men_mask, d2m_mat)
    m2e_mat = torch.zeros((batch_size, max_ent, max_men), dtype=torch.bool)
    men2ent_mask = fill(men2ent_mask, m2e_mat)
    e2e_mat = torch.zeros((batch_size, max_ent, max_ent), dtype=torch.bool)
    ent2ent_mask = fill(ent2ent_mask, e2e_mat)
    m2m_mat = torch.zeros((batch_size, max_men, max_men), dtype=torch.bool)
    men2men_mask = fill(men2men_mask, m2m_mat)

    rel_mat = torch.zeros((batch_size, max_ent, max_ent, RELATION_NUM), dtype=torch.float)
    for j, x in enumerate(rel_labels):
        rel_mat[j, :x.shape[0], :x.shape[1], :] = x
    rel_labels = rel_mat

    intrain_mat = torch.zeros((batch_size, max_ent, max_ent, RELATION_NUM), dtype=torch.bool)
    for j, x in enumerate(intrain_mask):
        intrain_mat[j, :x.shape[0], :x.shape[1], :] = x
    intrain_mask = intrain_mat

    return doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask,\
    doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask, title


class BERTRelationDataset(Dataset):
    def __init__(self, doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask,
                 doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask, title):
        self.doc_inputs = doc_inputs
        self.psn_inputs = psn_inputs
        self.ner_inputs = ner_inputs
        self.dis_inputs = dis_inputs
        self.rel_labels = rel_labels
        self.intrain_mask = intrain_mask
        self.doc2ent_mask = doc2ent_mask
        self.doc2men_mask = doc2men_mask
        self.men2ent_mask = men2ent_mask
        self.ent2ent_mask = ent2ent_mask
        self.men2men_mask = men2men_mask
        self.title = title

    def __getitem__(self, item):
        return torch.LongTensor(self.doc_inputs[item]), \
               torch.LongTensor(self.psn_inputs[item]), \
               torch.LongTensor(self.ner_inputs[item]), \
               torch.LongTensor(self.dis_inputs[item]), \
               torch.FloatTensor(self.rel_labels[item]), \
               torch.BoolTensor(self.intrain_mask[item]), \
               torch.BoolTensor(self.doc2ent_mask[item]), \
               torch.BoolTensor(self.doc2men_mask[item]), \
               torch.BoolTensor(self.men2ent_mask[item]), \
               torch.BoolTensor(self.ent2ent_mask[item]), \
               torch.BoolTensor(self.men2men_mask[item]), \
               self.title[item]

    def __len__(self):
        return len(self.doc_inputs)


def process_bert(data, vocab, tokenizer, is_train):
    doc_inputs = []
    psn_inputs = []
    ner_inputs = []
    dis_inputs = []
    rel_labels = []
    intrain_mask = []
    doc2ent_mask = []
    doc2men_mask = []
    men2ent_mask = []
    men2men_mask = []
    ent2ent_mask = []

    title = []
    
    for index, doc in tqdm.tqdm(enumerate(data), total=len(data)):
        tok_list = [x for x in doc['sents']]
        pos_list = [x for x in doc['sents']]
        _doc_inputs = []
        _psn_inputs = []
        _pos_inputs = []
        position = []

        i = 0
        j = 1
        for t, p in zip(tok_list, pos_list):
            new_toks = tokenizer.encode(t, add_special_tokens=False)
            _doc_inputs += new_toks
            _psn_inputs += [j] * len(new_toks)
            position.append([x + i for x in range(len(new_toks))])
            i += len(new_toks)

            if len(new_toks) > 0:
                j += 1
            if j >= 511:
                break

        men_num = len(doc['mentions'])
        ent_num = len(doc['vertexSet'])
        doc_len = len(_doc_inputs)

        _doc_inputs = [tokenizer.cls_token_id] + _doc_inputs + [tokenizer.sep_token_id]
        _psn_inputs = [0] + _psn_inputs + [j]

        _ner_inputs = np.zeros((doc_len,), dtype=np.int)
        _dis_inputs = np.zeros((men_num, men_num), dtype=np.int)
        _ref_inputs = np.ones((men_num, men_num), dtype=np.int)
        _rel_labels = np.zeros((ent_num, ent_num, RELATION_NUM), dtype=np.int)
        _intrain_mask = np.zeros((ent_num, ent_num, RELATION_NUM), dtype=np.bool)
        _doc2ent_mask = np.zeros((ent_num, doc_len), dtype=np.bool)
        _doc2men_mask = np.zeros((men_num, doc_len), dtype=np.bool)
        _men2ent_mask = np.zeros((ent_num, men_num), dtype=np.int)
        _ent2ent_mask = np.ones((ent_num, ent_num), dtype=np.int)
        _men2men_mask = np.ones((men_num, men_num), dtype=np.int)

        men_id = 0
        for ent_id, entity in enumerate(doc['vertexSet']):
            for mention in entity:
                s_pos, e_pos = mention['pos']
                s_pos = position[s_pos][0] if len(position[s_pos]) else position[s_pos+1][0]
                e_pos = position[e_pos-1][-1] + 1 if len(position[e_pos-1]) else position[e_pos-2][-1] + 1

                ner_id = vocab.ner2id[mention['type']]

                _ner_inputs[s_pos: e_pos] = ner_id
                _doc2ent_mask[ent_id][s_pos: e_pos] = 1
                _doc2men_mask[men_id][s_pos: e_pos] = 1
                men_id += 1

        for i, men in enumerate(doc["mentions"]):
            pos = men["pos"][0]
            # pos = position[pos][0]
            ent_id = men["ent_id"]
            _dis_inputs[i, :] += pos
            _dis_inputs[:, i] -= pos
            _men2ent_mask[ent_id, i] = 1

        _doc2men_mask = _doc2men_mask[np.array([x["id"] for x in doc["mentions"]])]

        for i in range(men_num):
            for j in range(men_num):
                if _dis_inputs[i, j] < 0:
                    _dis_inputs[i, j] = dis2idx[-_dis_inputs[i, j]] + 9
                else:
                    _dis_inputs[i, j] = dis2idx[_dis_inputs[i, j]]
                if doc["mentions"][i]["sent_id"] == doc["mentions"][j]["sent_id"]:
                    _ref_inputs[i, j] = 2
                    _ref_inputs[j, i] = 2

        _dis_inputs += 1
        _rel_labels[..., 0] = 1
        for label in doc.get('labels', []):
            h, t, r = label['h'], label['t'], vocab.rel2id[label['r']]

            _rel_labels[h, t, 0] = 0
            _rel_labels[h, t, r] = 1
            if not is_train and label["intrain"]:
                _intrain_mask[h, t, r] = 1

        doc_inputs.append(_doc_inputs)
        psn_inputs.append(_psn_inputs)
        ner_inputs.append(_ner_inputs)
        dis_inputs.append(_dis_inputs)
        rel_labels.append(_rel_labels)
        intrain_mask.append(_intrain_mask)
        doc2ent_mask.append(_doc2ent_mask)
        doc2men_mask.append(_doc2men_mask)
        men2ent_mask.append(_men2ent_mask)
        ent2ent_mask.append(_ent2ent_mask)
        men2men_mask.append(_men2men_mask)
        title.append(doc["title"])

    return doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask, doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask, title


def load_data(load_emb=True):
    with open('./data/docred_pre_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/docred_pre_dev.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/docred_pre_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open('./data/rel2id.json', 'r', encoding='utf-8') as f:
        rel2id = json.load(f)
    with open('./data/ner2id.json', 'r', encoding='utf-8') as f:
        ner2id = json.load(f)
    vocab = Vocabulary(rel2id, ner2id)

    if load_emb:
        with open("./data/word2id.json", 'r', encoding='utf-8') as f:
            token2id = json.load(f)
            vocab.token2id = token2id
            vocab.id2token = {v: k for k, v in token2id.items()}
    else:
        for doc in train_data:
            for sent in doc['sents']:
                for token in sent:
                    vocab.add_token(token)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained("./bert/bert-base-uncased")

    train_dataset = BERTRelationDataset(*process_bert(train_data, vocab, tokenizer, is_train=True))
    dev_dataset = BERTRelationDataset(*process_bert(dev_data, vocab, tokenizer, is_train=False))
    test_dataset = BERTRelationDataset(*process_bert(test_data, vocab, tokenizer, is_train=False))
    return train_dataset, dev_dataset, test_dataset, vocab

