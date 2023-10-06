# DRETC

## 1. Environments

- python (3.6.8)
- cuda (10.1)

## 2. Dependencies

- numpy (1.17.3)
- torch (1.6.0)
- transformers (3.5.1)
- pandas (1.1.5)
- scikit-learn (0.23.2)

## 3. Preparation

- Download [DocRED](https://github.com/thunlp/DocRED) dataset
- Put all the `train_annotated.json`, `dev.json`, `test.json`,`word2id.json`,`vec.npy`,`rel2id.json`,`ner2id` into the directory `data/`

```bash
>> python preprocess.py
```

## 4. Run

```bash
>> python main.py
```
