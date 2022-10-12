import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


def get_preprocess_function(tokenizer):
    def preprocess_function(examples):
        texts = examples["title"]
        result = tokenizer(texts, padding=False, max_length=128, truncation=True)
        return result

    return preprocess_function

def get_word_embedding(args, title_list):
    all_glove_embedding = dict()
    with open(args.embedding_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, "loading embedding"):
            cur_words = line.strip().split(' ')
            if len(cur_words) != 301:
                print(len(cur_words), 'something is wrong')
                continue
            tmp_vector = [float(w) for w in cur_words[1:]]
            tmp_vector_np = np.asarray(tmp_vector)
            all_glove_embedding[cur_words[0]] = tmp_vector_np

    encoding_list = []
    for tmp_p in tqdm(title_list, "get title embeddings"):
        words = tmp_p.split(' ')
        tmp_vector = np.zeros(300)
        counter = 0
        for w in words:
            if w in all_glove_embedding:
                tmp_vector += all_glove_embedding[w]
                counter += 1
        if counter > 0:
            tmp_vector /= counter
        encoding_list.append(tmp_vector)
    encoding_list = np.array(encoding_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(encoding_list.shape)
    return torch.from_numpy(encoding_list).to(device)


def get_LM_embedding(args, title_list):

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, from_tf=False, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_datasets = Dataset.from_dict({"title": title_list})

    preprocess_function = get_preprocess_function(tokenizer)
    title_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    title_dataloader = DataLoader(
        title_dataset, shuffle=False, collate_fn=data_collator, batch_size=64
    )

    encoding_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(title_dataloader, "encoding"):
            batch = {key: value.to(device) for key, value in batch.items()}
            encoding = model(**batch)
            pooled_encoding = encoding["pooler_output"]
            encoding_list.append(pooled_encoding)
    encoding_list = torch.cat(encoding_list)
    return encoding_list


def cosine_similarity(train_emebdding, valid_embedding, test_embedding):
    index_list = []
    normalized_train = train_emebdding / torch.norm(train_emebdding, dim=1, keepdim=True)
    for eval_embedding in [valid_embedding, test_embedding]:
        normalized_eval = eval_embedding / torch.norm(eval_embedding, dim=1, keepdim=True)
        sim = torch.matmul(normalized_eval, normalized_train.transpose(-1, -2))
        index = sim.argmax(dim=1)
        index_list.append(index.cpu().numpy())
    return index_list[0], index_list[1]


def gather_predicted_data(title_list, subevent_list, offset_list, valid_index_list, test_index_list):
    valid_start, valid_end = offset_list[1], offset_list[2]
    test_start, test_end = offset_list[2], offset_list[3]
    train_subevent_list = subevent_list[offset_list[0]: offset_list[1]]

    valid_data_list = [{"input": title, "pred": train_subevent_list[index], "label": subevent}
                 for title, subevent, index in zip(title_list[valid_start: valid_end],
                                                   subevent_list[valid_start: valid_end],
                                                   valid_index_list)]
    test_data_list = [{"input": title, "pred": train_subevent_list[index], "label": subevent}
                 for title, subevent, index in zip(title_list[test_start: test_end],
                                                   subevent_list[test_start: test_end],
                                                   test_index_list)]
    return valid_data_list, test_data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help="the path of dataset dir", type=str,
                        default="/home/data/zwanggy/APSI_data/48_len_inductive_data/sentence_train_42.json")
    parser.add_argument('--valid_file', help="the path of dataset dir", type=str,
                        default="/home/data/zwanggy/APSI_data/OOD_data/descript.json")
    parser.add_argument('--test_file', help="the path of dataset dir", type=str,
                        default="/home/data/zwanggy/APSI_data/OOD_data/descript.json")
    parser.add_argument("--output_dir", help="the path of the output file", type=str,
                        default='/home/data/zwanggy/event_outputs/top1_sentence_bert_descript')
    parser.add_argument("--sim_func", type=str, default="sbert",
                        choices=["sbert", "glove"],
                        help="the similarity function")
    parser.add_argument("--model_name_or_path", default="sentence-transformers/bert-base-wikipedia-sections-mean-tokens",
                        type=str, help="path to load the BERT/RoBERTa model")
    parser.add_argument("--embedding_path",
                        default="/home/data/corpora/word_embeddings/english_embeddings/glove/glove.6B.300d.txt",
                        type=str, help="path to glove embedding")

    args = parser.parse_args()

    generation_path = os.path.join(args.output_dir, "generation")
    if not os.path.exists(generation_path):
        os.makedirs(generation_path)

    all_title_list = list()
    all_subevent_list = list()
    offset_list = [0]

    for file in [args.train_file, args.valid_file, args.test_file]:
        with open(file, 'r') as fin:
            data_list = [json.loads(line) for line in fin]
            # data_list = data_list[: 5000]
            for tmp_process in data_list:
                all_title_list.append(tmp_process['title'])
                all_subevent_list.append(tmp_process["subevents"])
        offset_list.append(len(all_title_list))

    # get embedding here
    if args.sim_func == "sbert":
        encoding_list = get_LM_embedding(args, title_list=all_title_list)
    elif args.sim_func == "glove":
        encoding_list = get_word_embedding(args, title_list=all_title_list)
    else:
        encoding_list = None

    if encoding_list is not None:
        train_encoding = encoding_list[offset_list[0]: offset_list[1]]
        valid_encoding = encoding_list[offset_list[1]: offset_list[2]]
        test_encoding = encoding_list[offset_list[2]: offset_list[3]]
    else:
        train_encoding, valid_encoding, test_encoding = None, None, None

    # get similarity
    valid_index, test_index = cosine_similarity(train_encoding, valid_encoding, test_encoding)

    valid_pred, test_pred = gather_predicted_data(
        all_title_list, all_subevent_list, offset_list, valid_index, test_index)


    def output_generation(pred_list, mode):
        with open(os.path.join(generation_path, "{}.json".format(mode)), "w") as fout:
            for p in tqdm(pred_list, "output generations"):
                fout.write(json.dumps(p) + "\n")

    if valid_pred is not None:
        output_generation(valid_pred, "valid")
    if test_pred is not None:
        output_generation(test_pred, "test")

    print('end')