import jsonlines, json, string, random, logging, copy, random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import EditBatchSampler, dict_to, scr
from tqdm import tqdm
import copy

class StereoSetDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        model_name,
        max_length=64
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.config = config
        self.iscausal = True if "gpt" in model_name or "llama" in model_name else False
        if self.iscausal:
            self.mask_token = tokenizer.eos_token
            self.mask_token_id = tokenizer.eos_token_id
        else:
            self.mask_token = tokenizer.mask_token
            self.mask_token_id = tokenizer.mask_token_id

        data = json.load(open(data_path))
        for d in data:
            self.data.append({k: d[k] for k in ["id", "target", "bias_type", "context", "data"]})
        random.shuffle(self.data)
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        b = self.data[item]
        output = {
            "context": b["context"],
            "anti-stereotype": b["data"]["anti-stereotype"]['sentence'],
            "stereotype": b["data"]["stereotype"]['sentence'],
            "unrelated": b["data"]["unrelated"]['sentence'],
        }

        return output

    def collate_fn(self, batch):
        for b in batch:
            word_idx = None
            for idx, word in enumerate(b["context"].split(" ")):
                if "BLANK" in word: 
                    word_idx = idx
                    break
            if word_idx is None:
                raise Exception("No blank word found.")
            
            anti_word = b['anti-stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            stereo_word = b['stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            unrelated_word = b['unrelated'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            
            if "roberta" in self.tokenizer.__class__.__name__.lower():
                if word_idx !=0:
                    anti_word = " " + anti_word
                    stereo_word = " " + stereo_word
                    unrelated_word = " " + unrelated_word
                    anti_insertion_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_insertion_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_insertion_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                else:
                    anti_insertion_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_insertion_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_insertion_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
            
                anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_insertion_tokens))
                stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_insertion_tokens))
                unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_insertion_tokens))
            elif "gpt" in self.tokenizer.__class__.__name__.lower():
                if " BLANK" in b['context']:
                    anti_word = " " + anti_word
                    stereo_word = " " + stereo_word
                    unrelated_word = " " + unrelated_word
                    anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                    anti_sentence = b['context'].replace(" BLANK", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace(" BLANK", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace(" BLANK", self.mask_token * len(unrelated_blank_tokens))
                else:
                    anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                    anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))
            elif "llama" in self.tokenizer.__class__.__name__.lower():
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                if " BLANK " in b['context']:
                    anti_sentence = b['context'].replace(" BLANK ", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace(" BLANK ", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace(" BLANK ", self.mask_token * len(unrelated_blank_tokens))
                elif " BLANK" in b['context']:
                    anti_sentence = b['context'].replace(" BLANK", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace(" BLANK", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace(" BLANK", self.mask_token * len(unrelated_blank_tokens))
                elif b['context'].startswith("BLANK "):
                    anti_sentence = b['context'].replace("BLANK ", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace("BLANK ", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace("BLANK ", self.mask_token * len(unrelated_blank_tokens))
                else: # start with BLANK+punctuation
                    anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
                    stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
                    unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))
            else:   # bert
                anti_blank_tokens = self.tokenizer.encode(anti_word, add_special_tokens=False)
                stereo_blank_tokens = self.tokenizer.encode(stereo_word, add_special_tokens=False)
                unrelated_blank_tokens = self.tokenizer.encode(unrelated_word, add_special_tokens=False)
                anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_blank_tokens))
                stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_blank_tokens))
                unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_blank_tokens))  
            
            b['template_word'] = {"anti": anti_word, "stereo": stereo_word, "unrelated": unrelated_word}
            b["input_sentence"] = {"anti": anti_sentence, "stereo": stereo_sentence, "unrelated": unrelated_sentence}
            # b["insertion_tokens"] = {"anti": anti_insertion_tokens, "stereo": stereo_insertion_tokens, "unrelated": unrelated_insertion_tokens}

        if not "gpt" in self.tokenizer.__class__.__name__.lower():
            anti_src = [b["input_sentence"]["anti"] for b in batch]
            stereo_src = [b["input_sentence"]["stereo"] for b in batch]
            unrelated_src = [b["input_sentence"]["unrelated"] for b in batch]
            anti_labels = [b["anti-stereotype"] for b in batch]
            stereo_labels = [b["stereotype"] for b in batch]
            unrelated_labels = [b["unrelated"] for b in batch]
        else:
            anti_src = [self.tokenizer.bos_token + b["input_sentence"]["anti"] for b in batch]
            stereo_src = [self.tokenizer.bos_token + b["input_sentence"]["stereo"] for b in batch]
            unrelated_src = [self.tokenizer.bos_token + b["input_sentence"]["unrelated"] for b in batch]
            anti_labels = [self.tokenizer.bos_token + b["anti-stereotype"] for b in batch]
            stereo_labels = [self.tokenizer.bos_token + b["stereotype"] for b in batch]
            unrelated_labels = [self.tokenizer.bos_token + b["unrelated"] for b in batch]

        res = [("anti", anti_src, "anti_labels", anti_labels), 
               ("stereo", stereo_src, "stereo_labels", stereo_labels),
               ("unrelated", unrelated_src, "unrelated_labels", unrelated_labels)]

        batches = {}
        for strsrc, srcs, labelstr, label in res:
            
            encoded = self.tokenizer(
                srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            assert self.max_length == encoded['input_ids'].shape[1]
            labels = self.tokenizer(
                label,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            assert self.max_length == labels['input_ids'].shape[1]

            if not self.iscausal:
                batches[f"{strsrc}"] = copy.deepcopy(dict(encoded.items()))
                batches[f"{strsrc}"]['labels'] = labels['input_ids']

                for idx, input_ids in enumerate(batches[f"{strsrc}"]['labels']):
                    batches[f"{strsrc}"]['labels'][idx] = torch.where(encoded['input_ids'][idx] == self.mask_token_id, input_ids, -100)
            else:
                batches[f"{strsrc}"] = copy.deepcopy(dict(labels.items()))
                for idx in range(len(labels["input_ids"])):
                    labels['input_ids'][idx] = torch.where(labels["input_ids"][idx] != self.tokenizer.pad_token_id, labels['input_ids'][idx], -100)
                    labels['input_ids'][idx][0] = -100
                batches[f"{strsrc}"]["labels"] = labels["input_ids"]

        batches["raw"] = batch
        return batches

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, seed=self.config.seed)
        while True:
            edit_idxs = sampler.stereosample(batch_size)

            toks = self.collate_fn([self[idx] for idx in edit_idxs])

            edit_inner = {"anti":toks['anti'], "stereo":toks['stereo'], "raw": toks['raw']}

            edit_outer = edit_inner

            loc = toks["unrelated"]

            cond = None

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond
            }
            yield dict_to(batch, self.config.device)

if __name__ == "__main__":
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("/newdisk3/data/xxu/mend2/bert-base-uncased")
    dataset = StereoSetDataset(tokenizer, "data/stereoset/test.json", None)
    batch = [dataset[idx] for idx in [5,6,7,8]]
    sample = next(dataset.edit_generator(4))
    print(sample)

