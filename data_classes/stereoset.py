import json, string, random, logging, copy, random
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
        self._tokenizer = tokenizer
        self.data = []
        self.config = config
        self.iscausal = True if "gpt" in model_name or "llama" in model_name else False
        if not self.iscausal:
            self.mask_token = tokenizer.mask_token
            self.mask_token_id = tokenizer.mask_token_id

        data = json.load(open(data_path))
        for d in data:
            self.data.append({k: d[k] for k in ["id", "target", "bias_type", "context", "data"]})
        # random.shuffle(self.data)
        
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
        new_batch = []
        for b in batch:
            word_idx = None
            blank_cnt = 0
            strange = False
            for idx, word in enumerate(b["context"].split(" ")):
                if "BLANK" in word: 
                    word_idx = idx
                    blank_cnt += 1
                if ".BLANK" in word or "`BLANK" in word:
                    strange = True
            if strange:
                continue
            if blank_cnt > 1:
                continue
            if word_idx is None:
                raise Exception("No BLANK found.")
            
            anti_word = b['anti-stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            stereo_word = b['stereotype'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            unrelated_word = b['unrelated'].split(" ")[word_idx].translate(str.maketrans('', '', string.punctuation))
            
            if not self.iscausal:
                if "roberta" in self._tokenizer.__class__.__name__.lower():
                    if b['context'].startswith("BLANK"):
                        anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                        stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                        unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                        anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_insertion_tokens))
                        stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_insertion_tokens))
                        unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_insertion_tokens))
                    else:
                        anti_word = " " + anti_word
                        stereo_word = " " + stereo_word
                        unrelated_word = " " + unrelated_word
                        anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                        stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                        unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                        anti_sentence = b['context'].replace(" BLANK", self.mask_token * len(anti_insertion_tokens))
                        stereo_sentence = b['context'].replace(" BLANK", self.mask_token * len(stereo_insertion_tokens))
                        unrelated_sentence = b['context'].replace(" BLANK", self.mask_token * len(unrelated_insertion_tokens))
                else:   # bert
                    anti_insertion_tokens = self._tokenizer.encode(anti_word, add_special_tokens=False)
                    stereo_insertion_tokens = self._tokenizer.encode(stereo_word, add_special_tokens=False)
                    unrelated_insertion_tokens = self._tokenizer.encode(unrelated_word, add_special_tokens=False)
                    anti_sentence = b['context'].replace("BLANK", self.mask_token * len(anti_insertion_tokens))
                    stereo_sentence = b['context'].replace("BLANK", self.mask_token * len(stereo_insertion_tokens))
                    unrelated_sentence = b['context'].replace("BLANK", self.mask_token * len(unrelated_insertion_tokens))
                b['template_word_ids'] = {"anti": anti_insertion_tokens, "stereo": stereo_insertion_tokens, "unrelated": unrelated_insertion_tokens}
            else:
                anti_sentence = self._tokenizer.eos_token + b['context'].replace("BLANK", anti_word)
                stereo_sentence = self._tokenizer.eos_token + b['context'].replace("BLANK", stereo_word)
                unrelated_sentence = self._tokenizer.eos_token + b['context'].replace("BLANK", unrelated_word)
            b["input_sentence"] = {"anti": anti_sentence, "stereo": stereo_sentence, "unrelated": unrelated_sentence}   # with [MASK] for MLM, with eos for CLM
            # b["insertion_tokens"] = {"anti": anti_insertion_tokens, "stereo": stereo_insertion_tokens, "unrelated": unrelated_insertion_tokens}
            new_batch.append(b)
        batches = {}
        if not self.iscausal:
            # anti
            anti_srcs = [b["input_sentence"]["anti"] for b in new_batch]
            anti_inputs = self._tokenizer(
                anti_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            anti_inputs['labels'] = copy.deepcopy(anti_inputs['input_ids'])
            for batchin_idx, input_ids in enumerate(anti_inputs['labels']):
                mask_idxs = []
                for mask_idx, input_id  in enumerate(input_ids):
                    if input_id == self.mask_token_id:
                        mask_idxs.append(mask_idx)
                assert len(mask_idxs) == len(new_batch[batchin_idx]['template_word_ids']['anti'])
                for template_idx, mask_idx in enumerate(mask_idxs):
                    anti_inputs['labels'][batchin_idx][mask_idx] = new_batch[batchin_idx]['template_word_ids']['anti'][template_idx]
                anti_inputs['labels'][batchin_idx] = torch.where(anti_inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, anti_inputs['labels'][batchin_idx], -100)
            batches['anti'] = anti_inputs

            # stereo
            stereo_srcs = [b["input_sentence"]["stereo"] for b in new_batch]
            stereo_inputs = self._tokenizer(
                stereo_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            stereo_inputs['labels'] = copy.deepcopy(stereo_inputs['input_ids'])
            for batchin_idx, input_ids in enumerate(stereo_inputs['labels']):
                mask_idxs = []
                for mask_idx, input_id in enumerate(input_ids):
                    if input_id == self.mask_token_id:
                        mask_idxs.append(mask_idx)
                assert len(mask_idxs) == len(new_batch[batchin_idx]['template_word_ids']['stereo'])
                for template_idx, mask_idx in enumerate(mask_idxs):
                    stereo_inputs['labels'][batchin_idx][mask_idx] = new_batch[batchin_idx]['template_word_ids']['stereo'][template_idx]
                stereo_inputs['labels'][batchin_idx] = torch.where(stereo_inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, stereo_inputs['labels'][batchin_idx], -100)
            batches['stereo'] = stereo_inputs

            # unrelated
            unrelated_srcs = [b["input_sentence"]["unrelated"] for b in new_batch]
            unrelated_inputs = self._tokenizer(
                unrelated_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            unrelated_inputs['labels'] = copy.deepcopy(unrelated_inputs['input_ids'])
            for batchin_idx, input_ids in enumerate(unrelated_inputs['labels']):
                mask_idxs = []
                for mask_idx, input_id in enumerate(input_ids):
                    if input_id == self.mask_token_id:
                        mask_idxs.append(mask_idx)
                assert len(mask_idxs) == len(new_batch[batchin_idx]['template_word_ids']['unrelated'])
                for template_idx, mask_idx in enumerate(mask_idxs):
                    unrelated_inputs['labels'][batchin_idx][mask_idx] = new_batch[batchin_idx]['template_word_ids']['unrelated'][template_idx]
                unrelated_inputs['labels'][batchin_idx] = torch.where(unrelated_inputs['input_ids'][batchin_idx] == self._tokenizer.mask_token_id, unrelated_inputs['labels'][batchin_idx], -100)
            batches['unrelated'] = unrelated_inputs
        else:
            # gpt2
            # anti
            anti_srcs = [b["input_sentence"]["anti"] for b in new_batch]
            anti_inputs = self._tokenizer(
                anti_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            anti_inputs['labels'] = copy.deepcopy(anti_inputs['input_ids'])
            for batchin_idx in range(len(anti_inputs['labels'])):
                anti_inputs['labels'][batchin_idx] = torch.where(anti_inputs['labels'][batchin_idx]!=self._tokenizer.pad_token_id, anti_inputs['input_ids'][batchin_idx], -100)
                anti_inputs['labels'][batchin_idx][0] = -100
            batches['anti'] = anti_inputs

            # stereo
            stereo_srcs = [b["input_sentence"]["stereo"] for b in new_batch]
            stereo_inputs = self._tokenizer(
                stereo_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            stereo_inputs['labels'] = copy.deepcopy(stereo_inputs['input_ids'])
            for batchin_idx in range(len(stereo_inputs['labels'])):
                stereo_inputs['labels'][batchin_idx] = torch.where(stereo_inputs['labels'][batchin_idx]!=self._tokenizer.pad_token_id, stereo_inputs['input_ids'][batchin_idx], -100)
                stereo_inputs['labels'][batchin_idx][0] = -100
            batches['stereo'] = stereo_inputs

            # unrelated
            unrelated_srcs = [b["input_sentence"]["unrelated"] for b in new_batch]
            unrelated_inputs = self._tokenizer(
                unrelated_srcs,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            unrelated_inputs['labels'] = copy.deepcopy(unrelated_inputs['input_ids'])
            for batchin_idx in range(len(unrelated_inputs['labels'])):
                unrelated_inputs['labels'][batchin_idx] = torch.where(unrelated_inputs['labels'][batchin_idx]!=self._tokenizer.pad_token_id, unrelated_inputs['labels'][batchin_idx], -100)
                unrelated_inputs['labels'][batchin_idx][0] = -100
            batches['unrelated'] = unrelated_inputs


        batches["raw"] = new_batch
        return batches

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, seed=self.config.seed)
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
    tokenizer = BertTokenizerFast.from_pretrained("/newdisk3/data/xxu/bias-bench/gpt2")
    dataset = StereoSetDataset(tokenizer, "data/stereoset/test.json", None)
    # batch = [dataset[idx] for idx in [5,6,7,8]]
    sample = next(dataset.edit_generator(4))
    print(sample)

