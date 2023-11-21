import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import masked_log_probs
from utils import _logits, shift_targets



class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor, tokenizer):
        super().__init__()

        self.model = model
        self.config = config
        self.model_constructor = model_constructor
        self.tokenizer=tokenizer
        self.iscausal = "gpt" in model.__class__.__name__.lower() or "llama" in model.__class__.__name__.lower()
        if self.iscausal:
            self.loc_loss_fn = self._loc_causal_loss_fn
            self.edit_loss_fn = self._edit_causal_loss_fn
        else:
            self.loc_loss_fn = self._loc_loss_fn
            self.edit_loss_fn = self._edit_loss_fn
                
    def _loc_loss_fn(self, output, targ):
        output_soft = F.log_softmax(output.logits, dim=-1)
        n_tokens = 0

        def get_loc(p,t):
            cnt_tokens = 0
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    cnt_tokens += 1
            ele_probs = p[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(output_soft.device)).diag()
            return ele_probs.mean(), cnt_tokens
        
        loc_score, cnt = get_loc(output_soft[0], targ[0])
        n_tokens += cnt
        loc_score = loc_score.unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(output_soft, targ)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_loc(p, t)
            n_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)

        return {
            "avg_log_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": n_tokens, 
            "loc_score": loc_score
        }

    
    def _edit_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo):   # pred_anti: (batch_size, seq_len, vocab_size), targ_anti: (batch_size, seq_len)        

        anti_soft = F.log_softmax(pred_anti, dim=-1)
        # anti_n_tokens = 0

        def get_edit(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = [] # ids of [MASK] target tokens
            for i in range(len(t)): # seq_len
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    # anti_n_tokens += 1
            ele_probs = p[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
            return ele_probs.mean()
        
        anti_score = get_edit(anti_soft[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(anti_soft, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        

        stereo_soft = F.log_softmax(pred_stereo, dim=-1)
        # stereo_n_tokens = 0
        stereo_score = get_edit(stereo_soft[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(stereo_soft, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        # stereo_score = torch.tensor(stereo_scores, requires_grad=True).to(pred_stereo.device)  # (batch_size, 1)

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss = kl_loss(input=anti_score, target=stereo_score) + kl_loss(input=stereo_score, target=anti_score)
        pro_num = 0
        total = 0
        for anti, pro in zip(anti_score, stereo_score):
            if pro > anti:
                pro_num += 1
            total += 1
        ss_score = pro_num / total
        return {
            "ss_score": ss_score,
            "loss": loss,
            "anti_log_prob": anti_score.mean(),
            "stereo_log_prob": stereo_score.mean(),
            "anti_score": anti_score, 
            "stereo_score": stereo_score
        }
    
    def _edit_causal_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo, shift=True):
        # start_token = torch.tensor(self.tokenizer.bos_token_id).to(pred_anti.device).unsqueeze(0)
        # initial_token_probabilities = self.model(start_token)
        # initial_token_probabilities = initial_token_probabilities[0].log_softmax(dim=-1)

        # anti_probs = []
        # for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
        #     joint_sentence_probability = [
        #         initial_token_probabilities[0, t[0]].unsqueeze(0)]
        #     output = p.log_softmax(dim=-1)
        #     pad_idx = torch.where(t == -100)[0][0]
        #     tokens = t[:pad_idx]
        #     for idx in range(1, len(tokens)):
        #         joint_sentence_probability.append(
        #             output[idx-1, tokens[idx]].unsqueeze(0))
        #     assert len(tokens) == len(joint_sentence_probability)
        #     mean_probs = torch.cat(tuple(joint_sentence_probability), dim=0).mean()
        #     anti_probs.append(mean_probs.unsqueeze(0))
        # anti_score = torch.cat(tuple(anti_probs), dim=0).to(pred_anti.device)
        # anti_score.requires_grad_()

        # stereo_probs = []
        # for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
        #     joint_sentence_probability = [
        #         initial_token_probabilities[0, t[0]].unsqueeze(0)]
        #     output = p.log_softmax(dim=-1)
        #     pad_idx = torch.where(t == -100)[0][0]
        #     tokens = t[:pad_idx]
        #     for idx in range(1, len(tokens)):
        #         joint_sentence_probability.append(
        #             output[idx-1, tokens[idx]].unsqueeze(0))
        #     assert len(tokens) == len(joint_sentence_probability)
        #     mean_probs = torch.cat(tuple(joint_sentence_probability), dim=0).mean()
        #     stereo_probs.append(mean_probs.unsqueeze(0))
        # stereo_score = torch.cat(tuple(stereo_probs), dim=0).to(pred_stereo.device)
        # stereo_score.requires_grad_()
        
        
        # NULL_TOKEN = 0  # a placeholder used for masked target locations

        if shift and pred_anti.dim() == 3 and pred_stereo.dim() == 3:  # Dealing with sequences
            pred_anti = pred_anti[:, :-1]  # Remove last prediction in sequence
            targ_anti = targ_anti[:, 1:]  # Shift to align predictions and targets
            pred_stereo = pred_stereo[:, :-1]
            targ_stereo = targ_stereo[:, 1:]

        def get_score(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
            ele_probs = p.log_softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()
            return ele_probs.mean()
        
        anti_score = get_score(pred_anti[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_score(p, t).unsqueeze(0)), dim=0)   # (batch_size, )
          

        stereo_score = get_score(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_score(p, t).unsqueeze(0)), dim=0)   # (batch_size, )

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss = kl_loss(input=anti_score, target=stereo_score) + kl_loss(input=stereo_score, target=anti_score)

        pro_num = 0
        total = 0
        for anti, pro in zip(anti_score, stereo_score):
            if pro > anti:
                pro_num += 1
            total += 1
        ss_score = pro_num / total
        return {
            "ss_score": ss_score,
            "loss": loss,
            "anti_log_prob": anti_score.mean(),
            "stereo_log_prob": stereo_score.mean(),
            "anti_score": anti_score,
            "stereo_score": stereo_score
        }
    

    def _loc_causal_loss_fn(self, output, targ, shift=True):
        # start_token = torch.tensor(self.tokenizer.bos_token_id).to(output.device).unsqueeze(0)
        # initial_token_probabilities = self.model(start_token)
        # initial_token_probabilities = initial_token_probabilities[0].log_softmax(dim=-1)

        # output_probs = []
        # total_tokens = 0
        # for batch_idx, (p, t) in enumerate(zip(output, targ)):
        #     joint_sentence_probability = [
        #         initial_token_probabilities[0, t[0]].unsqueeze(0)]
        #     pad_idx = torch.where(t == -100)[0][0]
        #     pad_idx = t.index(-100)
        #     tokens = t[:pad_idx]
        #     for idx in range(1, len(tokens)):
        #         joint_sentence_probability.append(
        #             output[idx-1, tokens[idx]].unsqueeze(0))
        #     assert len(tokens) == len(joint_sentence_probability)
        #     total_tokens += len(tokens)
        #     mean_probs = torch.cat(tuple(joint_sentence_probability), dim=0).mean()
        #     output_probs.append(mean_probs.unsqueeze(0))
        # loc_score = torch.cat(tuple(output_probs), dim=0).to(output.device)
        # loc_score.requires_grad_()


        # NULL_TOKEN = 0  # a placeholder used for masked target locations

        pred = output.logits
        target = targ
        if shift and pred.dim() == 3:
            pred = pred[:, :-1]  # Remove last prediction in sequence
            target = target[:, 1:]  

        total_tokens = 0
        def get_score_loc(p,t):
            cnt_tokens = 0
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    cnt_tokens += 1
            ele_probs = p.log_softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(output.logits.device)).diag()
            return ele_probs.mean(), cnt_tokens

        loc_score, cnt = get_score_loc(pred[0], target[0])
        loc_score = loc_score.unsqueeze(0)
        total_tokens += cnt
        for batch_idx, (p, t) in enumerate(zip(pred, target)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_score_loc(p, t)
            total_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)   # (batch_size, )

        return {
            "avg_log_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": total_tokens,
            "loc_score": loc_score
        }

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass



    # def causal_modeling(self, batch):
    #     unconditional_start_token = self.tokenizer.bos_token
    #     start_token = (
    #         torch.tensor(self.tokenizer.encode(unconditional_start_token, add_special_tokens=False))
    #         .to(self.model.device)
    #         .unsqueeze(0)
    #     )

    #     initial_token_probabilities = self.model(start_token)
    #     initial_token_probabilities = torch.log_softmax(
    #         initial_token_probabilities[0], dim=-1
    #     )

    #     anti_batch_inputs = []
    #     for line in batch['raw']:
    #         anti_batch_inputs.append(line['anti-stereotype'])

    #     stereo_batch_inputs = []
    #     for line in batch['raw']:
    #         stereo_batch_inputs.append(line['stereotype'])

    #     anti_inputs = self.tokenizer(
    #             anti_batch_inputs, 
    #             return_tensors="pt",
    #             padding="max_length",
    #             max_length=64,
    #             truncation=True
    #     ).to(self.model.device)

    #     stereo_inputs = self.tokenizer(
    #             stereo_batch_inputs, 
    #             return_tensors="pt",
    #             padding="max_length",
    #             max_length=64,
    #             truncation=True
    #     ).to(self.model.device)

    #     def process(anti_tokens, anti_len, stereo_tokens, stereo_len):
    #         anti_tokens_tensor = anti_tokens.unsqueeze(0)
    #         stereo_tokens_tensor = stereo_tokens.to(self.model.device).unsqueeze(0)

    #         anti_joint_sentence_probability = initial_token_probabilities[0, 0, anti_tokens[0]].unsqueeze(0)
        
    #         stereo_joint_sentence_probability = initial_token_probabilities[0, 0, stereo_tokens[0]].unsqueeze(0)

    #         anti_output = torch.log_softmax(self.model(anti_tokens_tensor)[0], dim=-1)
    #         stereo_output = torch.log_softmax(self.model(stereo_tokens_tensor)[0], dim=-1)


    #         for idx in range(1, anti_len):
    #             anti_joint_sentence_probability = torch.cat((anti_joint_sentence_probability, anti_output[0, idx - 1, anti_tokens[idx]].unsqueeze(0)), dim=0)
            
    #         for idx in range(1, stereo_len):
    #             stereo_joint_sentence_probability = torch.cat((stereo_joint_sentence_probability, stereo_output[0, idx - 1, stereo_tokens[idx]].unsqueeze(0)), dim=0)
            
    #         return torch.mean(anti_joint_sentence_probability), torch.mean(stereo_joint_sentence_probability)
    #     anti_scores, stereo_scores = process(anti_inputs['input_ids'][0], sum(anti_inputs['attention_mask'][0]), stereo_inputs['input_ids'][0], sum(stereo_inputs['attention_mask'][0]))
    #     anti_scores = anti_scores.unsqueeze(0)
    #     stereo_scores = stereo_scores.unsqueeze(0)

    #     for idx in range(1, len(anti_batch_inputs)):
    #         anti_score, stereo_score = process(anti_inputs['input_ids'][idx], sum(anti_inputs['attention_mask'][idx]), stereo_inputs['input_ids'][idx], sum(stereo_inputs['attention_mask'][idx]))
    #         anti_scores = torch.cat((anti_scores, anti_score.unsqueeze(0)), dim=0)
    #         stereo_scores = torch.cat((stereo_scores, stereo_score.unsqueeze(0)), dim=0)
    #     return anti_scores, stereo_scores