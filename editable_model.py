import torch
import torch.nn as nn
import torch.nn.functional as F
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
        output_soft = F.softmax(output.logits, dim=-1)
        output_log_soft = F.log_softmax(output.logits, dim=-1)
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
        loc_score = loc_score.unsqueeze(0)
        n_tokens += cnt
        for batch_idx, (p, t) in enumerate(zip(output_soft, targ)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_loc(p, t)
            n_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)
        
        loc_log_score, _ = get_loc(output_log_soft[0], targ[0])
        loc_log_score = loc_log_score.unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(output_log_soft, targ)):
            if batch_idx==0:
                continue
            probs, _ = get_loc(p, t)
            loc_log_score = torch.cat((loc_log_score, probs.unsqueeze(0)), dim=0)

        return {
            # "avg_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": n_tokens, 
            "loc_score": loc_score,
            "loc_log_score": loc_log_score
        }

    
    def _edit_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo):   # pred_anti: (batch_size, seq_len, vocab_size), targ_anti: (batch_size, seq_len)        
        
        def get_edit(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = [] # ids of [MASK] target tokens
            for i in range(len(t)): # seq_len
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    # anti_n_tokens += 1
            ele_probs = p.softmax(-1)[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
            return ele_probs.mean()
        
        def get_log_edit(p,t):
            mask_idxs = (t != -100)
            template_word_idxs = [] # ids of [MASK] target tokens
            for i in range(len(t)): # seq_len
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    # anti_n_tokens += 1
            ele_probs = p.log_softmax(-1)[mask_idxs] # [MASK] probs, (mask_num, vocab_size)
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()   # [MASK] target tokens probs (mask_num,)
            return ele_probs.mean()
        
        anti_score = get_edit(pred_anti[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        stereo_score = get_edit(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        anti_log_score = get_log_edit(pred_anti[0], targ_anti[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_log_score = torch.cat((anti_log_score, get_log_edit(p,t).unsqueeze(0)), dim=0)                                              # mean probs of [MASK] target tokens
        
        stereo_log_score = get_log_edit(pred_stereo[0], targ_stereo[0]).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_log_score = torch.cat((stereo_log_score, get_log_edit(p, t).unsqueeze(0)), dim=0)              
        
        # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # loss = kl_loss(input=anti_log_score, target=stereo_log_score) + kl_loss(input=stereo_log_score, target=anti_log_score)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(input=anti_log_score, target=stereo_score) + kl_loss(input=stereo_log_score, target=anti_score)
        
        # SS
        stereo_preferred, anti_preferred, neither_preferred = 0, 0, 0
        for anti, pro in zip(anti_score, stereo_score):
            if pro > anti:
                stereo_preferred += 1
            elif anti > pro:
                anti_preferred += 1
            else:
                neither_preferred += 1
        
        return {
            "ss_score": 50.00 if stereo_preferred + anti_preferred==0.00 else stereo_preferred / (stereo_preferred + anti_preferred),
            "loss": loss,
            # "anti_prob": anti_score.mean(),
            # "stereo_prob": stereo_score.mean(),
            "anti_score": anti_score, 
            "stereo_score": stereo_score,
            "anti_log_score": anti_log_score,
            "stereo_log_score": stereo_log_score
        }
    
    def _edit_causal_loss_fn(self, pred_anti, targ_anti, pred_stereo, targ_stereo, shift=True):

        if shift and pred_anti.dim() == 3 and pred_stereo.dim() == 3:  # Dealing with sequences
            pred_anti = pred_anti[:, :-1]  # Remove last prediction in sequence
            targ_anti = targ_anti[:, 1:]  # Shift to align predictions and targets
            pred_stereo = pred_stereo[:, :-1]
            targ_stereo = targ_stereo[:, 1:]

        def get_score(p,t, log=True):
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
            if log:
                ele_probs = p.log_softmax(-1)[mask_idxs]
            else:
                ele_probs = p.softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(pred_anti.device)).diag()
            return ele_probs.mean()
        
        anti_score = get_score(pred_anti[0], targ_anti[0], False).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_score = torch.cat((anti_score, get_score(p, t, False).unsqueeze(0)), dim=0)   # (batch_size, )
          

        stereo_score = get_score(pred_stereo[0], targ_stereo[0], False).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_score = torch.cat((stereo_score, get_score(p, t, False).unsqueeze(0)), dim=0)   # (batch_size, )
        
        anti_log_score = get_score(pred_anti[0], targ_anti[0], True).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_anti, targ_anti)):
            if batch_idx==0:
                continue
            anti_log_score = torch.cat((anti_log_score, get_score(p, t, True).unsqueeze(0)), dim=0)
        
        stereo_log_score = get_score(pred_stereo[0], targ_stereo[0], True).unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred_stereo, targ_stereo)):
            if batch_idx==0:
                continue
            stereo_log_score = torch.cat((stereo_log_score, get_score(p, t, True).unsqueeze(0)), dim=0)

        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(input=anti_log_score, target=stereo_score) + kl_loss(input=stereo_log_score, target=anti_score)
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
            # "anti_prob": anti_score.mean(),
            # "stereo_prob": stereo_score.mean(),
            "anti_score": anti_score,
            "stereo_score": stereo_score,
            "anti_log_score": anti_log_score,
            "stereo_log_score": stereo_log_score
        }
    
    

    def _loc_causal_loss_fn(self, output, targ, shift=True):
        pred = output.logits
        target = targ
        if shift and pred.dim() == 3:
            pred = pred[:, :-1]  # Remove last prediction in sequence
            target = target[:, 1:]  

        total_tokens = 0
        def get_score_loc(p,t, log=True):
            cnt_tokens = 0
            mask_idxs = (t != -100)
            template_word_idxs = []
            for i in range(len(t)):
                if t[i] != -100:
                    template_word_idxs.append(t[i])
                    cnt_tokens += 1
            if log:
                ele_probs = p.log_softmax(-1)[mask_idxs]
            else:
                ele_probs = p.softmax(-1)[mask_idxs]
            ele_probs = ele_probs.index_select(dim=1, index=torch.tensor(template_word_idxs).to(output.logits.device)).diag()
            return ele_probs.mean(), cnt_tokens

        loc_score, cnt = get_score_loc(pred[0], target[0], False)
        loc_score = loc_score.unsqueeze(0)
        total_tokens += cnt
        for batch_idx, (p, t) in enumerate(zip(pred, target)):
            if batch_idx==0:
                continue
            probs, cnt_token = get_score_loc(p, t, False)
            total_tokens += cnt_token
            loc_score = torch.cat((loc_score, probs.unsqueeze(0)), dim=0)   # (batch_size, )
        
        loc_log_score, _ = get_score_loc(pred[0], target[0], True)
        loc_log_score = loc_log_score.unsqueeze(0)
        for batch_idx, (p, t) in enumerate(zip(pred, target)):
            if batch_idx==0:
                continue
            probs, _ = get_score_loc(p, t, True)
            loc_log_score = torch.cat((loc_log_score, probs.unsqueeze(0)), dim=0) # (batch_size, )

        return {
            # "avg_prob": loc_score.mean(),
            "loss": output.loss,
            "n_tokens": total_tokens,
            "loc_score": loc_score,
            "loc_log_score": loc_log_score
        }

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass