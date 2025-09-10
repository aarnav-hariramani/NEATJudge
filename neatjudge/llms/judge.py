from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

@dataclass
class TransformersJudge:
    model_name: str
    device: str = "auto"
    max_new_tokens: int = 1
    temperature: float = 0.0

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.device == "auto":
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device:
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def option_logprob(self, prompt: str, option_text: str) -> float:
        # Compute log P(option | prompt) using next-token logprobs across the option tokens.
        # Concatenate prompt + option; score only the option tokens.
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        enc_opt = self.tokenizer(option_text, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([enc_prompt["input_ids"], enc_opt["input_ids"]], dim=1)
        attn = torch.ones_like(input_ids)
        if self.model.device.type != "cpu":
            input_ids = input_ids.to(self.model.device)
            attn = attn.to(self.model.device)
        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]          # predict next token
        # slice for option tokens only
        opt_len = enc_opt["input_ids"].shape[1]
        logits_opt = logits[:, -opt_len:, :]
        labels_opt = input_ids[:, -opt_len:]
        logprobs = torch.log_softmax(logits_opt, dim=-1)
        tok_lp = logprobs.gather(-1, labels_opt.unsqueeze(-1)).squeeze(-1)  # [1, opt_len]
        return float(tok_lp.sum().item())

    def score_choices(self, prompt: str, options: List[str]) -> Tuple[int, List[float]]:
        lps = [self.option_logprob(prompt, o) for o in options]
        idx = int(np.argmax(lps))
        # convert to normalized probabilities for MC2
        m = max(lps)
        probs = np.exp(np.array(lps) - m)       # stabilize
        probs = probs / probs.sum()
        return idx, probs.tolist()
