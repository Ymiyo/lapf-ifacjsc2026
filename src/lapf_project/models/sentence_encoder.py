# src/lapf_project/models/sentence_encoder.py

from functools import lru_cache
from typing import List, Union

import torch
from torch import Tensor
from transformers import BertJapaneseTokenizer, BertModel


class SentenceBertJapanese:
    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = None,
    ) -> None:
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]  # (batch, seq, hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 16) -> Tensor:
        all_embeddings = []
        for batch_idx in range(0, len(sentences), batch_size):
            batch = sentences[batch_idx : batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            all_embeddings.append(sentence_embeddings)

        return torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)


@lru_cache(maxsize=1)
def get_sentence_encoder(
    device: Union[str, torch.device] = None,
    model_name_or_path: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
) -> SentenceBertJapanese:
    return SentenceBertJapanese(model_name_or_path=model_name_or_path, device=device)
