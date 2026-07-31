from pathlib import Path

import torch
from torch import nn
from transformers import BertModel

from ImageHashModel import ResNetVit
from options import HiDDenConfiguration


PROJECT_ROOT = Path(__file__).resolve().parent
BERT_MODEL_DIR = PROJECT_ROOT / "weights" / "chinese_bert"


class FusionModel(nn.Module):
    def __init__(
        self,
        config: HiDDenConfiguration,
        hidden_dim=256,
        n_class=50,
        image_dim=256,
    ):
        super().__init__()
        self.text_extractor = BertModel.from_pretrained(str(BERT_MODEL_DIR))
        embedding_dim = self.text_extractor.config.hidden_size
        self.image_extractor = ResNetVit(config)
        self.text_fc = nn.Linear(embedding_dim, hidden_dim)
        self.image_fc = (
            nn.Identity()
            if image_dim == hidden_dim
            else nn.Linear(image_dim, hidden_dim)
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.fuse_fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, tokens, segments, input_masks, image):
        output = self.text_extractor(
            input_ids=tokens,
            token_type_ids=segments,
            attention_mask=input_masks,
        )
        text_features = self.text_norm(
            self.text_fc(output.last_hidden_state[:, 0, :])
        )
        image_features = self.image_norm(
            self.image_fc(self.image_extractor(image))
        )
        combined_features = torch.cat((text_features, image_features), dim=1)
        return self.fuse_fc(combined_features)
