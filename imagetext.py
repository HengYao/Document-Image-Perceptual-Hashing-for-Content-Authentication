import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import BertTokenizer, BertModel, BertConfig
from ImageHashModel import ResNetVit
from options import HiDDenConfiguration



class FusionModel(nn.Module):
    def __init__(self, config: HiDDenConfiguration,hidden_dim=256, n_class=50,image_dim=256):
        super(FusionModel, self).__init__()
        modelConfig = BertConfig.from_pretrained('.../BERT/config.json')
        self.textExtractor = BertModel.from_pretrained(
            '.../BERT/pytorch_model.bin', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
        self.image_extractor = ResNetVit(config).to(torch.device('cuda'))
        self.text_fc = nn.Linear(embedding_dim, hidden_dim)
        self.fuse_fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, tokens,segments,input_masks, image):
        output = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        text_features = self.text_fc(text_embeddings)

        image_features = self.image_extractor(image)
        combined_features = torch.cat((text_features, image_features), dim=1)

        hash_value = self.fuse_fc(combined_features)

        return hash_value



