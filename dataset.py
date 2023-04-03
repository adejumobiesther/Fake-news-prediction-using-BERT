import config
import torch


class BERTDataset:
    def __init__(self, text_without_stopwords, target):
        self.text_without_stopwords = text_without_stopwords
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text_without_stopwords)

    def __getitem__(self, item):
        text_without_stopwords = str(self.text_without_stopwords[item])
        text_without_stopwords = " ".join(text_without_stopwords.split())

        inputs = self.tokenizer.encode_plus(
            text_without_stopwords,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
