# -*- coding: utf-8 -*-
import re
from collections import defaultdict

import torch
from config import opt
from model import TorchModel
from transformers import BertTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class SentenceLabel:
    def __init__(self, opt, model_path):
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained('badouai/bert')
        self.model = TorchModel(opt)

        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False,
                                r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "key", "value"])
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.cuda()
        state_dict = self.model.state_dict()
        state_dict.update(torch.load(model_path))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("模型加载完毕!")

    def predict(self, sentence):
        input_id = self.tokenizer.encode(sentence)[1:-1]
        with torch.no_grad():
            x = torch.LongTensor([input_id])
            x = x.cuda()
            res = self.model(x).logits[0]
            res = torch.argmax(res, dim=-1)
        res = res.cpu().detach().tolist()
        entities = self.decode(sentence, res)
        return entities

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

if __name__ == "__main__":
    sl = SentenceLabel(opt, "badouai/week12/epoch_20.pth")

    sentence = '明天下午，约翰将在纽约的联合国总部与联合国秘书长安东尼奥·古特雷斯会晤。他代表国际红十字会，讨论全球人道主义援助计划和应对人道危机的合作。这次会议对于加强国际援助合作至关重要，以帮助那些受到冲突和灾难影响的人们。'
    res = sl.predict(sentence)
    print(sentence)
    for key, value in res.items():
        print(key, value)