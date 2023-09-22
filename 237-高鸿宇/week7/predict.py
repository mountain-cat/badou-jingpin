import torch
from model import get_net
from getdata import sentence_to_sequence
from config import opt
from transformers import BertTokenizer

def predict(net, comments, opt):
    net.to(opt.device)
    tokenizer = BertTokenizer.from_pretrained('bert')
    with torch.no_grad():
        for comment in comments:
            seq = sentence_to_sequence(tokenizer, comment)
            x = torch.LongTensor([seq])
            x = x.to(opt.device)
            y_pred = net(x)[0]
            cls = torch.argmax(y_pred)
            cls = '好' if cls == 1 else '差'
            print('评论:', comment, f'此评论为{cls}评')


if __name__ == "__main__":
    comments = ["羊汤也太次了,哪有放油菜的,服了",
                '连双筷子都没有，明明点的葡萄汁结果送来的是酸梅汤，也真是无语了！',
                '菜味道很棒！送餐很及时！',
                '经过上次晚了2小时，这次超级快，20分钟就送到了……']
    
    net = get_net(opt.hidden_size, opt.n_classes, opt.weight_to_load)
    predict(net, comments, opt)