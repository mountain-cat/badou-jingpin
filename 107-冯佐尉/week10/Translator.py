import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)

    # 输入
    #     sec_seq, input sequence, 
    #     src_mask, 和input sequence等长的张量，其中每个位置时true和false，来表示pad
    # 输出
    #     enc_output, 复制了5个，(1,120,128)->(5,120,128)
    #     gen_seq, (5,30) 第一列都为2，第2列是dec_output中概率值最大的前beam_size个的索引
    #     scores,概率值最大的前beam_size个的分数,由概率值取ln得到
    def _get_init_state(self, src_seq, src_mask):  # src_seq, src_mask (1, 120) (1,1,120)
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask)  # ->(1,120,128) (1,input_max_len,hidden_size)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)  # ->(1, 1, 6219)
        # init_seq??？？是什么 是应该是sos token
        
        # 从大到小取前beam_size个概率值，保留概率值和相应的索引
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)  # 概率值 词表中的索引 ->(1,5) (1,5)
        scores = torch.log(best_k_probs).view(beam_size)  # 概率值取对数ln，然后resize到一维 beam_size
        
        gen_seq = self.blank_seqs.clone().detach()        # ？？?? # ->(5,30) 第1列有值，都为2
        gen_seq[:, 1] = best_k_idx[0]                     # 第二列为dec_output中，概率值最大的前beam_size个的索引
        enc_output = enc_output.repeat(beam_size, 1, 1)   # (1,120,128)->(5,120,128)
        return enc_output, gen_seq, scores  # (5,120,128) (5,30) (5)


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):  # dec_output的维度(beam_size, output_len, vac_len)
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size) # 取前beam_size个
        #(beam_size, step, vac_len) (5,step, 6219) -> (5, 6219) ==> (5, beam_size), (5, beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
        #  (beam_size, 5)+(beam_size,1)->(beam_size, 5)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)  
        # (beam_size, 5)->(1,beam_size) (1,beam_size) 转化为1维张量后，取前beam_size个，记录score和k2的索引
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size  # (1,beam_size)
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]  # ->(1,beam_size) 从(beam_size, 5)的索引，取出词表里的index

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]  # 更新一下当前step的最优节点的父节点
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx  #填入该step中，预测的前beam_size个最优值的索引

        return gen_seq, scores

    # vac_size=6219, max_seq_len=120
    def translate_sentence(self, src_seq):    # (1,120)->
        print(src_seq.shape)
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)  # (1,1,120) 

            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)   # ->(5,120,128) (5,30) (5)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length max_seq_len:30
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)  # gen_seq[:, :step]-(beam_size,step)
                # ->(beam_size,step,vacb_size) dec_output:(5,step,6219)  
                # _model_decode传入的第一个参数是上一步decoder的输出，是以batch的方式传入

                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)  # ->(5,30) (5)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx  
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # self.len_map:(1,30)   eos_los:(5,30)  ->seq_lens:(5,1)
                # 取出所有<eos> token的位置，并标上位置序号，然后每行取出最小的序号

                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:  # (5,30)->(5)->n
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)  # beam_size保留最大的分数的索引
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()  # 取出分数最大的那一行，从前往后取到<eos> token的位置
