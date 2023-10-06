import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from mul_attention import multihead_attention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GATLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x_ = self.ra1(x, p)
        x = x_+ x
        p_ = self.ra2(p, x, mask)
        p =  p_ + p
        return x, p


class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            # mask[:, None, :] is equal to mask.unsqueeze(1)
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)


class Gate(nn.Module):
    def __init__(self, hid_dim):
        super(Gate, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.linear1 = nn.Linear(hid_dim, hid_dim // 16)
        self.linear2 = nn.Linear(hid_dim // 16, 97)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, y, cls):
        # cls = self.dropout(cls)
        cls = self.linear1(cls)
        cls = self.activation(cls)
        gate = self.sigmoid(self.linear2(cls))
        gate = gate[:, None, None, :]
        return gate * x + (1 - gate) * y

class ConvAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, groups, dropout=0.1):
        super(ConvAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        input_channels = hid_dim * 2 + pre_channels
        self.groups = groups

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.linear1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=False)

        self.conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.score_layer = nn.Conv2d(channels, n_heads, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, pre_conv=None, mask=None, residual=True, self_loop=True):
        ori_x, ori_y = x, y

        B, M, _ = x.size()
        B, N, _ = y.size()

        fea_map = torch.cat([x.unsqueeze(2).repeat_interleave(N, 2), y.unsqueeze(1).repeat_interleave(M, 1)],
                            -1).permute(0, 3, 1, 2).contiguous()
        if pre_conv is not None:
            fea_map = torch.cat([fea_map, pre_conv], 1)
        fea_map = self.conv(fea_map)

        scores = self.activation(self.score_layer(fea_map))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask.eq(0), -9e10)

        x = self.linear1(self.dropout(x))
        y = self.linear2(self.dropout(y))

        out_x = torch.matmul(F.softmax(scores, -1), y.view(B, N, self.n_heads, -1).transpose(1, 2))
        out_x = out_x.transpose(1, 2).contiguous().view(B, M, -1)
        out_y = torch.matmul(F.softmax(scores.transpose(2, 3), -1), x.view(B, M, self.n_heads, -1).transpose(1, 2))
        out_y = out_y.transpose(1, 2).contiguous().view(B, N, -1)

        if self_loop:
            out_x = out_x + x
            out_y = out_y + y

        out_x = self.activation(out_x)
        out_y = self.activation(out_y)

        if residual:
            out_x = out_x + ori_x
            out_y = out_y + ori_y
        return out_x, out_y, fea_map


class ConvAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, layers, groups, dropout):
        super(ConvAttention, self).__init__()

        self.layers = nn.ModuleList([ConvAttentionLayer(hid_dim, n_heads, pre_channels if i == 0 else channels,
                                                        channels, groups, dropout=dropout) for i in range(layers)])

    def forward(self, x, y, fea_map=None, mask=None, residual=True, self_loop=True):
        fea_list = []
        for layer in self.layers:
            x, y, fea_map = layer(x, y, fea_map, mask, residual, self_loop)
            fea_list.append(fea_map)

        return x, y, fea_map.permute(0, 2, 3, 1).contiguous()


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class BERTModel(nn.Module):
    def __init__(self, vocab_size, tok_emb_size, ner_emb_size, pos_emb_size, dis_emb_size, hid_size,
                 channels, layers, chunk, dropout1, dropout2):
        super(BERTModel, self).__init__()
        self.chunk = chunk


        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("./bert/bert-base-uncased")

        self.ner_embs = nn.Embedding(7, ner_emb_size)
        self.pos_embs = nn.Embedding(52, pos_emb_size)
        self.dis_embs = nn.Embedding(20, dis_emb_size)
        emb_size = tok_emb_size + ner_emb_size
        # hid_size = emb_size

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        self.biaffine = Biaffine(n_in=hid_size, n_out=97, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(hid_size // 2, 97, dropout=dropout2)
        self.linear = nn.Linear(97, 97)

        self.conv1 = nn.Sequential(
            nn.Dropout2d(dropout2),
            nn.Conv2d(2 * hid_size + dis_emb_size, hid_size, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Dropout2d(dropout2),
            nn.Conv2d(hid_size, hid_size // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hid_size // 2, hid_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.mulatt1 = multihead_attention(hid_size, 4)
        self.mulatt2 = multihead_attention(hid_size, 4)
        self.mlp_ens = MLP(n_in=2 * hid_size, n_out=hid_size, dropout=dropout2)
        self.mlp_eno = MLP(n_in=2 * hid_size, n_out=hid_size, dropout=dropout2)

        self.mlp_s = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.mlp_o = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.layers1 = nn.ModuleList([GATLayer(hid_size) for _ in range(1)])
        self.layers2 = nn.ModuleList([GATLayer(hid_size) for _ in range(1)])
        self.layers3 = nn.ModuleList([GATLayer(hid_size) for _ in range(1)])
        self.layers4 = nn.ModuleList([GATLayer(hid_size) for _ in range(1)])
        self.ws1 = nn.Linear(hid_size, hid_size // 2, bias=False)
        self.ws2 = nn.Linear(hid_size // 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.mlp_3 = MLP(n_in=hid_size, n_out=97, dropout=dropout2)
        self.mlp_4 = MLP(n_in=emb_size, n_out=hid_size, dropout=dropout2)



        # self.dropout1 = nn.Dropout(dropout1)
        # self.dropout2 = nn.Dropout(dropout2)
        #
        # self.men2men_conv_att = ConvAttention(hid_size, 1, dis_emb_size, channels,
        #                                       groups=1, layers=layers, dropout=dropout1)
        #
        # self.mlp_sub = MLP(n_in=hid_size, n_out=hid_size // 2, dropout=dropout2)
        # self.mlp_obj = MLP(n_in=hid_size, n_out=hid_size // 2, dropout=dropout2)
        # self.biaffine = Biaffine(n_in=hid_size // 2, n_out=97, bias_x=True, bias_y=True)
        # self.mlp_rel = MLP(channels, channels, dropout=dropout2)
        # self.linear = nn.Linear(channels, 97)
        # self.gate = Gate(hid_size)

    def forward(self, doc_inputs, psn_inputs, ner_inputs, dis_inputs,
                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask):
        length = doc_inputs.ne(0).sum(dim=-1).to("cpu")

        att_mask = doc_inputs.ne(0)
        tok_embs = self.bert(doc_inputs, attention_mask=att_mask, position_ids=psn_inputs, token_type_ids=torch.zeros_like(psn_inputs, device=psn_inputs.device, dtype=torch.long))[0]
        ner_embs = self.ner_embs(ner_inputs)
        tok_embs = tok_embs[:, 1:-1]

        outs = torch.cat([tok_embs, ner_embs], dim=-1)

        outs = self.mlp_4(outs)

        max_e = doc2ent_mask.size(1)
        max_m = doc2men_mask.size(1)

        min_value = torch.min(outs).item()

        _outs = outs.unsqueeze(1).expand(-1, max_m, -1, -1)
        _outs = torch.masked_fill(_outs, doc2men_mask.eq(0).unsqueeze(-1), min_value)
        men_reps, _ = torch.max(_outs, dim=2)


        # add

        # rel_outputs3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        outs2 = outs.unsqueeze(1).expand(-1, max_e, -1, -1)
        outs2 = torch.masked_fill(outs2, doc2ent_mask.eq(0).unsqueeze(-1), 0)    # [32,29,393,256]
        outs2_ = outs2.reshape(-1, outs2.size(2), outs2.size(3))
        hbar = self.tanh(self.ws1(outs2_))
        alpha = self.ws2(hbar)  # batch_size * seq_len * 1
        alpha = torch.transpose(alpha, 1, 2)  # batch_size * 1  * seq_len
        alpha = torch.softmax(alpha, dim=-1)
        outs2_ = torch.bmm(alpha, outs2_).squeeze(1).reshape(outs2.size(0), outs2.size(1), outs2.size(3))

        outs2_1 = outs2_.unsqueeze(1).expand(-1, max_e, -1, -1)
        outs2_2 = outs2_.unsqueeze(2).expand(-1, -1, max_e, -1)
        rel_outputs3 = outs2_1 + outs2_2
        rel_outputs3 = self.mlp_3(rel_outputs3)



        # rel_outputs1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ref_emb = self.ref_embs(ref_inputs)  # [32,47,47,20]
        men_men1 = men_reps.unsqueeze(1).expand(-1, max_m, -1, -1)
        men_men2 = men_reps.unsqueeze(2).expand(-1, -1, max_m, -1)
        dis_emb = self.dis_embs(dis_inputs)  # [32,47,47,20]
        men_men = torch.cat([men_men1, men_men2, dis_emb], dim=3)  # [32,47,47,532]
        men_men = self.conv1(men_men.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [32,47,47,256]

        # men_men1 = men_reps.unsqueeze(1).expand(-1, max_m, -1, -1)
        # men_men2 = men_reps.unsqueeze(2).expand(-1, -1, max_m, -1)
        # dis_emb = self.dis_embs(dis_inputs)  # [32,47,47,20]
        # men_men = torch.cat([men_men1, men_men2, dis_emb], dim=3)  # [32,47,47,532]
        # men_men = self.conv1(men_men.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [32,47,47,256]

        min_f_value = torch.min(men_men).item()
        fea_list = []
        chunk = self.chunk   # 8
        men_men_ = torch.split(men_men, chunk, dim=0)
        m2e_mask2 = torch.split(men2ent_mask, chunk, dim=0)
        for fea_map, m2e_mask in zip(men_men_, m2e_mask2):
            fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
            fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, :, None, None], min_f_value)
            fea_map, _ = torch.max(fea_map, dim=2)


            fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
            fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, None, :, None], min_f_value)
            fea_map, _ = torch.max(fea_map, dim=3)

            fea_list.append(fea_map)
        en_en = torch.cat(fea_list, dim=0)   # [32,29,29,256]

        en_en1 = self.conv2(en_en.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)   # [32,29,29,128]

        en_en2 = self.dropout2(self.mlp_rel(en_en1))  # [32,29,29,97]
        rel_outputs1 = self.linear(en_en2)   # [32,29,29,97]


        # rel_outputs2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # men_s, men_o, _ = self.men2men_conv_att(men_reps, men_reps, dis_emb.permute(0, 3, 1, 2), men2men_mask)  # men2men_mask: [32,47,47]

        men_s1, _ = torch.max(men_men, dim=1)
        men_o1, _ = torch.max(men_men, dim=2)

        men1 = torch.cat([men_reps, men_s1],dim=1)
        men11 = self.mulatt1(men1, men1, men1)
        men2 = torch.cat([men_o1, men_reps],dim=1)
        men22 = self.mulatt2(men2, men2, men2)
        # men_12 = men11 + men22
        men_12 = men1 + men2


        # 57.49
        # men_s = men_12[:, :max_m, :]
        # men_o = men_12[:, max_m:, :]

        # 57.67
        men_o = men_12[:, :max_m, :]
        men_s = men_12[:, max_m:, :]


        min_x_value = torch.min(men_s).item()
        men_s = men_s.unsqueeze(1).expand(-1, max_e, -1, -1)   # [32,29，47,256]
        men_s = torch.masked_fill(men_s, men2ent_mask.eq(0).unsqueeze(-1), min_x_value)
        en_s, _ = torch.max(men_s, dim=2)  # [32,29,256]

        min_y_value = torch.min(men_o).item()
        men_o = men_o.unsqueeze(1).expand(-1, max_e, -1, -1)   # [32,29，47,256]
        men_o = torch.masked_fill(men_o, men2ent_mask.eq(0).unsqueeze(-1), min_y_value)
        en_o, _ = torch.max(men_o, dim=2)  # [32,29,256]

        en_s1, _ = torch.max(en_en, dim=1)
        en_o1, _ = torch.max(en_en, dim=2)

        # for l1 in self.layers1:
        #     en_s, en_o = l1(en_s, en_o)
        #
        # for l2 in self.layers2:
        #     en_s1, en_o1 = l2(en_s1, en_o1)
        #
        # for l3 in self.layers3:
        #     en_s, en_s1 = l3(en_s, en_s1)
        #
        # for l4 in self.layers4:
        #     en_o, en_o1 = l4(en_o, en_o1)


        # 58.01
        en_s = torch.cat([en_s, en_s1], dim=2)
        en_o = torch.cat([en_o, en_o1], dim=2)
        en_s = self.mlp_ens(en_s)
        en_o = self.mlp_eno(en_o)


        en_s = self.dropout2(self.mlp_s(en_s))  # [32，29，256]
        en_o = self.dropout2(self.mlp_o(en_o))  # [32，29，256]

        rel_outputs2 = self.biaffine(en_s, en_o)   # [32,29,29,97]


        return rel_outputs1 + rel_outputs2
        # return rel_outputs1
        # return rel_outputs1 + rel_outputs2 + rel_outputs3













        # dis_emb = self.dis_embs(dis_inputs).permute(0, 3, 1, 2)
        # x, y, fea_maps = self.men2men_conv_att(men_reps, men_reps, dis_emb, men2men_mask)
        #
        # min_x_value = torch.min(x).item()
        # x = x.unsqueeze(1).expand(-1, max_e, -1, -1)
        # x = torch.masked_fill(x, men2ent_mask.eq(0).unsqueeze(-1), min_x_value)
        # x, _ = torch.max(x, dim=2)
        #
        # min_y_value = torch.min(y).item()
        # y = y.unsqueeze(1).expand(-1, max_e, -1, -1)
        # y = torch.masked_fill(y, men2ent_mask.eq(0).unsqueeze(-1), min_y_value)
        # y, _ = torch.max(y, dim=2)
        #
        # min_f_value = torch.min(fea_maps).item()
        #
        # fea_list = []
        # chunk = self.chunk
        # fea_maps = torch.split(fea_maps, chunk, dim=0)
        # m2e_mask2 = torch.split(men2ent_mask, chunk, dim=0)
        # for fea_map, m2e_mask in zip(fea_maps, m2e_mask2):
        #     fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
        #     fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, :, None, None], min_f_value)
        #     fea_map, _ = torch.max(fea_map, dim=2)
        #
        #     fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
        #     fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, None, :, None], min_f_value)
        #     fea_map, _ = torch.max(fea_map, dim=3)
        #     fea_list.append(fea_map)
        # fea_maps = torch.cat(fea_list, dim=0)
        #
        # ent_sub = self.dropout2(self.mlp_sub(x))
        # ent_obj = self.dropout2(self.mlp_obj(y))
        #
        # rel_outputs1 = self.biaffine(ent_sub, ent_obj)
        #
        # fea_maps = self.dropout2(self.mlp_rel(fea_maps))
        # rel_outputs2 = self.linear(fea_maps)
        #
        # return rel_outputs1 + rel_outputs2
