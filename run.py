from typing import List
from dataset import Language, NmtDataset, bucketed_batch_indices, collate_fn
from model import Seq2Seq

import torch
import datetime
import torch.utils
import random
from tqdm import tqdm, trange

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

### You can edit this file by yourself, and it doesn't affect your final score.
### You may change batch_size, embedding_dim, the number of epoch, and all other variable.

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

attention_type = 'concat' # 'dot' or 'concat'
embedding_dim = 128
hidden_dim = 64
bucketing = True

def plot_attention(attention: torch.Tensor, trg_text: List[str], src_text: List[str], name: str):
    assert attention.shape[0] == len(trg_text) and attention.shape[1] == len(src_text)
    _, ax = plt.subplots()
    _ = ax.pcolor(attention)

    ax.set_xticks([tick + .5 for tick in range(len(src_text))], minor=False)
    ax.set_yticks([tick + .5 for tick in range(len(trg_text))], minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(src_text, rotation=90, minor=False)
    ax.set_yticklabels(trg_text, minor=False)
    plt.savefig('attention_' + name + '.png')

def load_model():
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()
    dataset = NmtDataset(src=french, trg=english)
    model = Seq2Seq(french, english, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load('/home/admin/projects/ai605/assn2/seq2seq_concat.pth'))
    model.eval()

def train():
    max_epoch = 200
    batch_size = 256

    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()
    dataset = NmtDataset(src=french, trg=english)

    max_pad_len = 5
    sentence_length = list(map(lambda pair: (len(pair[0]), len(pair[1])), dataset))
    batch_sampler = bucketed_batch_indices(sentence_length, batch_size=batch_size, max_pad_len=max_pad_len) if bucketing else None

    model = Seq2Seq(french, english, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, num_workers=2, batch_size=1 if bucketing else batch_size, batch_sampler=batch_sampler, shuffle=not bucketing)
    loss_log = tqdm(total=0, bar_format='{desc}', position=2)
    for epoch in trange(max_epoch, desc="Epoch", position=0):
        for src_sentence, trg_sentence in tqdm(dataloader, desc="Iteration", position=1):
            optimizer.zero_grad()
            src_sentence, trg_sentence = src_sentence.to(device), trg_sentence.to(device)
            loss = model(src_sentence, trg_sentence, teacher_force=0.5)
            loss.backward()
            optimizer.step()

            des = 'Loss per a non-<PAD> Word: {:06.4f}'.format(loss.cpu())
            loss_log.set_description_str(des)
    
    torch.save(model.state_dict(), "seq2seq_" + attention_type + ".pth")

def translate():
    SOS = Language.SOS_TOKEN_IDX
    EOS = Language.EOS_TOKEN_IDX

    french_train = Language(path='data/train.fr.txt')
    english_train = Language(path='data/train.en.txt')
    french_train.build_vocab()
    english_train.build_vocab()
    model = Seq2Seq(french_train, english_train, attention_type=attention_type,
                    embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load("seq2seq_" + attention_type + ".pth", map_location=device))

    french_test = Language(path='data/test.fr.txt')
    english_test = Language(path='data/test.en.txt')
    french_test.set_vocab(french_train.word2idx, french_train.idx2word)
    english_test.set_vocab(english_train.word2idx, english_train.idx2word)
    dataset = NmtDataset(src=french_test, trg=english_test)
    
    samples = [dataset[16][0], dataset[1][0], dataset[2][0]] # You may choose your own samples to plot

    for i, french in enumerate(samples):
        translated, attention = model.translate(torch.Tensor(french).to(dtype=torch.long, device=device))
        source_text = [french_train.idx2word[idx] for idx in french]
        translated_text = [english_train.idx2word[idx] for idx in translated]
        plot_attention(attention.cpu().detach(), translated_text, source_text, name=attention_type + '_' + str(i))

    f = open('translated.txt', mode='w', encoding='utf-8')
    f_bleu = open('pred.en.txt', mode='w', encoding='utf-8')
    for french, english in tqdm(dataset, desc='Translated'):
        translated, attention = model.translate(torch.Tensor(french).to(dtype=torch.long, device=device))
        source_text = [french_train.idx2word[idx] for idx in french]
        target_text = [english_train.idx2word[idx] for idx in english if idx != SOS and idx != EOS]
        translated_text = [english_train.idx2word[idx] for idx in translated if idx != EOS]

        f.write('French    : ' + ' '.join(source_text) + '\n')
        f.write('English   : ' + ' '.join(target_text) + '\n')
        f.write('Translated: ' + ' '.join(translated_text) + '\n\n')
        f_bleu.write(' '.join(translated_text) + '\n')
    f.close()
    f_bleu.close()

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(4321)
    torch.manual_seed(4321)
    print(datetime.datetime.now())
    # train()
    print(datetime.datetime.now())
    translate()
