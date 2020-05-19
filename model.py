import random

import torch 
import torch.nn as nn

from attention import AttentionBase, DotAttention, ConcatAttention

### You may import any Python standard libray or pyTorch sub-directory here.
import numpy as np
### END YOUR LIBRARIES

from dataset import Language, NmtDataset, collate_fn

class Seq2Seq(torch.nn.Module):
    def __init__(self, src: Language, trg: Language, attention_type: str, embedding_dim: int=128, hidden_dim: int=64):
        """ Seq2Seq with Attention model

        Parameters:
        src -- source language vocabs
        trg -- target language vocabs
        attention_type -- internal attention type: 'dot' or 'concat'
        embeding_dim -- embedding dimension
        hidden_dim -- hidden dimension
        """
        super().__init__()
        PAD = Language.PAD_TOKEN_IDX
        SRC_TOKEN_NUM = len(src.idx2word) # Including <PAD>
        TRG_TOKEN_NUM = len(trg.idx2word) # Including <PAD>

        ### Declare Embedding Layers
        # Doc for Embedding Layer: https://pytorch.org/docs/stable/nn.html#embedding 
        #
        # Note: You should set padding_idx options to embed <PAD> tokens to 0 values 
        self.src_embedding: nn.Embedding = None
        self.trg_embedding: nn.Embedding = None

        self.src_embedding = nn.Embedding(SRC_TOKEN_NUM, embedding_dim=embedding_dim, padding_idx=PAD)
        self.trg_embedding = nn.Embedding(TRG_TOKEN_NUM, embedding_dim=embedding_dim, padding_idx=PAD)

        ### Declare LSTM/LSTMCell Layers
        # Doc for LSTM Layer: https://pytorch.org/docs/stable/nn.html#lstm
        # Doc for LSTMCell Layer: https://pytorch.org/docs/stable/nn.html#lstmcell
        # Explanation for bidirection RNN in torch by @ceshine_en (English): https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
        #
        # Note 1: Don't forget setting batch_first option because our tensor follows [batch_size, sequence_length, ...] form.
        # Note 2: Use one layer LSTM with bias for encoder & decoder
        self.encoder: nn.LSTM = None
        self.decoder: nn.LSTMCell = None

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTMCell(embedding_dim, hidden_dim*2, bias=True)
        # Attention Layer
        if attention_type == 'dot':
            self.attention: AttentionBase = DotAttention()
        elif attention_type == 'concat':
            self.attention: AttentionBase = ConcatAttention(hidden_dim)

        ### Declare output Layers
        # Output layers to attain combined-output vector o_t in the handout
        # Doc for Sequential Layer: https://pytorch.org/docs/stable/nn.html#sequential
        # Doc for Linear Layer: https://pytorch.org/docs/stable/nn.html#linear 
        # Doc for Tanh Layer: https://pytorch.org/docs/stable/nn.html#tanh 
        #
        # Note: Shape of combined-output o_t should be (batch_size, TRG_TOKEN_NUM - 1) because we will exclude the probability of <PAD>
        self.output: nn.Sequential = None

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, TRG_TOKEN_NUM - 1, bias=False)
        )
        ### END YOUR CODE

    def forward(self, src_sentences: torch.Tensor, trg_sentences: torch.Tensor, teacher_force: float=.5):
        """ Seq2Seq forward function
        Note: As same with assignment 1, <PAD> should not affect the loss calculation.

        Parameters:
        src_sentences -- batched source sentences
                            in shape (batch_size, sentence_length)
        trg_sentences -- batched target sentences
                            in shape (batch_size, sentence_length)
        teacher_force -- the probability of teacher forcing

        Return:
        loss -- average loss per a non-<PAD> word
        """
        # You may use below notations
        batch_size = src_sentences.shape[0]
        PAD = Language.PAD_TOKEN_IDX
        SOS = Language.SOS_TOKEN_IDX
        EOS = Language.EOS_TOKEN_IDX

        encoder_masks = src_sentences == PAD

        ### Encoder part (~7 lines)
        # We strongly recommand you to use torch.nn.utils.rnn.pack_padded_sequence/pad_packed_sequence to deal with <PAD> and boost the performance.
        # Doc for pack_padded_sequence: https://pytorch.org/docs/stable/nn.html#pack-padded-sequence
        # Doc for pad_packed_sequence: https://pytorch.org/docs/stable/nn.html#pad-packed-sequence
        # Explanation for PackedSequence by @simonjisu (Korean): https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
        # Because you have already sorted sentences at collate_fn, you can use pack_padded_sequence without any modification.
        #
        # Variable:
        # encoder_hidden -- encoder_hidden is encoder hidden state which is same with h^enc in the handout
        #                   in shape (batch_size, sequence_length, hidden_dim)
        #                   All values in last dimension (hidden_dim dimension) are zeros for <PAD> location.
        # hidden_state -- Last encoder hidden state
        #                   in shape (batch_size, hidden_dim * 2)
        # cell_state -- Last encoder cell state
        #                   in shape (batch_size, hidden_dim * 2)
        encoder_hidden: torch.Tensor = None
        hidden_state: torch.Tensor = None
        cell_state: torch.Tensor = None

        src_embedding_seq = self.src_embedding(src_sentences)
        src_lengths = (src_sentences != 0).sum(dim=1)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(src_embedding_seq, src_lengths, batch_first=True, enforce_sorted=True)

        encoder_hidden_states, (hidden_state, cell_state) = self.encoder(packed_seq)
        encoder_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_hidden_states,
                                                                                 batch_first=True, padding_value=PAD)
        encoder_hidden[encoder_masks] = 0.

        ### END YOUR CODE

        # Loss initialize
        decoder_out = trg_sentences.new_full([batch_size], fill_value=SOS)
        decoder_c0 = torch.cat((cell_state[0], cell_state[1]), dim=1)
        decoder_h0 = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        sum_of_loss = 0.
        ce_loss = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
        for trg_word_idx in range(trg_sentences.shape[1] - 1):
            # Teacher forcing: feed correct labels with a probability of teacher_force
            decoder_input = trg_sentences[:, trg_word_idx] if torch.distributions.bernoulli.Bernoulli(teacher_force).sample() else decoder_out
            decoder_input_embedding = self.trg_embedding(decoder_input)  # y_t
            decoder_h0, decoder_c0 = self.decoder(decoder_input_embedding, (decoder_h0, decoder_c0))
            decoder_mask = trg_sentences[:, trg_word_idx + 1] == PAD
            decoder_h0[decoder_mask] = 0.

            attention_output, distribution = self.attention(encoder_hidden, encoder_masks, decoder_h0, decoder_mask)

            output_layer_input = torch.cat((decoder_h0, attention_output), dim=1)
            output_logit = self.output(output_layer_input)

            # You may use below notations
            decoder_target = trg_sentences[:, trg_word_idx+1]
            decoder_out = torch.argmax(output_logit, dim=1) + 1
            tmp = output_logit.new_full((batch_size, 1), fill_value=float('-inf'))
            new_logit = torch.cat((tmp, output_logit), dim=1)
            loss = ce_loss(new_logit, decoder_target)
            sum_of_loss += loss

        loss = sum_of_loss
        assert loss.shape == torch.Size([])
        return loss / (trg_sentences[:, 1:] != PAD).sum() # Return average loss per a non-<PAD> word

    def translate(self, sentence: torch.Tensor, max_len: int=30):
        """
        Parameters:
        sentence -- sentence to be translated
                        in shape (sentence_length, )
        max_len -- maximum word length of the translated stentence

        Return:
        translated -- translated sentence
                        in shape (translated_length, ) of which translated_length <= max_len
                        with torch.long type
        distrubutions -- stacked attention distribution
                        in shape (translated_length, sentence_length)
                        This is used for ploting
        """
        PAD = Language.PAD_TOKEN_IDX
        SOS = Language.SOS_TOKEN_IDX
        EOS = Language.EOS_TOKEN_IDX
        sentence_length = sentence.size(0)

        ### YOUR CODE HERE (~ 22 lines)
        # You can complete this part without any help by refering to the above forward function
        # Note: use argmax to get the next input word
        translated =[]
        distributions = []

        src_embedding_seq = self.src_embedding(sentence).unsqueeze(0)
        src_lengths = sentence_length

        encoder_hidden_states, (hidden_state, cell_state) = self.encoder(src_embedding_seq)
        encoder_hidden=encoder_hidden_states
        ### END YOUR CODE

        # Loss initialize
        decoder_out = sentence.new_full([1], fill_value=SOS)
        decoder_c0 = torch.cat((cell_state[0], cell_state[1]), dim=1)
        decoder_h0 = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        encoder_masks = (sentence == PAD).unsqueeze(0)
        decoder_mask = False
        for trg_word_idx in range(max_len):
            decoder_input = decoder_out
            decoder_input_embedding = self.trg_embedding(decoder_input)  # y_t
            decoder_h0, decoder_c0 = self.decoder(decoder_input_embedding, (decoder_h0, decoder_c0))

            decoder_h0[decoder_mask] = 0.
            attention_output, distribution = self.attention(encoder_hidden, encoder_masks, decoder_h0, decoder_mask)

            output_layer_input = torch.cat((decoder_h0, attention_output), dim=1)
            output_logit = self.output(output_layer_input)

            # You may use below notations
            decoder_out = torch.argmax(output_logit, dim=1) + 1
            translated.append(decoder_out)
            distributions.append(distribution.squeeze(0))
            if decoder_out == EOS:
                break
        translated = torch.stack(translated).squeeze(1)
        distributions = torch.stack(distributions)
        ### END YOUR CODE

        assert translated.dim() == 1 and distributions.shape == torch.Size([translated.size(0), sentence_length])
        return translated, distributions

#############################################
# Testing functions below.                  #
#############################################

def test_initializer_and_forward():
    print("=====Model Initializer & Forward Test Case======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()
    dataset = NmtDataset(src=french, trg=english)

    model = Seq2Seq(french, english, attention_type='dot')

    # the first test
    try:
        model.load_state_dict(torch.load("sanity_check.pth", map_location='cpu'))
    except Exception as e:
        print("Your model initializer is wrong. Check the handout and comments in details and implement the model precisely.")
        raise e
    print("The first test passed!")

    batch_size = 8
    max_pad_len = 5
    sentence_length = list(map(lambda pair: (len(pair[0]), len(pair[1])), dataset))

    batch_indices = [[0, 1, 2, 3, 4, 5, 6, 7]]
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, collate_fn=collate_fn, num_workers=0, batch_sampler=batch_indices)
    batch = next(iter(dataloader))
    loss = model(batch[0], batch[1])

    # the second test
    assert loss.detach().allclose(torch.tensor(3.03703070)), \
        "Loss of the model does not match expected result."
    print("The second test passed!")

    loss.backward()

    # the third test
    expected_grad = torch.Tensor([[-8.29117271e-05, -4.44278521e-05, -2.64967621e-05],
                                  [-3.89243884e-04, -1.29778590e-03, -4.56827343e-04],
                                  [-2.76966626e-03, -1.00148167e-03, -6.68873254e-05]])
    assert model.encoder.weight_ih_l0.grad[:3,:3].allclose(expected_grad), \
        "Gradient of the model does not match expected result."
    print("The third test passed!")

    print("All 3 tests passed!")

def test_translation():
    print("=====Translation Test Case======")
    french = Language(path='data/train.fr.txt')
    english = Language(path='data/train.en.txt')
    french.build_vocab()
    english.build_vocab()

    model = Seq2Seq(french, english, attention_type='dot')
    model.load_state_dict(torch.load("sanity_check.pth", map_location='cpu'))

    sentence = torch.Tensor([ 4,  6, 40, 41, 42, 43, 44, 13]).to(torch.long)
    translated, distributions = model.translate(sentence)

    # the first test
    assert translated.tolist() == [4, 16, 9, 56, 114, 51, 1, 14, 3], \
        "Your translation does not math expected result."
    print("The first test passed!")

    # the second test
    expected_dist = torch.Tensor([[9.98170257e-01, 1.74237683e-03, 7.48323873e-05],
                                  [1.94309454e-03, 9.82858062e-01, 4.87918453e-03],
                                  [2.26807110e-02, 7.29433298e-02, 3.17393959e-01]])
    assert distributions[:3,:3].allclose(expected_dist), \
        "Your attetion distribution does not math expected result."
    print("The second test passed!")

    # the third test
    sentence = torch.Tensor([ 4,  6, 40, 41, 42, 43, 44, 13]).to(torch.long)
    translated, _ = model.translate(sentence, max_len=4)
    assert translated.tolist() == [4, 16, 9, 56], \
        "max_len parameter dose not work properly."
    print("The third test passed!")

    print("All 3 tests passed!")

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    test_initializer_and_forward()
    test_translation()
