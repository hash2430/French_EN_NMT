from abc import ABC, abstractmethod
import random

### You may import any Python standard libray or pyTorch sub-directory here.

### END YOUR LIBRARIES

import torch 
import torch.nn as nn

class AttentionBase(nn.Module, ABC):
    """ Base attention class
    You don't need to modify anything in this class
    """
    @abstractmethod
    def forward(self, encoder_hidden: torch.Tensor, encoder_mask: torch.Tensor, decoder_hidden: torch.Tensor, decoder_mask: torch.Tensor):
        """ Abstract attention forward function
        Your forward function should follow below arguments & returns
        For ploting, you should return attention distribution result.

        Parameters:
        encoder_hidden -- encoder_hidden is encoder hidden state which is same with h^enc in the handout 
                            in shape (batch_size, encoder_sequence_length, hidden_dim)
                            All values in last dimension (hidden_dim dimension) are zeros for <PAD> location.
        encoder_mask -- encoder_mask is <PAD> mask for encoder
                            in shape (batch_size, sequence_length) with torch.bool type
                            True for <PAD> and False for non-<PAD>
                            Same with (encoder_hidden == 0.).all(-1)
        decoder_hidden -- decoder_hidden is decoder hidden state which is same with h^dec_t in the handout
                            in shape (batch_size, hidden_dim)
                            All values in last dimension (hidden_dim dimension) are zeros for <PAD> location.
        decoder_mask -- decoder_mask is <PAD> mask for decoder
                            in shape (batch_size, ) with torch.bool type
                            True for <PAD> and False for non-<PAD>
                            Same with (decoder_hidden == 0.).all(-1)

        Return:
        attention(context) -- attention is the result of attention which same with a_t in the handout
                            in shape (batch_size, hidden_dim)
        distribution(attenetion weights) -- distribution is the attention distribution same with alpha_t in the handout
                            in shape (batch_size, encoder_sequence_length)
        """
        pass

class DotAttention(AttentionBase):
    def forward(self,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_mask: torch.Tensor
    ):
        """ Dot product attention
        Implement dot product attention which compresses encoder_output on sequence_length axis by decoder_output

        Note 1: output should attent only on non-<PAD> encoder_output words

        Note 2: Try not to use 'for' iteration as you can. If you ues them, you will get only half of the score of this function.

        Hint: you may realize during the assignment 1 that the most easiest way to make SOMETHING to zero probability is 
        setting results of SOMETHING to -infinity by "result[SOMETHING] = float('-inf')" and do softmax on that dimension.

        Parameters / Returns: same as forward function in Attention base class
        """
        batch_size, sequence_length, hidden_dim = encoder_hidden.shape

        assert (encoder_mask == (encoder_hidden == 0.).all(-1)).all()
        assert (decoder_mask == (decoder_hidden == 0.).all(-1)).all()

        ### YOUR CODE HERE (~5 lines)
        attention: torch.Tensor = None
        distribution: torch.Tensor = None
        attention_energy = encoder_hidden.bmm(decoder_hidden.unsqueeze(-1)).squeeze(2)
        attention_energy[encoder_mask] = float('-inf')
        distribution = torch.nn.functional.softmax(attention_energy, dim=1)
        attention = encoder_hidden.transpose(1, 2).bmm(distribution.unsqueeze(-1)).squeeze(2)

        ### END YOUR CODE

        assert attention.shape == torch.Size([batch_size, hidden_dim])
        assert distribution.shape == torch.Size([batch_size, sequence_length])

        # Don't forget setting results of decoder <PAD> token values to zeros.
        # This would be helpful for debuging and other implementation details.
        attention[decoder_mask, :] = 0.

        return attention, distribution

class ConcatAttention(AttentionBase):
    def __init__(self, hidden_dim):
        """ Concat attention initializer
        Because there are variables in concat attention, you would need following attributes.
        Use these attributes in forward function

        Attributes:
        W_a -- Attention weight in the handout
                in shape (hidden_dim, 4 * hidden_dim)
        v_a -- Attention vector in the handout
                in shape (hidden_dim, )
        """
        super().__init__()

        self.W_a = nn.Parameter(torch.empty([hidden_dim, 4 * hidden_dim]))
        self.v_a = nn.Parameter(torch.empty([hidden_dim]))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.W_a.data)
        nn.init.normal_(self.v_a.data)

    def forward(self,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_mask: torch.Tensor
    ):
        """ Concat attention forward function
        Implement concat attention which compresses encoder_output on sequence_length axis by decoder_output

        Note: Try not to use 'for' iteration as you can. If you ues them, you will get only half of the score of this function.

        Parameters / Returns: same as forward function in Attention base class
        """
        batch_size, sequence_length, hidden_dim = encoder_hidden.shape

        assert (encoder_mask == (encoder_hidden == 0.).all(-1)).all()
        assert (decoder_mask == (decoder_hidden == 0.).all(-1)).all()

        ### YOUR CODE HERE (~6 lines)
        attention: torch.Tensor = None
        distribution: torch.Tensor = None

        decoder_hidden_expand = decoder_hidden.unsqueeze(dim=1).expand(batch_size,sequence_length,hidden_dim)
        concat_hidden = torch.cat((encoder_hidden, decoder_hidden_expand), dim=-1)
        C = concat_hidden.matmul(self.W_a.transpose(0,1))
        tanh_C = torch.tanh(C)
        attention_energy = tanh_C.matmul(self.v_a)
        attention_energy[encoder_mask] = float('-inf')
        distribution = torch.nn.functional.softmax(attention_energy, dim=1)
        attention = encoder_hidden.transpose(1, 2).bmm(distribution.unsqueeze(-1)).squeeze(-1)
        ### END YOUR CODE

        if attention.shape != torch.Size([batch_size, hidden_dim]):
            assert True
        assert distribution.shape == torch.Size([batch_size, sequence_length])

        # Don't forget setting results of decoder <PAD> token values to zeros.
        # This would be helpful for debuging and other implementation details.
        attention[decoder_mask, :] = 0.

        return attention, distribution

#############################################
# Testing functions below.                  #
#############################################

def dot_attention_test():
    print("======Dot Attention Test Case======")
    batch_size = 8
    sequence_length = 10
    hidden_dim = 2

    encoder_mask = torch.Tensor([[False] * (sequence_length - i) + [True] * i for i in range(0, batch_size)]).to(torch.bool)
    encoder_hidden = torch.normal(0, 1, [batch_size, sequence_length, hidden_dim])
    encoder_hidden[encoder_mask] = 0.
    decoder_hidden = torch.normal(0, 1, [batch_size, hidden_dim], requires_grad=True)
    decoder_mask = torch.Tensor([False] * batch_size).to(torch.bool)

    attention_function: Attention = DotAttention()

    attention, distribution = attention_function(encoder_hidden, encoder_mask, decoder_hidden, decoder_mask)

    # the first test
    assert distribution.sum(-1).allclose(torch.ones([batch_size])), \
        "Sum of the distribution is not one."
    print("The first test passed!")

    # the second test
    assert (distribution[encoder_mask] == 0.).all(), \
        "The distribution of <PAD> is not zeros"
    print("The second test passed!")

    # the third test
    expected_dist = torch.Tensor([[0.08083384, 0.10880914, 0.02757289],
                                  [0.04223004, 0.03302427, 0.01519270],
                                  [0.16208011, 0.07523207, 0.16550280]])
    assert distribution[:3,:3].allclose(expected_dist), \
        "Your attention distribution does not match expected result."
    print("The third test passed!")
    
    # the forth test
    expected_attn = torch.Tensor([[-0.22004035, -0.94449663],
                                  [-1.03192675,  0.59630102],
                                  [ 0.69415212, -0.21667977],
                                  [-0.13110159,  0.30330086]])
    assert attention[:4,:].allclose(expected_attn), \
        "Your attention result does not match expected result."
    print("The forth test passed!")

    # the fifth test
    (attention ** 2).sum().backward()
    expected_grad = torch.Tensor([[-0.31803015, -0.79647291],
                                  [-0.70453066,  0.68327230],
                                  [ 2.01890159, -0.34913260],
                                  [-0.31995568,  0.33685163]])
    assert decoder_hidden.grad[:4,:].allclose(expected_grad), \
        "Your attention gradient does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")

def concat_attention_test():
    print("======Concat Attention Test Case======")
    batch_size = 8
    sequence_length = 10
    hidden_dim = 2

    encoder_mask = torch.Tensor([[False] * (sequence_length - i) + [True] * i for i in range(0, batch_size)]).to(torch.bool)
    encoder_hidden = torch.normal(0, 1, [batch_size, sequence_length, hidden_dim * 2])
    encoder_hidden[encoder_mask] = 0.
    decoder_hidden = torch.normal(0, 1, [batch_size, hidden_dim * 2], requires_grad=True)
    decoder_mask = torch.Tensor([False] * batch_size).to(torch.bool)

    attention_function: AttentionBase = ConcatAttention(hidden_dim)

    attention, distribution = attention_function(encoder_hidden, encoder_mask, decoder_hidden, decoder_mask)

    # the first test
    assert distribution.sum(-1).allclose(torch.ones([batch_size])), \
        "Sum of the distribution is not one."
    print("The first test passed!")

    # the second test
    assert (distribution[encoder_mask] == 0.).all(), \
        "The distribution of <PAD> is not zeros"
    print("The second test passed!")

    # the third test
    expected_dist = torch.Tensor([[0.11372267, 0.04122099, 0.09134745],
                                  [0.13170099, 0.17426252, 0.03274137],
                                  [0.07704075, 0.06847615, 0.11857448]])
    assert distribution[:3,:3].allclose(expected_dist), \
        "Your attention distribution does not match expected result."
    print("The third test passed!")
    
    # the forth test
    expected_attn = torch.Tensor([[ 0.49971098, -0.13108040, -0.08359533],
                                  [ 0.30728793,  0.39589426, -0.09873602],
                                  [-0.05450586,  0.73270357, -0.76603746]])
    assert attention[:3,:3].allclose(expected_attn), \
        "Your attention result does not match expected result."
    print("The forth test passed!")

    # the fifth test
    (attention ** 2).sum().backward()
    expected_grad = torch.Tensor([[-1.77562630,  0.96031505, -0.87379438,  1.31658399],
                                  [-0.92111808, -1.06512451,  2.40100932, -3.36642313]])
    assert attention_function.W_a.grad[:2,:4].allclose(expected_grad), \
        "Your attention gradient does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    dot_attention_test()
    concat_attention_test()