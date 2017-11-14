# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#http://colah.github.io/
#  https://www.tensorflow.org/tutorials/recurrent
#
# uses: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# LSTMs - are what is mostly used
#
# intro - reccurent neural nets - NN's with loops in them for persistance
# related to sequences and list - due to chaining (as we can think of the loop as an infinite chain)
# past info for present task? can remember long term dependencies on information
#
# - 4 layers per module?
# core idea - cell state.
#
# (reminds me of the notion of binary)
#
# LSTM stages
#
# 1. throw away from cell state - uses sigmoid layer
# 2. new info for cell state - uses sigmoid layers (for input gate) + tanh layer (for vector)
# 3. update cell state
