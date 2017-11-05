import tensorflow as tf

def LReLu(x, leak=0.01):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def dynamicBiRNN(input, seqlen, n_hidden, cell_type, cell_name=''):
    batch_size = tf.shape(input)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32):
        if(cell_type == 'gru'):
            fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        elif(cell_type == 'lstm'):
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden)

        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32):
        if(cell_type == 'gru'):
            bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        elif(cell_type == 'lstm'):
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden)
        bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 sequence_length=seqlen)
    return outputs, output_states

def MLP(input, vname, keep_prob):
    dim_input  = input.shape[1]
    with tf.variable_scope("Model"):
        w1 = tf.get_variable("w1_"+vname,[dim_input, dim_input],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("bias1_" + vname,[dim_input],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())
        w2 = tf.get_variable("w2_" + vname,[dim_input, dim_input],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("bias2_" + vname,[dim_input],
                            dtype=tf.float32,
                             initializer=tf.constant_initializer())
    input = tf.nn.dropout(input,keep_prob)
    h1 = LReLu(tf.matmul(input, w1) + b1)
    h1 = tf.nn.dropout(h1,keep_prob)
    h2 = LReLu(tf.matmul(h1, w2) + b2)
    return h2


def get_structure(name, input, max_l, mask_parser_1, mask_parser_2):
    def _getDep(input, mask1, mask2):
        #input: batch_l, sent_l, rnn_size
        with tf.variable_scope("Structure/"+name, reuse=True, dtype=tf.float32):
            w_parser_p = tf.get_variable("w_parser_p")
            w_parser_c = tf.get_variable("w_parser_c")
            b_parser_p = tf.get_variable("bias_parser_p")
            b_parser_c = tf.get_variable("bias_parser_c")

            w_parser_s = tf.get_variable("w_parser_s")
            w_parser_root = tf.get_variable("w_parser_root")

        parent = tf.tanh(tf.tensordot(input, w_parser_p, [[2], [0]]) + b_parser_p)
        child = tf.tanh(tf.tensordot(input, w_parser_c, [[2], [0]])+b_parser_c)
        # rep = LReLu(parent+child)
        temp = tf.tensordot(parent,w_parser_s,[[-1],[0]])
        raw_scores_words_ = tf.matmul(temp,tf.matrix_transpose(child))

        # raw_scores_words_ = tf.squeeze(tf.tensordot(rep, w_parser_s, [[3], [0]]) , [3])
        raw_scores_root_ = tf.squeeze(tf.tensordot(input, w_parser_root, [[2], [0]]) , [2])
        raw_scores_words = tf.exp(raw_scores_words_)
        raw_scores_root = tf.exp(raw_scores_root_)
        tmp = tf.zeros_like(raw_scores_words[:,:,0])
        raw_scores_words = tf.matrix_set_diag(raw_scores_words,tmp)

        str_scores, LL = _getMatrixTree(raw_scores_root, raw_scores_words, mask1, mask2)
        return str_scores

    def _getMatrixTree(r, A, mask1, mask2):
        L = tf.reduce_sum(A, 1)
        L = tf.matrix_diag(L)
        L = L - A
        LL = L[:, 1:, :]
        LL = tf.concat([tf.expand_dims(r, [1]), LL], 1)
        LL_inv = tf.matrix_inverse(LL)  #batch_l, doc_l, doc_l
        d0 = tf.multiply(r, LL_inv[:, :, 0])
        LL_inv_diag = tf.expand_dims(tf.matrix_diag_part(LL_inv), 2)
        tmp1 = tf.matrix_transpose(tf.multiply(tf.matrix_transpose(A), LL_inv_diag))
        tmp2 = tf.multiply(A, tf.matrix_transpose(LL_inv))
        d = mask1 * tmp1 - mask2 * tmp2
        d = tf.concat([tf.expand_dims(d0,[1]), d], 1)
        return d, LL

    str_scores = _getDep(input, mask_parser_1, mask_parser_2)
    return str_scores