
import tensorflow as tf
from neural import dynamicBiRNN, LReLu, MLP, get_structure
import numpy as np



class StructureModel():
    def __init__(self, config):
        self.config = config
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        t_variables['batch_l'] = tf.placeholder(tf.int32)
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None, None])
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['max_sent_l'] = tf.placeholder(tf.int32)
        t_variables['max_doc_l'] = tf.placeholder(tf.int32)
        t_variables['gold_labels'] = tf.placeholder(tf.int32, [None])
        t_variables['mask_tokens'] = tf.placeholder(tf.float32, [None, None, None])
        t_variables['mask_sents'] = tf.placeholder(tf.float32, [None, None])
        t_variables['mask_parser_1'] = tf.placeholder(tf.float32, [None, None, None])
        t_variables['mask_parser_2'] = tf.placeholder(tf.float32, [None, None, None])
        self.t_variables = t_variables


    def get_feed_dict(self, batch):
        batch_size = len(batch)
        doc_l_matrix = np.zeros([batch_size], np.int32)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            doc_l_matrix[i] = n_sents
        max_doc_l = np.max(doc_l_matrix)
        max_sent_l = max([max([len(sent) for sent in doc.token_idxs]) for doc in batch])
        token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
        sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
        gold_matrix = np.zeros([batch_size], np.int32)
        mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float32)
        mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)
        for i, instance in enumerate(batch):
            n_sents = len(instance.token_idxs)
            gold_matrix[i] = instance.goldLabel
            for j, sent in enumerate(instance.token_idxs):
                token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
                mask_tokens_matrix[i, j, len(sent):] = 0
                sent_l_matrix[i, j] = len(sent)
            mask_sents_matrix[i, n_sents:] = 0
        mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        mask_parser_1[:, :, 0] = 0
        mask_parser_2[:, 0, :] = 0
        if (self.config.large_data):
            if (batch_size * max_doc_l * max_sent_l * max_sent_l > 16 * 200000):
                return [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]

        feed_dict = {self.t_variables['token_idxs']: token_idxs_matrix, self.t_variables['sent_l']: sent_l_matrix,
                     self.t_variables['mask_tokens']: mask_tokens_matrix, self.t_variables['mask_sents']: mask_sents_matrix,
                     self.t_variables['doc_l']: doc_l_matrix, self.t_variables['gold_labels']: gold_matrix,
                     self.t_variables['max_sent_l']: max_sent_l, self.t_variables['max_doc_l']: max_doc_l,
                     self.t_variables['mask_parser_1']: mask_parser_1, self.t_variables['mask_parser_2']: mask_parser_2,
                     self.t_variables['batch_l']: batch_size, self.t_variables['keep_prob']:self.config.keep_prob}
        return  feed_dict

    def build(self):
        with tf.variable_scope("Embeddings"):
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root = tf.get_variable("emb_root", [1, 1, 2 * self.config.dim_sem], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root_s = tf.get_variable("emb_root_s", [1, 1,2* self.config.dim_sem], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("Model"):
            w_comb = tf.get_variable("w_comb", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb = tf.get_variable("bias_comb", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_comb_s = tf.get_variable("w_comb_s", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb_s = tf.get_variable("bias_comb_s", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_softmax = tf.get_variable("w_softmax", [2 * self.config.dim_sem, self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_softmax = tf.get_variable("bias_softmax", [self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Structure/doc"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("Structure/sent"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        sent_l = self.t_variables['sent_l']
        doc_l = self.t_variables['doc_l']
        max_sent_l = self.t_variables['max_sent_l']
        max_doc_l = self.t_variables['max_doc_l']
        batch_l = self.t_variables['batch_l']

        tokens_input = tf.nn.embedding_lookup(self.embeddings, self.t_variables['token_idxs'][:, :max_doc_l, :max_sent_l])
        tokens_input = tf.nn.dropout(tokens_input, self.t_variables['keep_prob'])

        mask_tokens = self.t_variables['mask_tokens'][:, :max_doc_l, :max_sent_l]
        mask_sents = self.t_variables['mask_sents'][:, :max_doc_l]
        [_, _, _, rnn_size] = tokens_input.get_shape().as_list()
        tokens_input_do = tf.reshape(tokens_input, [batch_l * max_doc_l, max_sent_l, rnn_size])

        sent_l = tf.reshape(sent_l, [batch_l * max_doc_l])
        mask_tokens = tf.reshape(mask_tokens, [batch_l * max_doc_l, -1])

        tokens_output, _ = dynamicBiRNN(tokens_input_do, sent_l, n_hidden=self.config.dim_hidden,
                                        cell_type=self.config.rnn_cell, cell_name='Model/sent')
        tokens_sem = tf.concat([tokens_output[0][:,:,:self.config.dim_sem], tokens_output[1][:,:,:self.config.dim_sem]], 2)
        tokens_str = tf.concat([tokens_output[0][:,:,self.config.dim_sem:], tokens_output[1][:,:,self.config.dim_sem:]], 2)
        temp1 = tf.zeros([batch_l * max_doc_l, max_sent_l,1], tf.float32)
        temp2 = tf.zeros([batch_l * max_doc_l,1,max_sent_l], tf.float32)

        mask1 = tf.ones([batch_l * max_doc_l, max_sent_l, max_sent_l-1], tf.float32)
        mask2 = tf.ones([batch_l * max_doc_l, max_sent_l-1, max_sent_l], tf.float32)
        mask1 = tf.concat([temp1,mask1],2)
        mask2 = tf.concat([temp2,mask2],1)

        str_scores_s_ = get_structure('sent', tokens_str, max_sent_l, mask1, mask2)  # batch_l,  sent_l+1, sent_l
        str_scores_s = tf.matrix_transpose(str_scores_s_)  # soft parent
        tokens_sem_root = tf.concat([tf.tile(embeddings_root_s, [batch_l * max_doc_l, 1, 1]), tokens_sem], 1)
        tokens_output_ = tf.matmul(str_scores_s, tokens_sem_root)
        tokens_output = LReLu(tf.tensordot(tf.concat([tokens_sem, tokens_output_], 2), w_comb_s, [[2], [0]]) + b_comb_s)

        if (self.config.sent_attention == 'sum'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)
        elif (self.config.sent_attention == 'mean'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)/tf.expand_dims(tf.cast(sent_l,tf.float32),1)
        elif (self.config.sent_attention == 'max'):
            tokens_output = tokens_output + tf.expand_dims((mask_tokens-1)*999,2)
            tokens_output = tf.reduce_max(tokens_output, 1)

        sents_input = tf.reshape(tokens_output, [batch_l, max_doc_l, 2*self.config.dim_sem])
        sents_output, _ = dynamicBiRNN(sents_input, doc_l, n_hidden=self.config.dim_hidden, cell_type=self.config.rnn_cell, cell_name='Model/doc')

        sents_sem = tf.concat([sents_output[0][:,:,:self.config.dim_sem], sents_output[1][:,:,:self.config.dim_sem]], 2)
        sents_str = tf.concat([sents_output[0][:,:,self.config.dim_sem:], sents_output[1][:,:,self.config.dim_sem:]], 2)

        str_scores_ = get_structure('doc', sents_str,max_doc_l, self.t_variables['mask_parser_1'], self.t_variables['mask_parser_2'])  #batch_l,  sent_l+1, sent_l
        str_scores = tf.matrix_transpose(str_scores_)  # soft parent
        sents_sem_root = tf.concat([tf.tile(embeddings_root, [batch_l, 1, 1]), sents_sem], 1)
        sents_output_ = tf.matmul(str_scores, sents_sem_root)
        sents_output = LReLu(tf.tensordot(tf.concat([sents_sem, sents_output_], 2), w_comb, [[2], [0]]) + b_comb)

        if (self.config.doc_attention == 'sum'):
            sents_output = sents_output * tf.expand_dims(mask_sents,2)
            sents_output = tf.reduce_sum(sents_output, 1)
        elif (self.config.doc_attention == 'mean'):
            sents_output = sents_output * tf.expand_dims(mask_sents,2)
            sents_output = tf.reduce_sum(sents_output, 1)/tf.expand_dims(tf.cast(doc_l,tf.float32),1)
        elif (self.config.doc_attention == 'max'):
            sents_output = sents_output + tf.expand_dims((mask_sents-1)*999,2)
            sents_output = tf.reduce_max(sents_output, 1)

        final_output = MLP(sents_output, 'output', self.t_variables['keep_prob'])
        self.final_output = tf.matmul(final_output, w_softmax) + b_softmax

    def get_loss(self):
        if (self.config.opt == 'Adam'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        else:
            optimizer = tf.train.AdagradOptimizer(self.config.lr)
        with tf.variable_scope("Model"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.final_output,
                                                                  labels=self.t_variables['gold_labels'])
            self.loss = tf.reduce_mean(self.loss)
            model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Model')
            str_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Structure')
            for p in model_params + str_params:
                if ('bias' not in p.name):
                    self.loss += self.config.norm * tf.nn.l2_loss(p)
            self.opt = optimizer.minimize(self.loss)


