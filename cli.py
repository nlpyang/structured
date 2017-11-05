import os

import tensorflow as tf


from main import run as m

# options = dict( rnn_cell='lstm', mode='two-structure', data_file='yelp-2013-largevocab3.pkl',
#                max_sents=100, max_tokens=200, batch_size=16, lstm_hidden_dim_t=75, sent_attention='max',
#                doc_attention='att',
#                mlp_output=False, dropout=0.7, grad_clip='global', opt='Adagrad', lr=0.02, norm=1e-4, short_att=True,
#                lstm_hidden_dim_d=75, sem_dim=75, str_dim=50,
#                comb_atv='tanh')

flags = tf.app.flags

flags.DEFINE_string("rnn_cell", "lstm", "rnn_cell")
flags.DEFINE_string("data_file", "../data/yelp-2013-fake.pkl", "data_file")

flags.DEFINE_integer("batch_size", 16, "batch_size")
flags.DEFINE_integer("epochs", 30, "epochs")

flags.DEFINE_integer("dim_str", 50, "dim_str")
flags.DEFINE_integer("dim_sem", 75, "dim_sem")
flags.DEFINE_integer("dim_output", 5, "dim_output")
flags.DEFINE_float("keep_prob", 0.7, "keep_prob")
flags.DEFINE_string("opt", 'Adagrad', "opt")
flags.DEFINE_float("lr", 0.05, "lr")
flags.DEFINE_float("norm", 1e-4, "norm")
flags.DEFINE_integer("gpu", -1, "gpu")

flags.DEFINE_string("sent_attention", "max", "sent_attention")
flags.DEFINE_string("doc_attention", "max", "doc_attention")
flags.DEFINE_bool("large_data", False, "large_data")
flags.DEFINE_integer("log_period", 5000, "log_period")


def main(_):
    config = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    m(config)

if __name__ == "__main__":
    tf.app.run()
