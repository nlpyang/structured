from data_structure import DataSet
import tensorflow as tf
import numpy as np
import cPickle
import logging
from models import  StructureModel
import tqdm

def load_data(config):
    train, dev, test, embeddings, vocab = cPickle.load(open(config.data_file))
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    vocab = dict([(v.index,k) for k,v in vocab.items()])
    trainset.sort()
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab

def evaluate(sess, model, test_batches):
    corr_count, all_count = 0, 0
    for ct, batch in test_batches:
        feed_dict = model.get_feed_dict(batch)
        feed_dict[model.t_variables['keep_prob']] = 1
        predictions = sess.run(model.final_output, feed_dict=feed_dict)

        predictions = np.argmax(predictions, 1)
        corr_count += np.sum(predictions == feed_dict[model.t_variables['gold_labels']])
        all_count += len(batch)
    acc_test = 1.0 * corr_count / all_count
    return  acc_test

def run(config):
    import random

    hash = random.getrandbits(32)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ah = logging.FileHandler(str(hash)+'.log')
    ah.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ah.setFormatter(formatter)
    logger.addHandler(ah)

    num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab = load_data(config)
    print(embedding_matrix.shape)
    config.n_embed, config.d_embed = embedding_matrix.shape

    config.dim_hidden = config.dim_sem+config.dim_str

    print(config.__flags)
    logger.critical(str(config.__flags))

    model = StructureModel(config)
    model.build()
    model.get_loss()
    # trainer = Trainer(config)

    num_batches_per_epoch = int(num_examples / config.batch_size)
    num_steps = config.epochs * num_batches_per_epoch

    with tf.Session() as sess:
        gvi = tf.global_variables_initializer()
        sess.run(gvi)
        sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))
        loss = 0

        for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
            feed_dict = model.get_feed_dict(batch)
            outputs,_,_loss = sess.run([model.final_output, model.opt, model.loss], feed_dict=feed_dict)
            loss+=_loss
            if(ct%config.log_period==0):
                acc_test = evaluate(sess, model, test_batches)
                acc_dev = evaluate(sess, model, dev_batches)
                print('Step: {} Loss: {}\n'.format(ct, loss))
                print('Test ACC: {}\n'.format(acc_test))
                print('Dev  ACC: {}\n'.format(acc_dev))
                logger.debug('Step: {} Loss: {}\n'.format(ct, loss))
                logger.debug('Test ACC: {}\n'.format(acc_test))
                logger.debug('Dev  ACC: {}\n'.format(acc_dev))
                logger.handlers[0].flush()
                loss = 0

