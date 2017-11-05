import gensim
import numpy as np
import re
import random
import math
import unicodedata
import itertools
from utils import grouper
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', unicode(s,'utf-8'))
                  if unicodedata.category(c) != 'Mn')

class RawData:
    def __init__(self):
        self.userStr = ''
        self.productStr = ''
        self.reviewText = ''
        self.goldRating = -1
        self.predictedRating = -1
        self.userStr = ''


class DataSet:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(self.data)

    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._max_sent_len)
        self.data = sorted(self.data, key=lambda x: x._doc_len)

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        idxs = list(range(self.num_examples))
        _grouped = lambda: list(grouper(idxs, batch_size))

        if(rand):
            grouped = lambda: random.sample(_grouped(), num_batches_per_epoch)
        else:
            grouped = _grouped
        num_steps = num_epochs*num_batches_per_epoch
        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for i in range(num_steps):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            yield i,batch_data


class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

class Corpus:
    def __init__(self):
        self.doclst = {}

    def load(self, in_path, name):
        self.doclst[name] = []
        for line in open(in_path):
            items = line.split('<split1>')
            doc = RawData()
            doc.goldRating = int(items[0])
            doc.reviewText = items[1]
            self.doclst[name].append(doc)
    def preprocess(self):
        random.shuffle(self.doclst['train'])
        for dataset in self.doclst:
            for doc in self.doclst[dataset]:
                doc.sent_lst = doc.reviewText.split('<split2>')
                doc.sent_lst = [re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sent) for sent in doc.sent_lst]
                doc.sent_token_lst = [sent.split() for sent in doc.sent_lst]
                doc.sent_token_lst = [sent_tokens for sent_tokens in doc.sent_token_lst if(len(sent_tokens)!=0)]
            self.doclst[dataset] = [doc for doc in self.doclst[dataset] if len(doc.sent_token_lst)!=0]

    def build_vocab(self):
        self.vocab = {}
        for doc in self.doclst:
            for sent in doc.sent_token_lst:
                for token in sent:
                    if(token not in self.vocab):
                        self.vocab[token] = {'idx':len(self.vocab), 'count':1}
                    else:
                        self.vocab[token]['count'] += 1
    def w2v(self, options):
        sentences = []
        for doc in self.doclst['train']:
            sentences.extend(doc.sent_token_lst)
        if('dev' in self.doclst):
            for doc in self.doclst['dev']:
                sentences.extend(doc.sent_token_lst)
        print(sentences[0])
        if(options['skip_gram']):
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4, sg=1)
        else:
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4)
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run = True)  # trim by min_count & precalculate downsampling
        print(rtn)
        self.w2v_model.finalize_vocab()  # build tables & arrays
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        self.vocab = self.w2v_model.wv.vocab
        print('Vocab size: {}'.format(len(self.vocab)))

        # model.save('../data/w2v.data')

    def prepare(self, options):
        instances, instances_dev, instances_test = [],[],[]
        instances, embeddings, vocab = self.prepare_for_training(options)
        if ('dev' in self.doclst):
            instances_dev = self.prepare_for_test(options, 'dev')
        instances_test = self.prepare_for_test( options, 'test')
        return instances, instances_dev, instances_test, embeddings, vocab

    def prepare_notest(self, options):
        instances, instances_dev, instances_test = [],[],[]
        instances_, embeddings, vocab = self.prepare_for_training(options)
        print(len(instances))
        for bucket in instances_:
            num_test = len(bucket) / 10
            instances_test.append(bucket[:num_test])
            instances.append(bucket[num_test:])

        return instances, instances_dev, instances_test, embeddings, vocab


    def prepare_for_training(self, options):
        instancelst = []
        embeddings = np.zeros([len(self.vocab)+1,options['emb_size']])
        for word in self.vocab:
            embeddings[self.vocab[word].index] = self.w2v_model[word]
        self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        n_filtered = 0
        for i_doc, doc in enumerate(self.doclst['train']):
            instance = Instance()
            instance.idx = i_doc
            n_sents = len(doc.sent_token_lst)
            max_n_tokens = max([len(sent) for sent in doc.sent_token_lst])
            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue
            sent_token_idx = []
            for i in range(len(doc.sent_token_lst)):
                token_idxs = []
                for token in doc.sent_token_lst[i]:
                    if(token in self.vocab):
                        token_idxs.append(self.vocab[token].index)
                    else:
                        token_idxs.append(self.vocab['UNK'].index)
                sent_token_idx.append(token_idxs)
            instance.token_idxs = sent_token_idx
            instance.goldLabel = doc.goldRating
            instancelst.append(instance)
        print('n_filtered in train: {}'.format(n_filtered))
        return instancelst, embeddings, self.vocab
    def prepare_for_test(self, options, name):
        instancelst = []
        n_filtered = 0
        for i_doc, doc in enumerate(self.doclst[name]):
            instance = Instance()
            instance.idx = i_doc
            n_sents = len(doc.sent_token_lst)
            max_n_tokens = max([len(sent) for sent in doc.sent_token_lst])
            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue
            sent_token_idx = []
            for i in range(len(doc.sent_token_lst)):
                token_idxs = []
                for token in doc.sent_token_lst[i]:
                    if(token in self.vocab):
                        token_idxs.append(self.vocab[token].index)
                    else:
                        token_idxs.append(self.vocab['UNK'].index)
                sent_token_idx.append(token_idxs)
            instance.token_idxs = sent_token_idx
            instance.goldLabel = doc.goldRating
            instancelst.append(instance)
        print('n_filtered in {}: {}'.format(name, n_filtered))
        return instancelst