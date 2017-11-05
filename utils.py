import numpy as np
from six.moves import zip_longest

class Data:
    def __init__(self):
        self.userStr = ''
        self.productStr = ''
        self.reviewText = ''
        self.goldRating = -1
        self.predictedRating = -1
        self.userStr = ''

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1

class Corpus:
    def __init__(self):
        self.doclst = []
        self.instancelst = []
        self.instance_in_buckets = []
        # self.productStr = ''
    def preprocess(self):
        for doc in self.doclst:
            doc.sent_lst = doc.reviewText.split('<sssss>')
            # print(len(ins.sent_lst))
            doc.sent_token_lst = [sent.split() for sent in doc.sent_lst]
            # ins.sent_str_lst = ins.reviewText.split('<sssss>')
    def build_vocab(self):
        self.vocab = {}
        for doc in self.doclst:
            for sent in doc.sent_token_lst:
                for token in sent:
                    if(token not in self.vocab):
                        self.vocab[token] = {'idx':len(self.vocab), 'count':1}
                    else:
                        self.vocab[token]['count'] += 1

    def w2v(self):
        sentences = []
        for doc in self.doclst:
            sentences.extend(doc.sent_token_lst)
        print(sentences[0])
        self.w2v_model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=10, workers=4)
        self.vocab = self.w2v_model.vocab
        print('Vocab size:{}'.format(len(self.vocab)))
        # model.save('../data/w2v.data')

    def prepare_for_training(self, options):
        self.instance_in_buckets = [[] for _ in options['buckets']]
        embeddings = np.zeros([len(self.vocab)+1,100])
        for word in self.vocab:
            embeddings[self.vocab[word].index] = self.w2v_model[word]
        self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        n_filtered = 0
        for doc in self.doclst:
            instance = Instance()

            n_sents = len(doc.sent_token_lst)
            max_n_tokens = max([len(sent) for sent in doc.sent_token_lst])
            if(n_sents>options['max_sents']):
                n_filtered += 1
                continue
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue
            i_bucket = 0
            for i,bucket in enumerate(options['buckets']):
                if(n_sents<=bucket):
                    i_bucket = i
                    break
            # token_matrix = np.zeros([n_sents,options['max_tokens']],dtype=np.int32)
            sent_token_idx = []
            for i in range(len(doc.sent_token_lst)):
                token_idxs = []
                for token in doc.sent_token_lst[i]:
                    if(token in self.vocab):
                        token_idxs.append(self.vocab[token].index)
                    else:
                        token_idxs.append(self.vocab['UNK'].index)
                sent_token_idx.append(token_idxs)
            #     token_matrix[i,:len(token_idxs)] = np.asarray(token_idxs)
            instance.token_idxs = sent_token_idx
            instance.goldLabel = doc.goldRating
            self.instancelst.append(instance)
            self.instance_in_buckets[i_bucket].append(instance)
        print('n_filtered: {}'.format(n_filtered))
        return self.instance_in_buckets, embeddings

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    out = list(out)
    if num_groups is not None:
        default = (fillvalue,) * n
        assert isinstance(num_groups, int)
        out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
    if shorten:
        assert fillvalue is None
        out = (tuple(e for e in each if e is not None) for each in out)
    return out
