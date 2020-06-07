import torch.utils.data as data
import collections
from pathlib import Path
from typing import Collection, Iterator


# noinspection SpellCheckingInspection
class ConllUEntry:
    def __init__(self, id_, form, lemma, upostag, xpostag, feats, head, deprel, deps, misc):
        self.id_ = int(id_)
        self.form = form
        self.lemma = lemma
        self.upostag = upostag
        self.xpostag = xpostag
        self.feats = feats
        self.head = int(head)
        self.deprel = deprel
        self.deps = deps
        self.misc = misc


# noinspection SpellCheckingInspection
class ConllUDataset(data.IterableDataset):
    unspecified = '_'
    root = ConllUEntry(0, '<ROOT>', '<ROOT>', '<ROOT>', '<ROOT>', unspecified, -1, 'RROOT', unspecified, unspecified)

    def __init__(self, path: Path):
        super(ConllUDataset).__init__()
        self.path: Path = path

    def __iter__(self) -> Iterator[Collection[ConllUEntry]]:
        sentence = []
        with self.path.open('r') as conllu:
            for line in conllu.readlines():
                tokens = list(line.split('\t'))
                if not tokens or line.strip() == '':
                    if len(sentence) > 1:
                        yield [self.root] + sentence
                        sentence = []
                    continue
                sentence.append(ConllUEntry(*tokens))
        if len(sentence) > 1:
            yield sentence


# noinspection SpellCheckingInspection
class IndexedConllUDataset(data.Dataset):
    def __init__(self, path: Path, transform: bool = False, reserve_oov: bool = True):
        super(IndexedConllUDataset).__init__()
        self.reserve_oov = reserve_oov
        self.sentences = []
        self.vocab = collections.Counter()
        self.pos = set()
        self.word_counts = None
        for sentence in ConllUDataset(path):
            words, tags, heads = [], [], []
            for entry in sentence:
                self.vocab[entry.form] += 1
                self.pos.add(entry.upostag)
                words.append(entry.form)
                tags.append(entry.upostag)
                heads.append(entry.head)
            self.sentences.append((words, tags, heads))
        if transform:
            self.word_to_idx = {w: i + reserve_oov for i, w in enumerate(self.vocab)}
            self.pos_to_idx = {t: i + reserve_oov for i, t in enumerate(self.pos)}
            if reserve_oov:
                self.word_to_idx['<OOV>'] = self.pos_to_idx['<OOV>'] = 0  # Reserve index 0 for out-of-vocabulary words
            self.transform(self.word_to_idx, self.pos_to_idx)

    def transform(self, word_to_idx, pos_to_idx):
        transformed = []
        for words, pos, heads in self.sentences:
            if self.reserve_oov:
                word_idx = [word_to_idx.get(w, 0) for w in words]
                pos_idx = [pos_to_idx.get(t, 0) for t in pos]
            else:
                word_idx = [word_to_idx[w] for w in words]
                pos_idx = [pos_to_idx[t] for t in pos]
            transformed.append((word_idx, pos_idx, heads))
        self.sentences = transformed
        self.word_counts = {word_to_idx[i]: c for i, c in self.vocab.items() if i in word_to_idx}
        if self.reserve_oov:
            self.word_counts[0] = 0

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)
