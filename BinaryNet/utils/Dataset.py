import torchaudio
import torch

import os

from BinaryNet.utils.SoundUtils import pad_or_trim_center

class FilteredSubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, subset=None, target=None):
        super().__init__(root, download=False)
        def load_list(filename):
            with open(os.path.join(self._path, filename)) as f:
                return set(os.path.normpath(line.strip()) for line in f)
        val = load_list("validation_list.txt")
        test = load_list("testing_list.txt")

        PATH_SPLITTER = "\\"
        all_paths = set(os.path.normpath(p.replace(self._path, "").lstrip(PATH_SPLITTER)) for p in self._walker)

        self.subset = subset
        if subset == "validation":
            self._walker = [os.path.join(self._path, p) for p in all_paths & val]
        elif subset == "testing":
            self._walker = [os.path.join(self._path, p) for p in all_paths & test]
        elif subset == "training":
            self._walker = [os.path.join(self._path, p) for p in all_paths - (test | val)]

        self.target = target
        if target is not None:
            check_target = lambda x: any(PATH_SPLITTER + t + PATH_SPLITTER in x for t in target)
            self._walker = list(filter(check_target, self._walker))
        else:
            self.target = list(set(p.replace(self._path, "").lstrip(PATH_SPLITTER).split(PATH_SPLITTER)[0] for p in self._walker))

TARGET_WORDS = ['yes','no','up','down','left','right','on','off','stop','go']

class SCWithFeatures(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, i):
        w, sr, label, *_ = self.base[i]
        w = pad_or_trim_center(w)
        x = self.transform(w)
        return x, self.base.target.index(label)