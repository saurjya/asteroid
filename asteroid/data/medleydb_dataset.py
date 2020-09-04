import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
import random
import torchaudio

class MedleydbDataset(data.Dataset):
    dataset_name = "MedleyDB"

    def __init__(self, json_dir, n_src=1, n_poly=2, sample_rate=44100, segment=5.0, threshold=0.1):
        super().__init__()
        # Task setting
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.n_poly = n_poly
        self.threshold = threshold
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment)
        self.n_src = n_src
        self.like_test = self.seg_len is None
        # Load json files
        sources_json = [
            os.path.join(json_dir, source + ".json")
            for source in [f"inst{n+1}" for n in range(n_src)]
        ]
        sources_conf = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_conf = np.array(json.load(f))

        # Filter out short utterances only when segment is specified
        # orig_len = len(mix_infos)
        drop_utt, drop_len, orig_len = 0, 0, 0
        sources_infos = []
        index_array = []
        
        if not self.like_test:
            for i in range(len(sources_conf)):
                conf = sources_conf[i][1]
                #print(sources_conf[i][0])
                #index_array = []
                duration = sources_conf[i][1][-1][0]
                index_array.append(np.zeros(int(duration//segment) + 1))
                for timestamp, confidence in conf:
                    j = int(timestamp // segment)
                    #print(j)
                    index_array[i][j] = index_array[i][j] + confidence
                orig_len = orig_len + duration
                seg_dur = duration / len(index_array[i])

                for k in range(len(index_array[i])):
                    conf_thresh = threshold * float(len(sources_conf[i][0]))
                    if index_array[i][k] < conf_thresh:
                        drop_utt += 1
                        drop_len += seg_dur
                        continue
                    else:
                        sources_infos.append((sources_conf[i][0], k, sources_conf[i][2]))

        print(
            "Drop {} utts ({:.2f} h) from ({:.2f} h) with less than {} percent activity".format(
                drop_utt, drop_len / 3600, orig_len / 3600, threshold
            )
        )
        # self.mix = mix_infos
        #print(sources_infos[0])
        self.sources = sources_infos

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Load sources
        source_arrays = []
        
        for i in range(self.n_poly):
            if i:
                idx = random.choice(range(len(self.sources)))

            start = self.sources[idx][1] * self.sources[idx][2]
            if self.like_test:
                stop = None
            else:
                stop = start + (self.seg_len * self.sources[idx][2])

            if self.sources[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((self.seg_len * self.sources[idx][2],))
            else:
                s, sr = sf.read(self.sources[idx][0], start=start, stop=stop, dtype="float32")
            source_arrays.append(s)
        source = torch.from_numpy(np.vstack(source_arrays))
        source = torchaudio.transforms.Resample(sr,self.sample_rate)(source)
        mix = torch.stack(list(source)).sum(0)
        return mix, source

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "harmony_sep"
        infos["licenses"] = [wsj0_license]
        return infos


wsj0_license = dict(
    title="CSR-I (WSJ0) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC93S6A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)
