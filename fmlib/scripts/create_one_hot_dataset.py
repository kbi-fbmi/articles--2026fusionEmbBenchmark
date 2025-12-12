import pickle

import numpy as np
from tqdm import tqdm

from fmlib import fm, io

PATH_DATA = "/mnt/e/Data/Fuse/fusionai_train_sim.txt"
fusion_data = io.load_fusions_from_fusionaitxt(PATH_DATA)


sq1a = fm.extr_key(fusion_data, "sequence1")
sq1 = np.array([io.convert_sequence_to_onehot_ACGT(seq) for seq in tqdm(sq1a, desc="Encoding sq1")])
    pickle.dump(sq1, f)
with open("sq1_oh_train.pkl", "wb") as f:
del sq1, sq1a

sq2a = fm.extr_key(fusion_data, "sequence2")
sq2 = np.array([io.convert_sequence_to_onehot_ACGT(seq) for seq in tqdm(sq2a, desc="Encoding sq2")])
with open("sq2_oh_train.pkl", "wb") as f:
    pickle.dump(sq2, f)
del sq2, sq2a
