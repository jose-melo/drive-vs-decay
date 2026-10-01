import os

from scipy.io import arff
import pandas as pd

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE

class Jannis(BaseDataset):

    def __init__(self, args):
        super(Jannis, self).__init__(args)

        self.args = args
        self.is_data_loaded = False
        self.name = "jannis"
        self.tmp_file_names = ["jannis.arff"]
        self.task_type = TASK_TYPE.MULTI_CLASS

    def load(self):

        data, _ = arff.loadarff(os.path.join(self.data_path, self.tmp_file_names[0]))
        data = pd.DataFrame(data)

        self.X = data.iloc[:, 1:].to_numpy()
        self.y = data["class"].apply(int).to_numpy()

        self.N, self.D = self.X.shape

        self.cardinalities = []
        self.num_or_cat = {}

        self.cat_features = []
        self.num_features = list(range(self.D))

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loaded = True
