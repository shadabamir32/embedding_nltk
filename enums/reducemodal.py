from enum import Enum

class ReduceModal(str, Enum):
    PCA = "PCA"
    TSNE = "TSNE"