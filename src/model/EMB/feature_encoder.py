import random

import numpy as np
import torch
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from ...datasets import who_is_who, oc782k

