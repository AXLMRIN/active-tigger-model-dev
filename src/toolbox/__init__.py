ROOT_MODELS = "./models"
ROOT_DATA = "./data"
ROOT_RESULTS = "./results"
ROOT_FIGURES = "./figures"

from .train_embedding_models import *
from .test_embedding_models import *
from .optimize_classifiers import *
from .save_embeddings import *
from .visualise_results import *
from .general import *