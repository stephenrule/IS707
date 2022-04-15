import json
import os
import tempfile
from copy import deepcopy
from typing import Dict, Iterable, List

import torch
from allennlp.common import JsonDict
from allennlp.common.params import Params
from allennlp.data import (
    Field,
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.models import Model
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict

class modelload():
    def loadmodelfrompath(path: str):
        serialization_dir = path
        config_file = os.path.join(serialization_dir, "config.json")
        vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
        weights_file = os.path.join(serialization_dir, "weights.th")
        loaded_params = Params.from_file(config_file)
        loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
        loaded_vocab = loaded_model.vocab  # Vocabulary is loaded in Model.load()
        return loaded_model, loaded_vocab

class BiDAF_predictor(Predictor):

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage": passage, "question": question})
model, voc = modelload.loadmodelfrompath('/home/kxt/project/IS707/BiDAF')
BF = BiDAF_predictor(model)


