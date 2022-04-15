import json
import os
import tempfile
from copy import deepcopy
from typing import Dict, Iterable, List
import allennlp_models
import numpy
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
from allennlp.data.fields import LabelField, TextField, ListField, SpanField, SequenceLabelField, IndexField
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

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from allennlp_models.rc.metrics import SquadEmAndF1
import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

from allennlp_models.rc.dataset_readers import utils

logger = logging.getLogger(__name__)

SQUAD2_NO_ANSWER_TOKEN = "@@<NO_ANSWER>@@"
"""
The default `no_answer_token` for the [`squad2`](#squad2) reader.
"""
from allennlp_models.rc.models.utils import (
    get_best_span,
    replace_masked_values_with_big_negative_number,
)

logger = logging.getLogger(__name__)


@Model.register("Bidaf")
class BidirectionalAttentionFlow(Model):
    """
    This class implements Minjoon Seo's [Bidirectional Attention Flow model]
    (https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d)
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : `int`
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    matrix_attention : `MatrixAttention`
        The attention function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : `Seq2SeqEncoder`
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : `float`, optional (default=`0.2`)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : `bool`, optional (default=`True`)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    regularizer : `RegularizerApplicator`, optional (default=`None`)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        num_highway_layers: int,
        phrase_layer: Seq2SeqEncoder,
        matrix_attention: MatrixAttention,
        modeling_layer: Seq2SeqEncoder,
        span_end_encoder: Seq2SeqEncoder,
        dropout: float = 0.2,
        mask_lstms: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(
            Highway(text_field_embedder.get_output_dim(), num_highway_layers)
        )
        self._phrase_layer = phrase_layer
        self._matrix_attention = matrix_attention
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(
            modeling_layer.get_input_dim(),
            4 * encoding_dim,
            "modeling layer input dim",
            "4 * encoding dim",
        )
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            phrase_layer.get_input_dim(),
            "text field embedder output dim",
            "phrase layer input dim",
        )
        check_dimensions_match(
            span_end_encoder.get_input_dim(),
            4 * encoding_dim + 3 * modeling_dim,
            "span end encoder input dim",
            "4 * encoding dim + 3 * modeling dim",
        )

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(  # type: ignore
        self,
        question: Dict[str, torch.LongTensor],
        passage: Dict[str, torch.LongTensor],
        span_start: torch.IntTensor = None,
        span_end: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        question : `Dict[str, torch.LongTensor]`
            From a ``TextField``.
        passage : `Dict[str, torch.LongTensor]`
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : `torch.IntTensor`, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : `torch.IntTensor`, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : `List[Dict[str, Any]]`, optional
            metadata : `List[Dict[str, Any]]`, optional
            If present, this should contain the question tokens, passage tokens, original passage
            text, and token offsets into the passage for each instance in the batch.  The length
            of this list should be the batch size, and each dictionary should have the keys
            ``question_tokens``, ``passage_tokens``, ``original_passage``, and ``token_offsets``.

        Returns
        -------
        An output dictionary consisting of:

        span_start_logits : `torch.FloatTensor`
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : `torch.FloatTensor`
            The result of ``softmax(span_start_logits)``.
        span_end_logits : `torch.FloatTensor`
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : `torch.FloatTensor`
            The result of ``softmax(span_end_logits)``.
        best_span : `torch.IntTensor`
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        best_span_str : `List[str]`
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question)
        passage_mask = util.get_text_field_mask(passage)
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = replace_masked_values_with_big_negative_number(
            passage_question_similarity, question_mask.unsqueeze(1)
        )
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(
            batch_size, passage_length, encoding_dim
        )

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat(
            [
                encoded_passage,
                passage_question_vectors,
                encoded_passage * passage_question_vectors,
                encoded_passage * tiled_question_passage_vector,
            ],
            dim=-1,
        )

        modeled_passage = self._dropout(
            self._modeling_layer(final_merged_passage, passage_lstm_mask)
        )
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(
            batch_size, passage_length, modeling_dim
        )

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat(
            [
                final_merged_passage,
                modeled_passage,
                tiled_start_representation,
                modeled_passage * tiled_start_representation,
            ],
            dim=-1,
        )
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout(
            self._span_end_encoder(span_end_representation, passage_lstm_mask)
        )
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        # Replace the masked values with a very negative constant.
        span_start_logits = replace_masked_values_with_big_negative_number(
            span_start_logits, passage_mask
        )
        span_end_logits = replace_masked_values_with_big_negative_number(
            span_end_logits, passage_mask
        )
        best_span = get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        # Compute the loss for training.
        if span_start is not None:
            loss = nll_loss(
                util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1)
            )
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(
                util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1)
            )
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.cat([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict["best_span_str"] = []
            question_tokens = []
            passage_tokens = []
            token_offsets = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]["question_tokens"])
                passage_tokens.append(metadata[i]["passage_tokens"])
                token_offsets.append(metadata[i]["token_offsets"])
                passage_str = metadata[i]["original_passage"]
                offsets = metadata[i]["token_offsets"]
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict["best_span_str"].append(best_span_string)
                answer_texts = metadata[i].get("answer_texts", [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens
            output_dict["token_offsets"] = token_offsets
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            "start_acc": self._span_start_accuracy.get_metric(reset),
            "end_acc": self._span_end_accuracy.get_metric(reset),
            "span_acc": self._span_accuracy.get_metric(reset),
            "em": exact_match,
            "f1": f1_score,
        }

    @staticmethod
    def get_best_span(
        span_start_logits: torch.Tensor, span_end_logits: torch.Tensor
    ) -> torch.Tensor:
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = (
            torch.triu(torch.ones((passage_length, passage_length), device=device))
            .log()
            .unsqueeze(0)
        )
        valid_span_log_probs = span_log_probs + span_log_mask

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length
        return torch.stack([span_start_indices, span_end_indices], dim=-1)

    default_predictor = "reading_comprehension"


@DatasetReader.register("Squad")
class SquadReader(DatasetReader):
    """
    !!! Note
        If you're training on SQuAD v1.1 you should use the [`squad1()`](#squad1) classmethod
        to instantiate this reader, and for SQuAD v2.0 you should use the
        [`squad2()`](#squad2) classmethod.

        Also, for transformer-based models you should be using the
        [`TransformerSquadReader`](../transformer_squad#transformersquadreader).

    Dataset reader suitable for JSON-formatted SQuAD-like datasets.
    It will generate `Instances` with the following fields:

      - `question`, a `TextField`,
      - `passage`, another `TextField`,
      - `span_start` and `span_end`, both `IndexFields` into the `passage` `TextField`,
      - and `metadata`, a `MetadataField` that stores the instance's ID, the original passage text,
        gold answer strings, and token offsets into the original passage, accessible as `metadata['id']`,
        `metadata['original_passage']`, `metadata['answer_texts']` and
        `metadata['token_offsets']`, respectively. This is so that we can more easily use the official
        SQuAD evaluation scripts to get metrics.

    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.

    # Parameters

    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the question and the passage.  See :class:`Tokenizer`.
        Default is `SpacyTokenizer()`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    passage_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : `int`, optional (default=`None`)
        If specified, we will cut the question if the length of question exceeds this limit.
    skip_impossible_questions: `bool`, optional (default=`False`)
        If this is true, we will skip examples with questions that don't contain the answer spans.
    no_answer_token: `Optional[str]`, optional (default=`None`)
        A special token to append to each context. If using a SQuAD 2.0-style dataset, this
        should be set, otherwise an exception will be raised if an impossible question is
        encountered.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        no_answer_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        if "skip_invalid_examples" in kwargs:
            import warnings

            warnings.warn(
                "'skip_invalid_examples' is deprecated, please use 'skip_impossible_questions' instead",
                DeprecationWarning,
            )
            skip_impossible_questions = kwargs.pop("skip_invalid_examples")

        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_impossible_questions = skip_impossible_questions
        self.no_answer_token = no_answer_token

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in self.shard_iterable(paragraph_json["qas"]):
                    question_text = question_answer["question"].strip().replace("\n", "")
                    is_impossible = question_answer.get("is_impossible", False)
                    if is_impossible:
                        answer_texts: List[str] = []
                        span_starts: List[int] = []
                        span_ends: List[int] = []
                    else:
                        answer_texts = [answer["text"] for answer in question_answer["answers"]]
                        span_starts = [
                            answer["answer_start"] for answer in question_answer["answers"]
                        ]
                        span_ends = [
                            start + len(answer) for start, answer in zip(span_starts, answer_texts)
                        ]
                    additional_metadata = {"id": question_answer.get("id", None)}
                    instance = self.text_to_instance(
                        question_text,
                        paragraph,
                        is_impossible=is_impossible,
                        char_spans=zip(span_starts, span_ends),
                        answer_texts=answer_texts,
                        passage_tokens=tokenized_paragraph,
                        additional_metadata=additional_metadata,
                    )
                    if instance is not None:
                        yield instance

    def text_to_instance(
        self,  # type: ignore
        question_text: str,
        passage_text: str,
        is_impossible: bool = None,
        char_spans: Iterable[Tuple[int, int]] = None,
        answer_texts: List[str] = None,
        passage_tokens: List[Token] = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Optional[Instance]:
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)

        if self.no_answer_token is not None:
            if self.passage_length_limit is not None:
                passage_tokens = passage_tokens[: self.passage_length_limit - 1]
            passage_tokens = passage_tokens + [
                Token(
                    text=self.no_answer_token,
                    idx=passage_tokens[-1].idx + len(passage_tokens[-1].text) + 1,  # type: ignore
                    lemma_=self.no_answer_token,
                )
            ]
        elif self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]

        question_tokens = self._tokenizer.tokenize(question_text)
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        if is_impossible:
            if self.no_answer_token is None:
                raise ValueError(
                    "This is a SQuAD 2.0 dataset, yet your using a SQuAD reader has 'no_answer_token' "
                    "set to `None`. "
                    "Consider specifying the 'no_answer_token' or using the 'squad2' reader instead, which "
                    f"by default uses '{SQUAD2_NO_ANSWER_TOKEN}' as the 'no_answer_token'."
                )
            answer_texts = [self.no_answer_token]
            token_spans: List[Tuple[int, int]] = [
                (len(passage_tokens) - 1, len(passage_tokens) - 1)
            ]
        else:
            char_spans = char_spans or []
            # We need to convert character indices in `passage_text` to token indices in
            # `passage_tokens`, as the latter is what we'll actually use for supervision.
            token_spans = []
            passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
            for char_span_start, char_span_end in char_spans:
                if char_span_end > passage_offsets[-1][1]:
                    continue
                (span_start, span_end), error = utils.char_span_to_token_span(
                    passage_offsets, (char_span_start, char_span_end)
                )
                if error:
                    logger.debug("Passage: %s", passage_text)
                    logger.debug("Passage tokens (with no-answer): %s", passage_tokens)
                    logger.debug("Question text: %s", question_text)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug(
                        "Tokens in answer: %s",
                        passage_tokens[span_start : span_end + 1],
                    )
                    logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))
            # The original answer is filtered out
            if char_spans and not token_spans:
                if self.skip_impossible_questions:
                    return None
                else:
                    if self.no_answer_token is not None:
                        answer_texts = [self.no_answer_token]
                    token_spans.append(
                        (
                            len(passage_tokens) - 1,
                            len(passage_tokens) - 1,
                        )
                    )
        return utils.make_reading_comprehension_instance(
            question_tokens,
            passage_tokens,
            self._token_indexers,
            passage_text,
            token_spans,
            answer_texts,
            additional_metadata,
        )

    @classmethod
    def squad1(
        cls,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        **kwargs,
    ) -> "SquadReader":
        """
        Gives a `SquadReader` suitable for SQuAD v1.1.
        """
        return cls(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            passage_length_limit=passage_length_limit,
            question_length_limit=question_length_limit,
            skip_impossible_questions=skip_impossible_questions,
            no_answer_token=None,
            **kwargs,
        )

    @classmethod
    def squad2(
        cls,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        passage_length_limit: int = None,
        question_length_limit: int = None,
        skip_impossible_questions: bool = False,
        no_answer_token: str = SQUAD2_NO_ANSWER_TOKEN,
        **kwargs,
    ) -> "SquadReader":
        """
        Gives a `SquadReader` suitable for SQuAD v2.0.
        """
        return cls(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            passage_length_limit=passage_length_limit,
            question_length_limit=question_length_limit,
            skip_impossible_questions=skip_impossible_questions,
            no_answer_token=no_answer_token,
            **kwargs,
        )


DatasetReader.register("Squad1", constructor="squad1")(SquadReader)
DatasetReader.register("Squad2", constructor="squad2")(SquadReader)


class modelload():
    def loadmodelfrompath(path: str):
        serialization_dir = path
        config_file = os.path.join(serialization_dir, "config.json")
        vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
        weights_file = os.path.join(serialization_dir, "weights.th")
        loaded_params = Params.from_file(config_file)
        loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
        #loaded_vocab = loaded_model.vocab  # Vocabulary is loaded in Model.load()
        return loaded_model #, loaded_vocab

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

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        # For BiDAF
        if "best_span" in outputs:
            span_start_label = outputs["best_span"][0]
            span_end_label = outputs["best_span"][1]
            passage_field: SequenceField = new_instance["passage"]  # type: ignore
            new_instance.add_field(
                "span_start", IndexField(int(span_start_label), passage_field), self._model.vocab
            )
            new_instance.add_field(
                "span_end", IndexField(int(span_end_label), passage_field), self._model.vocab
            )

        # For NAQANet model. It has the fields: answer_as_passage_spans, answer_as_question_spans,
        # answer_as_add_sub_expressions, answer_as_counts. We need labels for all.
        elif "answer" in outputs:
            answer_type = outputs["answer"]["answer_type"]

            # When the problem is a counting problem
            if answer_type == "count":
                field = ListField([LabelField(int(outputs["answer"]["count"]), skip_indexing=True)])
                new_instance.add_field("answer_as_counts", field, self._model.vocab)

            # When the answer is in the passage
            elif answer_type == "passage_span":
                # TODO(mattg): Currently we only handle one predicted span.
                span = outputs["answer"]["spans"][0]

                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                offsets = new_instance["metadata"].metadata["passage_token_offsets"]  # type: ignore
                for index, offset in enumerate(offsets):
                    if offset[0] == span[0]:
                        word_span_start = index
                    if offset[1] == span[1]:
                        word_span_end = index

                passage_field: SequenceField = new_instance["passage"]  # type: ignore
                field = ListField([SpanField(word_span_start, word_span_end, passage_field)])
                new_instance.add_field("answer_as_passage_spans", field, self._model.vocab)

            # When the answer is an arithmetic calculation
            elif answer_type == "arithmetic":
                # The different numbers in the passage that the model encounters
                sequence_labels = outputs["answer"]["numbers"]

                numbers_field: ListField = instance["number_indices"]  # type: ignore

                # The numbers in the passage are given signs, that's what we are labeling here.
                # Negative signs are given the class label 2 (for 0 and 1, the sign matches the
                # label).
                labels = []
                for label in sequence_labels:
                    if label["sign"] == -1:
                        labels.append(2)
                    else:
                        labels.append(label["sign"])
                # There's a dummy number added in the dataset reader to handle passages with no
                # numbers; it has a label of 0 (not included).
                labels.append(0)

                field = ListField([SequenceLabelField(labels, numbers_field)])
                new_instance.add_field("answer_as_add_sub_expressions", field, self._model.vocab)

            # When the answer is in the question
            elif answer_type == "question_span":
                span = outputs["answer"]["spans"][0]

                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                question_offsets = new_instance["metadata"].metadata[  # type: ignore
                    "question_token_offsets"
                ]
                for index, offset in enumerate(question_offsets):
                    if offset[0] == span[0]:
                        word_span_start = index
                    if offset[1] == span[1]:
                        word_span_end = index

                question_field: SequenceField = new_instance["question"]  # type: ignore
                field = ListField([SpanField(word_span_start, word_span_end, question_field)])
                new_instance.add_field("answer_as_question_spans", field, self._model.vocab)

        return [new_instance]


model = modelload.loadmodelfrompath('/home/kxt/project/IS707/BiDAF/model')
SQ = SquadReader.squad1
BF = BiDAF_predictor(model, SQ)
output = BF.predict("How old is mike?", "mike is 18 years old.")
print(output["best_span_str"])


