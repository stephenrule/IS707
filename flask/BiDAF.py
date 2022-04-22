import numpy
from allennlp.data.fields import LabelField, TextField, ListField, SpanField, SequenceLabelField, IndexField
from allennlp.models.archival import archive_model, load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from typing import Any, Dict, List, Tuple, Optional, Iterable
from allennlp.data.instance import Instance
from allennlp_models.rc.dataset_readers import utils


@Predictor.register("Reading_comprehension")
class ReadingComprehension_Predictor(Predictor):
    """
    Predictor for the :class:`~allennlp_rc.models.bidaf.BidirectionalAttentionFlow` model, and any
    other model that takes a question and passage as input.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

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


archive = load_archive('../BiDAF/model.tar.gz')
BF = ReadingComprehension_Predictor(archive.model, archive.dataset_reader)

# Test. First is the Question and second is the passage. 
#output = BF.predict("How old is mike?", "mike is 18 years old.")
#print(output["best_span_str"])


#### Storage Facility Work
def chatbot_ai(question):
    passage = ("UMBC Storage is located at 1234 Hilltop Cir, Baltimore, MD, 21250. "  
          "We are open 7 days a week from 9:00 AM to 5:00 PM Monday through Friday and open 9:00 AM to 8:00 PM on Saturday and Sunday. "
          "Small storage unit is 5ft by 5ft with a total area of 25 square feet and cost $50. "
          "Medium storage unit is 10ft by 10ft with a total area of 100 square feet and cost $75. "
          "Large storage unit is 10ft by 20ft with a total area of 200 square feet and cost $95.")

    output = BF.predict(question, passage)
    return(output["best_span_str"])

# Test1 - Works. Gave answer of $50
#q1 = "What is the cost of a small storage unit?"
#test1 = chatbot_ai(q1)
#print(test1)
