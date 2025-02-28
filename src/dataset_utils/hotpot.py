from datasets import load_dataset, load_from_disk
from dataset_utils.abstract_dataset import AbstractDataset
from transformers import AutoTokenizer


class Hotpot(AbstractDataset):

    def __init__(self):
        super(Hotpot, self).__init__()
        llm_name = 'gpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def get_dataset(self, logger):

        # full_dataset = load_dataset("hotpot_qa", "fullwiki")
        full_dataset = load_from_disk("/home/zxgong/laser-main/data/hotpot_qa_one_word")
        num_val = len(full_dataset["validation"])

        # As the hotpot QA does not have answers for test set, we use the train set
        train = []
        # ctr = 0
        for dp in full_dataset["train"]:
            # if ctr >= num_val:
            #     break
            # ctr += 1
            question = dp["question"].strip()
            answer = dp["answer"].strip()
            num_tokens = len(self.tokenizer(answer).input_ids)
            # if num_tokens <= 15:
            train.append({"question": question,
                              "answer": answer})

        validation = []
        for dp in full_dataset["validation"]:
            question = dp["question"].strip()
            answer = dp["answer"].strip()
            num_tokens = len(self.tokenizer(answer).input_ids)
            # if num_tokens <= 15:
            validation.append({"question": question,
                                   "answer": answer})

        dataset = train + validation
        num_dp = len(dataset)
        logger.log(f"Read dataset of size {num_dp} of which the first {len(train)} examples are from the "
                   f"train set and the remaining {len(validation)} from the validation split.")

        
        return dataset
