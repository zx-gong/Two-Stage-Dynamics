from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import pickle

llm_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(llm_name)

parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPTJ LLM on CounterFact')
parser.add_argument('--path', type=str, default="./hotpot", help='place where to save the dataset')
args = parser.parse_args()

full_dataset = load_dataset("hotpot_qa", "fullwiki")
num_val = len(full_dataset["validation"])

# As the hotpot QA does not have answers for test set, we use the train set
train = []
ctr = 0
for dp in full_dataset["train"]:
    if ctr >= num_val:
        break
    ctr += 1
    question = dp["question"].strip()
    answer = dp["answer"].strip()
    num_tokens = len(tokenizer(answer).input_ids)
    if num_tokens <= 15:
        train.append({"question": question,
                        "answer": answer})

validation = []
for dp in full_dataset["validation"]:
    question = dp["question"].strip()
    answer = dp["answer"].strip()
    num_tokens = len(tokenizer(answer).input_ids)
    if num_tokens <= 15:
        validation.append({"question": question,
                            "answer": answer})

dataset = train + validation
num_dp = len(dataset)

with open(args.path, "wb") as f:
    pickle.dump(dataset, f)