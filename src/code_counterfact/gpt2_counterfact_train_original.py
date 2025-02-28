import os
import time
import torch
import pickle
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress
import pdb
import random

class GPT2Experiment:

    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, model, tokenizer, dataset, args, llm_name):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        model_edit = LaserWrapper.get_edited_model(model=model,
                                                   lname=args.lname,
                                                   lnum=args.lnum,
                                                   rate=args.rate,
                                                   intervention=args.intervention,
                                                   logger=logger,
                                                   in_place=True)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")
        
        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 300  # 定义训练的epoch数量
        
        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        for epoch in range(num_epochs+1):
            epoch = epoch+1
            train_loss = 0.0  # 每个epoch重置总损失
            predictions = []  # 每个epoch重置预测结果
            is_correct_num = 0
            model.train()
            self.logger.log(f"Epoch {epoch}/{num_epochs}")

            num_batches = dataset_size // args.batch_size
            if dataset_size % args.batch_size != 0:
                num_batches += 1  
            for batch_idx in tqdm(range(num_batches)):
                batch_start = batch_idx * args.batch_size
                batch_end = min(dataset_size, (batch_idx + 1) * args.batch_size)

                questions = []
                answers = []
                for i in range(batch_start, batch_end):
                    question, answer = dataset[i]
                    questions.append(question)
                    answers.append(answer)
                
                # 使用批量处理的方式进行tokenize
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token       
                inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(self.device)
                gold_answer_token_ids = [tokenizer(answer)["input_ids"] for answer in answers]
                
                # 检查每个答案只有1个token
                for answer_ids in gold_answer_token_ids:
                    assert len(answer_ids) == 1,  f"For GPTJ+CounterFact special case, we assume the answer has 1 token. Found {answer_ids}."

                # pdb.set_trace()
                gold_answer_token_ids = torch.tensor([ids[0] for ids in gold_answer_token_ids]).to(self.device)

                optimizer.zero_grad()
                # 前向传播
                results = model_edit(inputs.input_ids)
                logits = results.logits  # shape: (batch_size, sequence_length, vocab_size)
                last_token_logits = logits[:, -1, :]
                
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # shape: (batch_size, sequence_length, vocab_size)
                last_token_logprobs = log_probs[:, -1, :]  # shape: (batch_size, vocab_size), 最后一个token的归一化log概率

                # 计算损失
                loss = criterion(last_token_logits, gold_answer_token_ids)
                loss.backward()
                optimizer.step()

                total_loss = loss.item()
                train_loss += total_loss

                # 计算准确率和其他指标
                sorted_logprobs, sorted_indices = torch.sort(last_token_logprobs, descending=True)
                top_k_logprobs = sorted_logprobs[:, :10].detach().cpu().numpy()  # 取前10个概率
                top_k_indices = sorted_indices[:, :10].detach()  # 取前10个索引
          
                for batch_i in range(batch_end - batch_start):
                    top_k_tokens = tokenizer.batch_decode(top_k_indices[batch_i])
                    assert len(top_k_tokens) == 10
                    is_correct = answers[batch_i].lower().strip() == top_k_tokens[0].lower().strip()
                    top_1_acc = float(is_correct)
                    top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                    top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])
                    # 计算问题和答案的总log概率
                    selected_log_prob = log_probs[batch_i, :-1, :]  # 去掉最后一个token
                    indices = inputs.input_ids[batch_i, 1:].unsqueeze(1)  # question - 1 x 1
                    selected_log_prob = torch.gather(selected_log_prob, index=indices, dim=1)  # 选择问题的log概率
                    question_log_prob = selected_log_prob.sum().item()
                    answer_log_prob = last_token_logprobs[batch_i, gold_answer_token_ids[batch_i]].item()
                    total_log_prob = question_log_prob + answer_log_prob

                    logprob_results = ContextAnswerLogProb(
                        total_log_prob=total_log_prob,
                        answer_log_prob=answer_log_prob,
                        answer_len=1
                    )

                    self.dataset_metric.accept(is_correct=is_correct,
                                            f1pr_score=None,
                                            log_prob_results=logprob_results,
                                            top_k_acc={1: top_1_acc})

                    predictions_ = {
                        "ix": batch_start + batch_i,
                        "question": questions[batch_i],
                        "gold-answer": answers[batch_i],
                        "generation": top_k_tokens[0],  # 我们将top1视为1步生成
                        "correct": is_correct
                    }
                    if is_correct == True: is_correct_num = is_correct_num + 1
                    predictions.append(predictions_)

            self.logger.log(f"Epoch {epoch}/{num_epochs} correct num: {is_correct_num}/{dataset_size}")
            if epoch <= 21:
                for i in range(dataset_size):
                    self.logger.log(f"Epoch {epoch}: {i}/{dataset_size}: {predictions[i]}")
            elif epoch > 21 and epoch <= 94:
                random_indices = random.sample(range(dataset_size), 30)
                for i in random_indices:
                    self.logger.log(f"Epoch {epoch}: {i}/{dataset_size}: {predictions[i]}")
            elif epoch > 94 and epoch <= 100:
                for i in range(dataset_size):
                    self.logger.log(f"Epoch {epoch}: {i}/{dataset_size}: {predictions[i]}")
            elif epoch > 100 and epoch % 20 == 0:
                random_indices = random.sample(range(dataset_size), 30)
                for i in random_indices:
                    self.logger.log(f"Epoch {epoch}: {i}/{dataset_size}: {predictions[i]}")
                    
            # 保存模型权重
            save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
            if epoch <= 100 and epoch % 10 == 0:
                torch.save(model.state_dict(), f"{save_dir}/gpt2_counterfact_epoch_{epoch}.pt")
            elif epoch > 100 and epoch <= 300 and epoch % 20 == 0:
                torch.save(model.state_dict(), f"{save_dir}/gpt2_counterfact_epoch_{epoch}.pt")
                
            # 计算并打印该epoch的平均损失
            avg_train_loss = train_loss / dataset_size
            print(f"Epoch {epoch} completed. Average training loss: {avg_train_loss}")
            self.logger.log(f"Epoch {epoch} completed. Average training loss: {avg_train_loss}")
            
        save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
        torch.save(model.state_dict(), f"{save_dir}/gpt2_counterfact_epoch_{epoch}.pt")


    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        time_start = time.time()
        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"

        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v

        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with GPT2 LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont',
                                 "all", "mlp", "attn"],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=-1, help='Layers to edit', choices=list(range(-1, 28)))
    parser.add_argument('--model_path',
                        type=str,
                        default="/home/zxgong/laser-main/data/Llama2/Llama-2-7b-hf",
                        help="Place where model weights are stored")
    parser.add_argument('--home_dir', type=str,
                        default="/home/zxgong/laser-main/data/iclr/counterfact/gpt2_results",
                        help='Directory where the data is')
    parser.add_argument('--dataset_file', type=str,
                        default="/home/zxgong/laser-main/data/counterfact",
                        help='Directory where the data is')

    args = parser.parse_args()

    # Step 2: Load model and tokenizer
    llm_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Step 3: Create save directory and logger
    home_dir = args.home_dir
    dataset_loc = args.dataset_file
    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")
    
    # Step 4: Create an experiment
    experiment = GPT2Experiment(save_dir=save_dir, logger=logger)
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)

    # Step 5: Read the dataset
    with open(args.dataset_file, "rb") as f:
        data = pickle.load(f)
    num_dp = len(data)
    dataset = []
    for i in range(num_dp):
        question = data[i]["question"]
        answer = data[i]["gold-answer"]
        assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
        dataset.append((question, answer))
    logger.log(f"Read dataset of size {num_dp}")

    # Step 6: Run intervention
    experiment.train(model=model,
                         tokenizer=tokenizer,
                         dataset=dataset,
                         args=args,
                         llm_name=llm_name)

    logger.log("Experimented Completed.")