from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loaded_model', type=str, default='Salesforce/codegen-350M-mono')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_samples_per_task', type=int, default=3)
FLAGS = parser.parse_args()

eos_token = 50256
stop_words = ["\n\n"]
problems = read_problems()
beam_width = 4
num_beam_groups = 4
beam_diversity_rate = 0.7


def trim_with_stopwords(output, stopwords, original_prompt) -> str:
    for j in range(len(output)):
        len_prompt = len(original_prompt)
        answer = output[j][len_prompt:]
        for w in sorted(stopwords, reverse=True):
            for i in range(len(output[j])):
                if answer[i:].startswith(w):
                    return answer[:i]
    return output


def main(args):
    loaded = args.loaded_model
    device = torch.device(args.device)
    num_samples_per_task = args.num_samples_per_task
    tokenizer = AutoTokenizer.from_pretrained(loaded)
    model = AutoModelForCausalLM.from_pretrained(loaded)
    model.to(device)
    # samples = [
    #     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    #     # for task_id in select_ids
    #     for task_id in problems
    #     for _ in range(num_samples_per_task)
    # ]
    def generate_one_completion(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(
            input_ids.to(device),
            max_new_tokens=200,
            eos_token_id=eos_token,
            pad_token_id=eos_token,
            num_beams=beam_width,
            num_beam_groups=num_beam_groups,
            diversity_penalty=beam_diversity_rate
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        trimmed_text = trim_with_stopwords(generated_text, stop_words, prompt)
        return trimmed_text

    print("samples started")
    samples = []
    for task_id in problems:
        print(f"Generating task {task_id}")
        for _ in range(num_samples_per_task):
            sample = dict(
                task_id=task_id,
                completion=generate_one_completion(problems[task_id]["prompt"])
            )
            samples.append(sample)
    write_jsonl(f"samples_{beam_width}_{beam_diversity_rate}_transformers{loaded}.jsonl", samples)


if __name__== "__main__":
    main(FLAGS)