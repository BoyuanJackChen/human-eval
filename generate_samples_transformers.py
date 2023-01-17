from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CodeGenModel

eof_token = 50256
stop_words = ["\n\n"]
problems = read_problems()
beam_width = 4
num_beam_groups = 4
beam_diversity_rate = 0.7

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def trim_with_stopwords(output, stopwords) -> str:
    for j in range(len(output)):
        for w in sorted(stopwords, reverse=True):
            for i in range(len(output[j])):
                if output[j][i:].startswith(w):
                    return output[j][:i]
    return output

def generate_one_completion(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        num_beams=beam_width,
        num_beam_groups= num_beam_groups,
        diversity_penalty=beam_diversity_rate
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    trimmed_text = trim_with_stopwords(generated_text, stop_words)
    return trimmed_text
    # result = openai.Completion.create(
    #     engine='codegen',
    #     prompt=prompt,
    #     max_tokens=200,
    #     logprobs=1,
    #     stop=["\n\n"],
    #     beam_width=beam_width,
    #     beam_search_diversity_rate=beam_diversity_rate
    # )
    # return result

if __name__== "__main__":
    num_samples_per_task = 3
    print("samples started")
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        # for task_id in select_ids
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(f"samples_{beam_width}_{beam_diversity_rate}_transformers.jsonl", samples)