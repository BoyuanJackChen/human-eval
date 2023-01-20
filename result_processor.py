import json

def read_jsonl_into_dict(source_file, target_file):
    with open(target_file, 'r') as f1, open(source_file, 'r') as f2:
        for line1 in f1.readlines():
            target_dict = json.loads(line1)
            line2 = f2.readline()
            source_dict = json.loads(line2)
            completed_text = source_dict['completion']
            target_dict['completion']['choices'][0]['text'] = completed_text
            target_dict['completion']['choices'][1]['text'] = completed_text
            target_dict['completion']['choices'][2]['text'] = completed_text
            target_dict['completion']['choices'][3]['text'] = completed_text
            with open('new_2B.jsonl', 'a') as new_file:
                json.dump(target_dict, new_file)
                new_file.write('\n')

    # with open(file_path, 'r') as f:
    #     # Read each line
    #     for line in f.readlines():
    #         # Load the line into a dictionary
    #         line_dict = json.loads(line)
    #
    #         completed_text = line_dict['completion']
    #         choice = {'text': completed_text, 'index': 0, 'finish_reason': 'stop', 'logprobs': []}
    #         choices = [choice]
    #         new_line = {'choices': choices}
    #         # print(line_dict['completion'].keys())
    #         # print(line_dict['completion']['choices'][0].keys())
    #         # print(line_dict['completion']['choices'][1].keys())
    #         # print(line_dict['completion']['choices'][0]['text'])
    #         # print(line_dict['completion']['choices'][0]['index'])
    #         # print(line_dict['completion']['choices'][1]['text'])
    #         # print(line_dict['completion']['choices'][1]['index'])
    #         # print(line_dict['completion']['choices'][0]['finish_reason'])
    #         # print(line_dict['completion']['choices'][1]['finish_reason'])
    #         # input()
    #
    #         with open('new_2B.jsonl', 'a') as new_file:
    #             json.dump(line_dict, new_file)
    #             new_file.write('\n')

# raw_2B = read_jsonl_into_dict("./samples_4_0.7_transformerscodegen-2B-mono.jsonl")
read_jsonl_into_dict("./samples_4_0.7_transformerscodegen-2B-mono.jsonl", "./samples_4_0.5.jsonl")