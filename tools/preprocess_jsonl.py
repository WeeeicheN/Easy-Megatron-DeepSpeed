import json

jsonl_input_path = "/home/dwc/data/GSM8k.jsonl"
jsonl_output_path = "/home/dwc/data/gsm8k.jsonl"

with open(jsonl_output_path, 'w') as out_file:
    with open(jsonl_input_path, 'r') as in_file:
        for line in in_file:
            data = json.loads(line)

            # prompt = data['prompt'].replace("<|user|>\n", "").replace("<|assistant|>\n", "")
            # completion = data['completion']

            # out_file.write(prompt.strip() + "\n" + completion.strip() + "\n" + "<|eot_id|>" + "\n\n")  # 添加换行和空行分隔每个样本
            
            question = data['question']
            answer = data['answer']

            data = {'text': question.strip() + "\n" +answer.strip()}

            json.dump(data, out_file)
            out_file.write('\n')
            
print("转换完成，结果已保存至:", jsonl_output_path)