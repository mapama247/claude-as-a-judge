import os
import json
import argparse
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import InferenceClient

# USAGE: python generate_responses.py -i ./data/toy_prompts.txt -m google/gemma-2b google/gemma-2b-it

def parse_args(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_data", type=str, required=True, help="Path to txt/jsonl file with prompts (one per line).")
    parser.add_argument("-m", "--models", required=True, nargs="+", help="List of LLMs (HF ids or local paths).")
    parser.add_argument("-o", "--output_dir", default="./output", type=str, required=False, help="Directory were generated files will be stored.")
    parser.add_argument("-l", "--limit", default=None, type=int, required=False, help="Limits the number of samples for fast debugging.")
    parser.add_argument("-r", "--repetition_penalty", default=1.0, type=float, required=False, help="HF generation argument.")
    parser.add_argument("-t", "--temperature", default=0.7, type=float, required=False, help="HF generation argument.")
    parser.add_argument("-p", "--top_p", default=0.95, type=float, required=False, help="HF generation argument.")
    parser.add_argument("-n", "--max_new_tokens", default=1024, type=int, required=False, help="HF generation argument.")
    parser.add_argument("-c", "--use_inference_client", default=False, type=bool, required=False, help="Whether to use HF's Inference client.")
    args = parser.parse_args()
    if verbose: print(args)
    return args

def read_txt_file(filepath: str) -> List[str]:
    # return [line.rstrip() for line in open(filepath)]
    with open(filepath) as f:
        lines = [line.rstrip() for line in f]
        return lines

def read_jsonl_file(filepath: str, field: str = "prompt") -> List[dict]:
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(json.loads(line)[field])
    return lines

def write_jsonl_file(filepath: str, data: List[dict]) -> None:
    with open(filepath, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")
    print(f"New JSONL file saved: {filepath}")


def write_json_file(filepath: str, data: dict) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"New JSON file saved: {filepath}")


def format_prompt(model_id, message):
    # wip (check official chat templates)
    if "mistral" in model_id:
        return "<s>" + f"[INST] {message} [/INST]"
    elif "gemma" in model_id:
        return f"User: {message}\n\nAssistant: "
    else:
        return message


def main():
    args = parse_args()

    if args.input_data.endswith(".txt"):
        prompts = read_txt_file(args.input_data)
    elif args.input_data.endswith(".jsonl"):
        prompts = read_jsonl_file(args.input_data)
    else:
        raise NotImplementedError("The input file must be either .txt or .jsonl, other formats are not supported.")

    output_dir = os.path.join(args.output_dir, args.input_data.split("/")[-1].split(".")[0])
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    generate_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
    )

    all_responses = {}
    for model_id in args.models:
        responses = []
        print(f"### {model_id.upper()} ###")
        if args.use_inference_client:
            client = InferenceClient(model_id)
        else:
            # model = AutoModel.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        for idx,prompt in enumerate(tqdm(prompts)):
            # formatted_prompt = format_prompt(model_id, prompt)
            if args.use_inference_client:
                try:
                    output = client.text_generation(prompt, **generate_kwargs, details=True, return_full_text=False)
                    response = output.generated_text
                except Exception as e:
                    print(e)
                    continue
            else:
                if tokenizer.chat_template is None:
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda") # add padding=True to process in batches
                else:
                    # TODO: use the chat template if available
                    # formatted_prompt = tokenizer.decode(tokenizer.apply_chat_template([{"role": "user", "content": prompt}]))
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")  # add padding=True to process in batches
                outputs = model.generate(**inputs, **generate_kwargs)
                outputs = outputs[:, inputs.input_ids.shape[1]:]
                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] # remove the final [0] if you're doing batches

            # responses.append({"id":idx, "prompt":prompt, "response":response})
            responses.append({"prompt":prompt, "response":response})
            if idx not in all_responses:
                all_responses[idx] = {
                    "prompt": prompt,
                    "responses": [{"model_id": model_id, "response": response}],
                }
            else:
                all_responses[idx]["responses"].append({"model_id": model_id, "response": response})

        if not args.use_inference_client:
            del model, tokenizer

        write_jsonl_file(f"{output_dir}/responses_{model_id.split('/')[-1]}.jsonl", responses)
    write_json_file(f"{output_dir}/responses_all.json", all_responses)

if __name__=="__main__":
    main()