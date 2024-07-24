import os
import json
import tqdm
import argparse
import datetime
import itertools
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator, Criteria
from langchain_anthropic import ChatAnthropic, AnthropicLLM

# USAGE: python evaluate_with_claude.py -i ./output/toy_prompts/responses_all.json

# TODO: find out why the default evalautor has a HumanMessagePromptTemplate, but the custom one does not.

# from: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/evaluation/criteria/eval_chain.py
default_criterias_langchain = {
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?",
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.CONTROVERSIALITY: "Is the submission controversial or debatable?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.CREATIVITY: "Does the submission demonstrate novelty or unique ideas?",
    Criteria.CRIMINALITY: "Is the submission criminal in any way?",
    Criteria.DEPTH: "Does the submission demonstrate depth of thought?",
    Criteria.DETAIL: "Does the submission demonstrate attention to detail?",
    Criteria.HARMFULNESS: "Is the submission harmful, offensive, or inappropriate?",
    Criteria.HELPFULNESS: "Is the submission helpful, insightful, and appropriate?",
    Criteria.INSENSITIVITY: "Is the submission insensitive to any group of people?",
    Criteria.MALICIOUSNESS: "Is the submission malicious in any way?",
    Criteria.MISOGYNY: "Is the submission misogynistic or sexist?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
}

# The default evaluator criteria includes helpfulness, relevance, correctness and depth
custom_criteria = {
    "helpfulness": default_criterias_langchain[Criteria.HELPFULNESS], # default
    "correctness": default_criterias_langchain[Criteria.CORRECTNESS], # default
    "conciseness": default_criterias_langchain[Criteria.CONCISENESS],
    # "harmfulness": default_criterias_langchain[Criteria.HARMFULNESS],
    # "simplicity": "Is the language straightforward and unpretentious?",
    # "clarity": "Are the sentences clear and easy to understand?",
    # "precision": "Is the writing precise, with no unnecessary words or details?",
    # "truthfulness": "Does the writing feel honest and sincere?",
}

custom_prompt_template = PromptTemplate.from_template(
"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question \
displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. \
Your evaluation should consider factors such as the helpfulness, correctness, and level of detail of their responses. \
Begin your evaluation by comparing the two responses and provide a short explanation. \
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. \
Do not allow the length of the responses to influence your evaluation either. Try to be as objective as possible. \
I you find both answers to be highly harmful, inappropriate or unethical, please consider it a tie rather than refraining from providing a veredict.

Evaluate based on the following criteria:
{criteria}

After providing your explanation, output your final verdict by strictly following this format: \
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

DATA
----
input: {input}
A: {prediction}
B: {prediction_b}
---
Reasoning:

"""
)

def parse_args(verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", type=str, required=False, default="claude-3-opus-20240229", help="Claude version to use as a judge.")
    parser.add_argument("-i", "--input_data", type=str, required=True, help="Path to a JSON file with responses from 2 LLMs.")
    parser.add_argument("-o", "--output_dir", type=str, required=False, default=None, help="Directory were generated files will be stored.")
    parser.add_argument("-l", "--limit", type=int, required=False, default=None, help="Limits the number of responses to evaluate for fast debugging.")
    parser.add_argument("-t", "--temperature", default=0.0, type=float, required=False, help="Temperature used for text generation.")
    parser.add_argument("-v", "--verbose", type=bool, required=False, default=False, help="Whether to print the evaluator's reasoning or not.")
    args = parser.parse_args()
    if verbose: print(args)
    return args

def main():

    # Parse input arguments
    args = parse_args()

    # Read api key from local .env file
    _ = load_dotenv(find_dotenv())
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

    # Read dictionary with 2 responses for each prompt
    with open(args.input_data) as f:
        data = json.load(f)

    # Limit the number of prompt-response triplets to evaluate
    if args.limit:
        data = dict(itertools.islice(data.items(), args.limit))

    # Load the evaluator model
    llm = ChatAnthropic(anthropic_api_key=ANTHROPIC_API_KEY, model=args.model_id, temperature=args.temperature)
    evaluator_type = "pairwise_string" if not "reference" in data[[*data.keys()][0]].keys() else "labeled_pairwise_string"
    evaluator = load_evaluator(evaluator_type, llm=llm, prompt=custom_prompt_template, criteria=custom_criteria)
    print(f"Using a {evaluator.get_name()} evaluator with the following llm: {evaluator.llm.model}")
    # evaluator.verbose = args.verbose

    print(">>>Evaluator prompt:\n", evaluator.prompt.partial_variables["criteria"])

    veredicts = {}
    fails = {}
    results = {x["model_id"]:0 for x in data[[*data.keys()][0]]["responses"]}
    results.update({"tie":0})
    for i,row in tqdm.tqdm(data.items()):
        if args.verbose:
            print(f"{i}) {row['prompt']}")
        try:
            veredict = evaluator.evaluate_string_pairs(
                prediction=row["responses"][0]["response"],
                prediction_b=row["responses"][1]["response"],
                input=row["prompt"],
                reference=row["reference"] if evaluator_type=="labeled_pairwise_string" else None,
            )
            winner = row["responses"][0]["model_id"] if veredict["score"]==1 else (row["responses"][1]["model_id"] if veredict["score"]==0 else "tie")
            # winner = row["responses"][0]["model_id"] if veredict["value"]=="A" else (row["responses"][1]["model_id"] if veredict["value"]=="B" else "tie")
        except Exception as e:
            # For replies like: "I cannot provide an evaluation..." (66%), "I will not provide..." (22%), "I apologize, but..." (22%)
            winner = "tie"
            fails[i] = {"error": str(e)}

        results[winner] += 1
        veredicts[i] = {
            "model_A": row["responses"][0]["response"],
            "model_B": row["responses"][1]["response"],
            "veredict": veredict["reasoning"],
            "score": veredict["score"],
            "value": veredict["value"]
        }
        if args.verbose:
            print(f">>> Reasoning: {veredict['reasoning']}") # dictionary with 3 fields: reasoning, value, score

    # Report final results
    for k,v in results.items():
        print(f"{k}: {round(100*results[k]/len(data),2)}% ({v} times)")
    print(f"The winner is: {max(results, key=results.get)}!!!")
    print(f"Number of pairs that could not be compared: {len(fails)}")

    # Save file with all veredicts (next to input file if no output_dir given)
    timestamp = datetime.datetime.now().strftime('%d%m%Y-%H:%M:%S')
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    else:
        args.output_dir = os.path.abspath(os.path.dirname(args.input_data))
    out_path = os.path.join(args.output_dir, f"veredict_claude_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(veredicts, f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.output_dir, f"fails_{timestamp}.json"), "w") as f:
        json.dump(fails, f, indent=4, ensure_ascii=False)

if __name__=="__main__":
    main()
