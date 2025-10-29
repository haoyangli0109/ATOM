import argparse

import torch
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

# Add engine arguments
EngineArgs.add_cli_args(parser)

# Add example-specific arguments
parser.add_argument(
    "--temperature", type=float, default=0.6, help="temperature for sampling"
)


def main():
    args = parser.parse_args()
    
    # Create engine from args
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        "如何在一个月内增肌10公斤",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    print("This is prompts:", prompts)

    # # Calculate total token number of prompts
    # prompt_token_ids = [llm.tokenizer.encode(prompt) for prompt in prompts]
    # prompt_lens = [len(token_ids) for token_ids in prompt_token_ids]
    # # Create warmup inputs with same shapes as all prompts
    # # Generate random token IDs for each prompt length to match expected input shapes
    # warmup_prompts = []
    # for i, prompt_len in enumerate(prompt_lens):
    #     warmup_prompt = torch.randint(
    #         0, llm.tokenizer.vocab_size, size=(prompt_len,)
    #     ).tolist()
    #     warmup_prompts.append(warmup_prompt)
    # # Run warmup with the same batch structure as the actual prompts (no profiling)
    # _ = llm.generate(warmup_prompts, sampling_params)

    # generate (with profiling)
    # outputs_ = llm.generate(["Benchmark: "], SamplingParams())
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
