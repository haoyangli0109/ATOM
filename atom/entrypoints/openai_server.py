import argparse
import asyncio
import time
from typing import List, Optional, Dict, Any, Union
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]  # Support both single string and list of strings
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# Global engine
engine = None
tokenizer: Optional[AutoTokenizer] = None
model_name: str = ""


async def generate_async(prompt: str, sampling_params: SamplingParams) -> Dict[str, Any]:
    """Async wrapper for engine generation."""
    global engine
    
    # Use the new async generate method that handles each request independently
    async for output in engine.generate_async(prompt, sampling_params):
        return output


app = FastAPI(title="Atom OpenAI API Server")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    global engine, tokenizer, model_name
    
    try:
        # Convert messages to prompt using chat template
        prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )
        
        # Generate response
        output = await generate_async(prompt, sampling_params)
        
        # Format response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["text"],
                    },
                    "finish_reason": output["finish_reason"],
                }
            ],
            usage={
                "prompt_tokens": output["num_tokens_input"],
                "completion_tokens": output["num_tokens_output"],
                "total_tokens": output["num_tokens_input"] + output["num_tokens_output"],
            },
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Handle text completion requests."""
    global engine, model_name
    
    try:
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_strings=request.stop,
            ignore_eos=request.ignore_eos,
        )
        
        # Handle both single prompt and batch prompts
        if isinstance(request.prompt, str):
            # Single prompt
            output = await generate_async(request.prompt, sampling_params)
            choices = [
                {
                    "index": 0,
                    "text": output["text"],
                    "finish_reason": output["finish_reason"],
                }
            ]
            total_prompt_tokens = output["num_tokens_input"]
            total_completion_tokens = output["num_tokens_output"]
        else:
            # Batch prompts - process concurrently
            tasks = [
                generate_async(prompt, sampling_params)
                for prompt in request.prompt
            ]
            outputs = await asyncio.gather(*tasks)
            
            choices = [
                {
                    "index": i,
                    "text": output["text"],
                    "finish_reason": output["finish_reason"],
                }
                for i, output in enumerate(outputs)
            ]
            total_prompt_tokens = sum(output["num_tokens_input"] for output in outputs)
            total_completion_tokens = sum(output["num_tokens_output"] for output in outputs)
        
        # Format response
        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    global model_name
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "atom",
            }
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    global engine, tokenizer, model_name
    
    parser = argparse.ArgumentParser(description="Atom OpenAI API Server")
    
    # Add engine arguments
    EngineArgs.add_cli_args(parser)
    
    # Add server-specific arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--server-port", type=int, default=8000, 
        help="Server port (note: --port is used for internal engine communication)"
    )
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_name = args.model
    
    # Initialize engine
    print(f"Initializing engine with model {args.model}...")
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine()
    
    print(f"Starting server on {args.host}:{args.server_port}...")
    uvicorn.run(app, host=args.host, port=args.server_port)


if __name__ == "__main__":
    main()
