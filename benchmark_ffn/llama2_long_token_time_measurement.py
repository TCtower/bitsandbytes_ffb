import argparse
import datetime
import logging
import os
import torch

import bitsandbytes as bnb
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

# configure logging, dump to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# generate a folder for the logs and results
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
logdir = os.path.join("logs_long_generate", timestamp)
os.makedirs(logdir, exist_ok=True)
# change the current working directory to the log folder
os.chdir(logdir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


log_file = f"llama2_generate.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

bnb_version = bnb.__version__ if hasattr(bnb, "__version__") else "0.41.0"
logger.info("Bitsandbytes Version", bnb_version)
logger.info("Transformers Version", transformers.__version__)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--enable_quantization", action="store_true", default=False)
    parser.add_argument("-t", "--llm_int8_threshold", type=float, default=1)
    parser.add_argument("-d", "--dry_run", action="store_true", default=False)
    
    args = parser.parse_args()

    quantization = '8bit' if args.enable_quantization else None    
    llm_int8_threshold = args.llm_int8_threshold

    logger.info(f"quantization: {quantization}")
    logger.info(f"llm_int8_threshold: {llm_int8_threshold}")

    if args.dry_run:
        logger.info("Dry run, exiting...")
        exit(0)

    logger.info("Loading LlamaTokenizer and LlamaForCausalLM")


    quantization_config = BitsAndBytesConfig.from_dict(
        {
            "load_in_8bit": True if quantization == '8bit' else False, 
            "load_in_4bit": True if quantization == '4bit' else False,
            "llm_int8_threshold": llm_int8_threshold
        }) if quantization else None

    tokenizer = LlamaTokenizer.from_pretrained("/home/yuxuan/Llama-2-7b-chat-hf", device_map="cuda:0")
    if quantization_config:
        model = LlamaForCausalLM.from_pretrained("/home/yuxuan/Llama-2-7b-chat-hf", device_map="cuda:0", quantization_config=quantization_config)
    else:
        model = LlamaForCausalLM.from_pretrained("/home/yuxuan/Llama-2-7b-chat-hf", device_map="cuda:0")
    logger.info("Loading LlamaTokenizer and LlamaForCausalLM done")

    # running generation task for 5 tokens for three times 
    for i in range(3):
        logger.info(f"Running generation task for 7 tokens for the {i}th time")

        prompt = """A chat between a curious user and an artificial intelligence assistant. 
The assistant gives helpful, detailed, and polite answers to the user's questions. 
USER: Hi, could you share a tale about a charming llama that grows Medusa-like 
hair and starts its own coffee shop? ASSISTANT:"""


        # record start time
        start = datetime.datetime.now()
        logger.info(f"start time: {start}")
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        output = model.generate(**model_inputs, max_new_tokens=100)
        logger.info(f"output: {output}")

        # record end time
        end = datetime.datetime.now()

        # calculate time elapsed
        time_elapsed = end - start
        logger.info(f"time elapsed: {time_elapsed}")
        
        # number of tokens generated
        logger.info(f"number of tokens generated: {len(output[0])}")

        # number of tokens generated per second
        logger.info(f"number of tokens generated per second: {len(output[0]) / time_elapsed.total_seconds()}")

        print(tokenizer.decode(output[0], skip_special_tokens=True))