#!/usr/bin/env python3
import argparse
import os
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv()

    model = os.getenv("MODEL")
    outbase = os.getenv("OUTBASE")
    outdir = os.getenv("OUTDIR")
    llamabase = os.getenv("LLAMABASE", "~/Documents/llama.cpp")
    ggml_version = os.getenv("GGML_VERSION", "v3")

    if not os.path.isdir(model):
        raise Exception(f"Could not find model dir at {model}")

    if not os.path.isfile(f"{model}/config.json"):
        raise Exception(f"Could not find config.json in {model}")

    os.makedirs(outdir, exist_ok=True)

    print("Building llama.cpp")
    subprocess.run(f"cd {llamabase} && git pull && make clean && LLAMA_METAL=1 make", shell=True, check=True)

    fp16 = f"{outdir}/{outbase}.ggmlv2.fp16.bin"

    print(f"Making unquantised GGML at {fp16}")
    if not os.path.isfile(fp16):
        subprocess.run(f"python3 {llamabase}/convert.py {model} --outtype f16 --outfile {fp16}", shell=True, check=True)
    else:
        print(f"Unquantised GGML already exists at: {fp16}")

    print("Making quants")
    for type in ["q2_k", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]:
        outfile = f"{outdir}/{outbase}.ggml{ggml_version}.{type}.bin"
        print(f"Making {type} : {outfile}")
        subprocess.run(f"{llamabase}/quantize {fp16} {outfile} {type}", shell=True, check=True)

    os.remove(fp16)

if __name__ == "__main__":
    main()