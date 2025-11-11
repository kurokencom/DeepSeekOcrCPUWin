from transformers import AutoModel, AutoTokenizer
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, default="test.png", help="Path to the input image file.")
    parser.add_argument("--output_path", type=str, default="result", help="Path to the output directory.")
    args = parser.parse_args()

    model_name = 'model'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.float16)
    model = model.eval()

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    # Use absolute paths to avoid ambiguity
    image_file_path = os.path.abspath(args.image_file)
    output_path_dir = os.path.abspath(args.output_path)

    print(f"Processing image: {image_file_path}")
    print(f"Output will be saved to: {output_path_dir}")

    if not os.path.exists(image_file_path):
        print(f"Error: Image file not found at {image_file_path}")
        return

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file_path,
        output_path=output_path_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )
    print("Processing complete.")
    print(res)

if __name__ == "__main__":
    main()
