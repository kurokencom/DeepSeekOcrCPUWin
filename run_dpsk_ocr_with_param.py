from transformers import AutoModel, AutoTokenizer
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, default="test.png", help="Path to the input image file.")
    parser.add_argument("--output_path", type=str, default="result", help="Path to the output directory.")
    parser.add_argument("--model_name", type=str, default="DeepSeek-OCR-master/DeepSeek-OCR-hf/local_model", help="Path to the model directory.")
    parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ", help="Prompt to use for OCR.")
    parser.add_argument("--base_size", type=int, default=1024, help="Base size for image processing.")
    parser.add_argument("--image_size", type=int, default=640, help="Image size for processing.")
    parser.add_argument("--crop_mode", type=bool, default=True, help="Enable or disable crop mode.")
    parser.add_argument("--save_results", type=bool, default=True, help="Enable or disable saving results.")
    parser.add_argument("--test_compress", type=bool, default=True, help="Enable or disable test compression.")
    parser.add_argument("--eval_mode", type=bool, default=False, help="Enable or disable eval mode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum new tokens for generation.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None, help="No repeat ngram size for generation.")
    args = parser.parse_args()

    model_name = os.path.abspath(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.float16)
    model = model.eval()

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
        prompt=args.prompt,
        image_file=image_file_path,
        output_path=output_path_dir,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        save_results=args.save_results,
        test_compress=args.test_compress,
        eval_mode=args.eval_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    print("Processing complete.")
    print(res)

if __name__ == "__main__":
    main()
