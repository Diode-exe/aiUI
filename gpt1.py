"""Run GPT-1 with streaming output using Hugging Face Transformers."""

from threading import Thread
import os
import logging
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, TextIteratorStreamer

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

class GPT1Streamer:
    """Class to run GPT-1 with streaming output."""
    def __init__(self, model_name="openai-gpt", gui_ref=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_dir = "gpt1_local"
        self.gui = gui_ref
        if self.gui:
            logging.info("GUI working.")
        else:
            logging.info("No GUI, will print to console.")

    def _append_gui_text(self, text):
        """Safely append text to Tk widgets from any thread."""
        if not self.gui:
            return

        def _append():
            self.gui.output_text.insert("end", text)
            self.gui.output_text.see("end")

        self.gui.root.after(0, _append)

    def run_gpt1_streamed(self, prompt, max_length=250):
        """Run GPT-1 with streaming output."""
        if os.path.exists(self.model_dir) and \
           os.path.exists(os.path.join(self.model_dir, "config.json")):
            logging.info("Loading GPT-1 from local disk...")
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained(self.model_dir)
            self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_dir)
        else:
            logging.info("Local model not found. Downloading from Hugging Face...")
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained(self.model_name)
            self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_name)

            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # This saves the whole set of necessary files to the folder
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logging.info("Model and tokenizer saved to %s", self.model_dir)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 1. Initialize the streamer
        # skip_prompt=True ensures we only print the NEW stuff
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 2. Define the generation arguments
        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            streamer=streamer,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            use_cache=False,        # Disable caching
            past_key_values=None    # model gets confused without this
        )

        # 3. Start generation in a separate thread
        # This prevents the program from freezing while waiting for the model
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # 4. Iterate over the streamer and print live
        if self.gui:
            self._append_gui_text(f"--- GPT-1 Output (Streaming) ---\n{prompt}")
        else:
            print(f"--- GPT-1 Output (Streaming) ---\n{prompt}", end="", flush=True)

        for new_text in streamer:
            if self.gui:
                self._append_gui_text(new_text)
            else:
                print(new_text, end="", flush=True)

        print("\n--- Generation Finished ---")
