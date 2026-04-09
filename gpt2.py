"""Run GPT-2 with streaming output using Hugging Face Transformers."""

from threading import Thread
import os
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

class GPT2Streamer:
    """Class to run GPT-2 with streaming output."""
    def __init__(self, model_name="gpt2", gui_ref=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_dir = "gpt2_local"
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

    def run_gpt2_streamed(self, prompt, max_length=250):
        """Run GPT-2 with streaming output."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextIteratorStreamer
        from transformers import StoppingCriteria, StoppingCriteriaList
        # reset stop flag for this run
        self.stop_requested = False

        if os.path.exists(self.model_dir) and \
           os.path.exists(os.path.join(self.model_dir, "config.json")):
            logging.info("Loading GPT-2 from local disk...")
            if self.gui:
                self.gui.status_var.set("Loading GPT-2 from local disk...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        else:
            logging.info("Local model not found. Downloading from Hugging Face...")
            if self.gui:
                self.gui.status_var.set("Downloading GPT-2 from Hugging Face...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # This saves the whole set of necessary files to the folder
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logging.info("Model and tokenizer saved to %s", self.model_dir)
            if self.gui:
                self.gui.status_var.set("Model and tokenizer saved to local disk")

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 1. Initialize the streamer
        # skip_prompt=True ensures we only print the NEW stuff
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 2. Define a stopping criteria that can be triggered externally
        class _StreamStopCriteria(StoppingCriteria):
            def __init__(self, parent):
                self.parent = parent

            def __call__(self, input_ids, scores, **kwargs):
                return getattr(self.parent, "stop_requested", False)

        self._stop_criteria = StoppingCriteriaList([_StreamStopCriteria(self)])

        # generation arguments
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_length": max_length,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "no_repeat_ngram_size": 2,
            "stopping_criteria": self._stop_criteria,
        }

        # 3. Start generation in a separate thread
        # This prevents the program from freezing while waiting for the model
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # 4. Iterate over the streamer and print live
        if self.gui:
            self._append_gui_text(f"--- GPT-2 Output (Streaming) ---\n{prompt}")
        else:
            print(f"--- GPT-2 Output (Streaming) ---\n{prompt}", end="", flush=True)

        for new_text in streamer:
            if self.gui:
                self._append_gui_text(new_text)
            else:
                print(new_text, end="", flush=True)

        # generation finished; reset stop flag
        self.stop_requested = False

        if self.gui:
            self.gui.status_var.set("GPT-2 Generation Finished")
        else:
            print("\n--- GPT-2 Generation Finished ---")

    def request_stop(self):
        """Request that the running generation stop as soon as the model checks criteria."""
        self.stop_requested = True
