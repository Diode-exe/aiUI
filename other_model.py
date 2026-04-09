import logging
import os
from threading import Thread

class OtherModelStreamer:
    """Class to run another model with streaming output."""
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", gui_ref=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.stop_requested = False
        self._stop_criteria = None
        self.model_dir = "other_model_local"
        self.gui = gui_ref
        if self.gui:
            logging.info("GUI working.")
        else:
            logging.info("No GUI, will print to console.")

    def _append_gui_text(self, text):
        """Safely append text to Tk widgets from any thread."""
        if not self.gui:
            print(text, end="", flush=True)
            return

        def _append():
            self.gui.output_text.insert("end", text)
            self.gui.output_text.see("end")

        self.gui.root.after(0, _append)

    def run_other_model_streamed(self, prompt, max_length=250):
        """Run other model with streaming output."""
        # Import heavy transformer classes lazily to avoid import-time failures
        try:
            from transformers import AutoModelForCausalLM
        except Exception:
            try:
                # older transformers exposed AutoModelWithLMHead
                from transformers import AutoModelWithLMHead as AutoModelForCausalLM
            except Exception:
                AutoModelForCausalLM = None
        try:
            from transformers import AutoTokenizer, TextIteratorStreamer
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception as e:
            raise ImportError(
                "Required classes not available from transformers. "
                "Try upgrading with `pip install -U transformers`."
            ) from e
        if os.path.exists(self.model_dir) and \
           os.path.exists(os.path.join(self.model_dir, "config.json")):
                logging.info("Loading Other Model from local disk...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        else:
            logging.info("Local model not found. Downloading from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if AutoModelForCausalLM is None:
                raise ImportError(
                    "transformers does not provide AutoModelForCausalLM or a compatible fallback. "
                    "Please upgrade your `transformers` package: `pip install -U transformers`"
                )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # This saves the whole set of necessary files to the folder
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logging.info("Model and tokenizer saved to %s", self.model_dir)

        # reset stop flag for this run
        self.stop_requested = False

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
            "no_repeat_ngram_size": 3,
            "stopping_criteria": self._stop_criteria,
        }

        # 3. Start generation in a separate thread
        # This prevents the program from freezing while waiting for the model
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        # 4. Iterate over the streamer and print live
        if self.gui:
            self._append_gui_text(f"--- Other Model Output (Streaming) ---\n{prompt}")
        else:
            print(f"--- Other Model Output (Streaming) ---\n{prompt}", end="", flush=True)

        for new_text in streamer:
            if self.gui:
                self._append_gui_text(new_text)
            else:
                print(new_text, end="", flush=True)

        # generation finished; reset stop flag
        self.stop_requested = False

        print("\n--- Other Model Generation Finished ---")
