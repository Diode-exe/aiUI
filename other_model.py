import logging
import os
from threading import Thread

class OtherModelStreamer:
    """Class to run another model with streaming output."""
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", gui_ref=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.auto_model_cls = None
        self.auto_tokenizer_cls = None
        self.text_iterator_streamer_cls = None
        self.stopping_criteria_cls = None
        self.stopping_criteria_list_cls = None
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

    def _import_transformers(self):
        """Import transformer classes lazily and cache the symbols used by this streamer."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            try:
                # older transformers exposed AutoModelWithLMHead
                from transformers import AutoModelWithLMHead as AutoModelForCausalLM
            except ImportError:
                AutoModelForCausalLM = None

        try:
            from transformers import AutoTokenizer, TextIteratorStreamer
            from transformers import StoppingCriteria, StoppingCriteriaList
        except ImportError as e:
            raise ImportError(
                "Required classes not available from transformers. "
                "Try upgrading with `pip install -U transformers`."
            ) from e

        self.auto_model_cls = AutoModelForCausalLM
        self.auto_tokenizer_cls = AutoTokenizer
        self.text_iterator_streamer_cls = TextIteratorStreamer
        self.stopping_criteria_cls = StoppingCriteria
        self.stopping_criteria_list_cls = StoppingCriteriaList

    def load_model(self):
        self._import_transformers()
        if os.path.exists(self.model_dir) and \
        os.path.exists(os.path.join(self.model_dir, "config.json")):
            logging.info("Loading Other Model from local disk...")
            if self.gui:
                self.gui.status_var.set("Loading Other Model from local disk...")
            self.tokenizer = self.auto_tokenizer_cls.from_pretrained(self.model_dir)
            self.model = self.auto_model_cls.from_pretrained(self.model_dir)
        else:
            logging.info("Local model not found. Downloading from Hugging Face...")
            if self.gui:
                self.gui.status_var.set("Downloading Other Model from Hugging Face...")
            self.tokenizer = self.auto_tokenizer_cls.from_pretrained(self.model_name)
            if self.auto_model_cls is None:
                raise ImportError(
                    "transformers does not provide AutoModelForCausalLM or a compatible fallback. "
                    "Please upgrade your `transformers` package: `pip install -U transformers`"
                )
            self.model = self.auto_model_cls.from_pretrained(self.model_name)

            # Create directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)

            # This saves the whole set of necessary files to the folder
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logging.info("Model and tokenizer saved to %s", self.model_dir)

    def run_other_model_streamed(self, prompt, max_length=250):
        """Run other model with streaming output."""
        # Import heavy transformer classes lazily to avoid import-time failures
        if (
            self.text_iterator_streamer_cls is None
            or self.stopping_criteria_cls is None
            or self.stopping_criteria_list_cls is None
        ):
            self._import_transformers()

        if not self.model or not self.tokenizer:
            self.load_model()

        # reset stop flag for this run
        self.stop_requested = False

        inputs = self.tokenizer(prompt, return_tensors="pt")

        # 1. Initialize the streamer
        # skip_prompt=True ensures we only print the NEW stuff
        streamer = self.text_iterator_streamer_cls(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 2. Define a stopping criteria that can be triggered externally
        class _StreamStopCriteria(self.stopping_criteria_cls):
            def __init__(self, parent):
                self.parent = parent

            def __call__(self, input_ids, scores, **kwargs):
                return getattr(self.parent, "stop_requested", False)

        self._stop_criteria = self.stopping_criteria_list_cls([_StreamStopCriteria(self)])

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

        if self.gui:
            self.gui.status_var.set("Other Model Generation Finished")
            self.gui.output_text.insert("end", "\n--- Other Model Generation Finished ---\n")
        else:
            print("\n--- Other Model Generation Finished ---")

    def request_stop(self):
        """Request that the running generation stop as soon as the model checks criteria."""
        self.stop_requested = True
        self.stop_requested = True