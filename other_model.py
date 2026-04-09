import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

class OtherModelStreamer:
    """Class to run another model with streaming output."""
    def __init__(self, model_name="gpt2", gui_ref=None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.stop_requested = False
        self._stop_criteria = None
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

    def run_other_model_streamed(self, prompt, max_length=250):
        """Run another model with streaming output."""
        logging.info("Loading model %s...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)