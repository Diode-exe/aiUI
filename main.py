"""Main application to run the GPT-1 streaming GUI."""

import logging
from threading import Thread
import tkinter as tk
from gpt1 import GPT1Streamer
from gpt2 import GPT2Streamer
from mode_chooser import Mode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GUI:
    """Simple GUI to interact with GPT streaming output."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPT Streaming Output")
        self.root.withdraw()
        self.prompt_label = tk.Label(self.root, text="Enter your prompt:")
        self.prompt_label.pack()

        self.prompt_entry = tk.Entry(self.root, width=50)
        self.prompt_entry.pack()

        # command set later to avoid circular dependency
        self.generate_button = tk.Button(self.root, text="Generate", command=None)
        self.generate_button.pack()

        self.output_text = tk.Text(self.root, height=20, width=60)
        self.output_text.pack()

class GPT:
    """Class to manage GPT streaming generation."""
    def __init__(self, gui_ref):
        self.gpt1 = GPT1Streamer(gui_ref=gui_ref)
        self.gpt2 = GPT2Streamer(gui_ref=gui_ref)
        self.gui = gui_ref
        self.gui.generate_button.config(command=self.generate_text)

    def generate_text(self):
        """Generate text using GPT-1 and display it in the GUI."""
        prompt = self.gui.prompt_entry.get()
        if mode_chosen == "GPT-1":
            logging.info("Starting GPT-1 streaming generation.")
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            gpt1_thread = Thread(target=self.gpt1.run_gpt1_streamed, args=(prompt,), daemon=True)
            gpt1_thread.start()
        elif mode_chosen == "GPT-2":
            logging.info("Starting GPT-2 streaming generation.")
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            gpt2_thread = Thread(target=self.gpt2.run_gpt2_streamed, args=(prompt,), daemon=True)
            gpt2_thread.start()
        else:
            logging.warning("Mode %s not implemented yet.", mode_chosen)
            self.gui.output_text.delete(1.0, tk.END)
            self.gui.output_text.insert("end", f"Mode '{mode_chosen}' not implemented yet.")

gui = GUI()

mode_choose = Mode(gui_ref=gui)
mode_chosen = mode_choose.chooser()

gpt_class = GPT(gui_ref=gui)

gui.root.mainloop()
