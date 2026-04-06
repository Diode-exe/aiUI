"""Main application to run the GPT-1 streaming GUI."""

from threading import Thread
import tkinter as tk
from gpt1 import GPT1Streamer
# from mode_chooser import Mode

class GUI:
    """Simple GUI to interact with GPT-1 streaming output."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPT-1 Streaming Output")
        self.setup_widgets()

    def setup_widgets(self):
        """Set up the GUI widgets."""
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
    """Class to manage GPT-1 streaming generation."""
    def __init__(self, gui_ref):
        self.gpt1 = GPT1Streamer(gui_ref=gui_ref)
        self.gui = gui_ref

    def generate_text(self):
        """Generate text using GPT-1 and display it in the GUI."""
        prompt = self.gui.prompt_entry.get()
        if prompt:
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            Thread(target=self.gpt1.run_gpt1_streamed, args=(prompt,), daemon=True).start()

gui = GUI()
gpt = GPT(gui)
gui.generate_button.config(command=gpt.generate_text)
# mode_choose = Mode(gui_ref=gui)
# mode_choose.chooser()
gui.root.mainloop()
