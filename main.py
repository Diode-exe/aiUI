"""Main application to run the GPT-1 streaming GUI."""

import logging
import sys
from threading import Thread
import tkinter as tk
from tkinter import messagebox
from gpt1 import GPT1Streamer
from gpt2 import GPT2Streamer
from other_model import OtherModelStreamer
from mode_chooser import Mode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GUI:
    """Simple GUI to interact with GPT streaming output."""
    def __init__(self):
        self.mode_chosen = None
        self.mode_choose = None
        self.root = tk.Tk()
        self.root.title("GPT Streaming Output")
        self.root.protocol("WM_DELETE_WINDOW", self.ask_to_kill)
        self.root.withdraw()
        self.field_frame = tk.Frame(self.root)
        self.prompt_label = tk.Label(self.field_frame, text="Enter your prompt:")
        self.prompt_label.pack(side="left")

        self.prompt_entry = tk.Entry(self.field_frame, width=50)
        self.prompt_entry.pack(side="left", padx=10)
        self.length_label = tk.Label(self.field_frame, text="Max Length:")
        self.length_label.pack(side="right")
        self.length_entry = tk.Entry(self.field_frame, width=10)
        self.length_entry.pack(side="right")
        self.field_frame.pack()

        # buttons (command set later to avoid circular dependency)
        self.button_frame = tk.Frame(self.root)
        self.generate_button = tk.Button(self.button_frame, text="Generate", command=None)
        self.generate_button.pack(side="left", padx=6)
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=None)
        self.stop_button.pack(side="left", padx=6)
        self.button_frame.pack(pady=6)

        # key bindings: Enter -> generate, Escape -> stop
        self.root.bind("<Return>", lambda e: self.generate_button.invoke())
        self.root.bind("<Escape>", lambda e: self.stop_button.invoke())

        self.output_text = tk.Text(self.root, height=20, width=60)
        self.output_text.pack()

    def kill(self):
        """Kill the chooser window if it's still open."""
        if self.root.winfo_exists():
            self.root.destroy()

    def ask_to_kill(self):
        """Ask the user to confirm exiting the application."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.kill()

class GPT:
    """Class to manage GPT streaming generation."""
    def __init__(self, gui_ref):
        logging.info("Initializing GPT class...")
        self.gpt1 = GPT1Streamer(gui_ref=gui_ref)
        self.gpt2 = GPT2Streamer(gui_ref=gui_ref)
        self.gui = gui_ref
        self.gui.generate_button.config(command=self.generate_text)
        # wire Stop button to the stop handler
        self.gui.stop_button.config(command=self.stop_generation)
        self.length = 250
        # Keep thread references on the instance so they persist across calls
        self.gpt1_thread = None
        self.gpt2_thread = None
        self.other_model = OtherModelStreamer(gui_ref=gui_ref)

    def generate_text(self):
        """Generate text using GPT-1 and display it in the GUI."""
        logging.info("Generate button clicked.")
        # If a generation thread is already running, do not start another.
        # Stopping threads forcibly (e.g. using private methods) is unsafe, so we simply refuse.
        if self.gpt1_thread and self.gpt1_thread.is_alive():
            logging.warning("GPT-1 generation is still running. Please wait until it finishes.")
            return
        if self.gpt2_thread and self.gpt2_thread.is_alive():
            logging.warning("GPT-2 generation is still running. Please wait until it finishes.")
            return
        prompt = self.gui.prompt_entry.get()
        logging.info("Prompt received: %s", prompt)
        # initialize length variable to satisfy static analysis and provide a default
        if self.gui.mode_chosen == "GPT-1":
            logging.info("Starting GPT-1 streaming generation.")
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            try:
                length = int(self.gui.length_entry.get())
            except ValueError:
                length = 250
            self.gpt1_thread = Thread(target=self.gpt1.run_gpt1_streamed,
                                      args=(prompt, length), daemon=True)
            self.gpt1_thread.start()
        elif self.gui.mode_chosen == "GPT-2":
            logging.info("Starting GPT-2 streaming generation.")
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            # Read length from GUI; fall back to 250 if blank or invalid
            try:
                length = int(self.gui.length_entry.get())
            except ValueError:
                length = 250
            self.gpt2_thread = Thread(target=self.gpt2.run_gpt2_streamed,
                                      args=(prompt, length), daemon=True)
            self.gpt2_thread.start()
        else:
            logging.info("Starting Other model streaming generation.")
            self.gui.output_text.delete(1.0, tk.END)  # Clear previous output
            try:
                length = int(self.gui.length_entry.get())
            except ValueError:
                length = 250
            self.other_model_thread = Thread(target=self.other_model.run_other_model_streamed,
                                             args=(prompt, length), daemon=True)
            self.other_model_thread.start()

    def stop_generation(self):
        """Request the currently running generation to stop."""
        logging.info("Stop requested from GUI.")
        # Prefer to stop the thread that is currently running
        if self.gpt1_thread and self.gpt1_thread.is_alive():
            logging.info("Requesting stop for GPT-1")
            self.gpt1.request_stop()
            self.gui.output_text.insert("end", "\n[Stop requested for GPT-1]\n")
            return
        if self.gpt2_thread and self.gpt2_thread.is_alive():
            logging.info("Requesting stop for GPT-2")
            self.gpt2.request_stop()
            self.gui.output_text.insert("end", "\n[Stop requested for GPT-2]\n")
            return
        if self.other_model_thread and self.other_model_thread.is_alive():
            logging.info("Requesting stop for Other model")
            self.other_model.request_stop()
            self.gui.output_text.insert("end", "\n[Stop requested for Other model]\n")
            return

        logging.info("No active generation to stop.")
        self.gui.output_text.insert("end", "\n[No active generation to stop]\n")

gui = GUI()

gui.mode_choose = Mode(gui_ref=gui)
gui.mode_chosen = gui.mode_choose.chooser()

try:
    gpt_class = GPT(gui_ref=gui)
except tk.TclError as e:
    logging.error("Failed to initialize GPT class due to Tkinter error: %s", e)
    logging.info("May have exited via Cancel button. Exiting application.")
    sys.exit(0)

gui.root.mainloop()
