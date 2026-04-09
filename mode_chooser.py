"""Mode chooser dialog for selecting generation mode."""

import tkinter as tk
from tkinter import messagebox
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Mode:
    """Class to manage mode selection via a modal dialog."""
    def __init__(self, gui_ref=None):
        self.mode_name = None
        self.gui_ref = gui_ref
        self.ok_btn = None
        self.cancel_btn = None
        self.chooser_window = tk.Toplevel(master=self.gui_ref.root)
        self.chooser_window.title("Select Generation Mode")
        self.chooser_window.bind("<Escape>", lambda e:
            self.invoke_button(self.cancel_btn))  # esc to exit
        self.chooser_window.bind("<Return>", lambda e:
            self.invoke_button(self.ok_btn))  # enter to confirm
        # Bind the StringVar to the chooser window so selections are captured
        self.var = tk.StringVar(master=self.chooser_window, value="GPT-1")  # Default selection

    def chooser(self, options=("GPT-1", "GPT-2", "Other")):
        """Open a modal Toplevel with radio buttons and return the choice."""

        # Use the instance StringVar so on_ok can read the selected value
        self.var.set(self.mode_name or options[0])
        for opt in options:
            tk.Radiobutton(self.chooser_window, text=opt, variable=self.var,
                           value=opt).pack(anchor="w", padx=8, pady=2)

        btn_frame = tk.Frame(self.chooser_window)
        btn_frame.pack(pady=8)
        self.ok_btn = tk.Button(btn_frame, text="OK", command=self.on_ok)
        self.ok_btn.pack(side="left", padx=6)
        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.ask_to_kill)
        self.cancel_btn.pack(side="left", padx=6)
        self.chooser_window.grab_set()
        self.chooser_window.wait_window()
        return self.mode_name

    def on_ok(self):
        """Handle OK button click: save selection, close chooser, and reopen main window."""
        self.mode_name = self.var.get()
        self.chooser_window.destroy()
        self.reappear()
        return self.mode_name

    def reappear(self):
        """Reopen the main window"""
        self.gui_ref.root.deiconify()

    def invoke_button(self, button):
        """Invoke a button's command programmatically."""
        try:
            if button and button.winfo_exists():
                button.invoke()
        except AttributeError:
            logging.error("Button does not exist or has no command to invoke.")
