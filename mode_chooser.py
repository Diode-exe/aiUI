"""Mode chooser dialog for selecting generation mode."""

import tkinter as tk


class Mode:
    """Class to manage mode selection via a modal dialog."""
    def __init__(self, gui_ref=None):
        self.mode_name = None
        self.gui_ref = gui_ref
        self.chooser_window = tk.Tk()
        self.chooser_window.title("Select Generation Mode")

    def chooser(self, options=("GPT-1", "GPT-2", "Other")):
        """Open a modal Toplevel with radio buttons and return the choice."""

        var = tk.StringVar(master=self.chooser_window, value=self.mode_name or options[0])
        for opt in options:
            tk.Radiobutton(self.chooser_window, text=opt, variable=var, value=opt).pack(anchor="w", padx=8, pady=2)

        def on_ok():
            self.mode_name = var.get()
            self.chooser_window.destroy()
            self.reappear()
            return self.mode_name

        btn_frame = tk.Frame(self.chooser_window)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=self.chooser_window.destroy).pack(side="left", padx=6)

        self.chooser_window.mainloop()
        return self.mode_name

    def reappear(self):
        """Reopen the main window"""
        self.gui_ref.root.deiconify()
