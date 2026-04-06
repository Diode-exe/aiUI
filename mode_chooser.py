"""Mode chooser dialog for selecting generation mode."""

import tkinter as tk


class Mode:
    def __init__(self, mode_name=None, gui_ref=None):
        self.mode_name = mode_name
        self.gui = gui_ref

    def chooser(self, options=("Fast", "Balanced", "Accurate")):
        """Open a modal Toplevel with radio buttons and return the choice."""
        if not self.gui or not hasattr(self.gui, "root"):
            # No GUI provided: return default or existing mode
            self.mode_name = self.mode_name or options[0]
            return self.mode_name

        top = tk.Toplevel(self.gui.root)
        top.title("Choose mode")

        var = tk.StringVar(master=top, value=self.mode_name or options[0])
        for opt in options:
            tk.Radiobutton(top, text=opt, variable=var, value=opt).pack(anchor="w", padx=8, pady=2)

        def on_ok():
            self.mode_name = var.get()
            top.destroy()

        btn_frame = tk.Frame(top)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side="left", padx=6)

        top.transient(self.gui.root)
        top.grab_set()
        self.gui.root.wait_window(top)
        return self.mode_name
