class Mode:
    def __init__(self, mode_name, gui_ref=None):
        self.mode_name = mode_name
        self.gui = gui_ref

    def chooser(self):
        chooser_window = self.gui.Toplevel(self.gui.root)
        chooser_window.title("Choose Mode")
        self.gui.Label(chooser_window, text="Select a mode:").pack(pady=10)
        