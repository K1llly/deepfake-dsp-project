import customtkinter as ctk
from app.utils.logger import get_logger


LEVEL_COLORS = {
    "info": "#aaaaaa",
    "success": "#2ecc71",
    "error": "#e74c3c",
    "warning": "#f39c12",
}


class LogsPanel(ctk.CTkFrame):
    """Scrollable log viewer with color-coded entries."""

    def __init__(self, parent):
        super().__init__(parent)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        ctk.CTkLabel(
            header, text="Application Logs", font=("Arial", 20, "bold")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            header, text="Clear Logs", width=100,
            command=self._on_clear,
            fg_color="#555555", hover_color="#666666"
        ).pack(side="right", padx=5)

        # Log display
        self.log_display = ctk.CTkTextbox(
            self, font=("Courier", 12), state="disabled",
            wrap="word", fg_color="#0d0d0d"
        )
        self.log_display.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # Configure text tags for colors
        self.log_display._textbox.tag_configure("info", foreground="#aaaaaa")
        self.log_display._textbox.tag_configure("success", foreground="#2ecc71")
        self.log_display._textbox.tag_configure("error", foreground="#e74c3c")
        self.log_display._textbox.tag_configure("warning", foreground="#f39c12")

        # Subscribe to logger
        logger = get_logger()

        # Load existing entries
        for entry in logger.get_entries():
            self._append_entry(entry)

        # Listen for new entries
        logger.subscribe(self._on_new_entry)

    def _on_new_entry(self, entry):
        """Called from any thread when a new log entry is added."""
        self.after(0, lambda: self._append_entry(entry))

    def _append_entry(self, entry):
        """Add a log entry to the display."""
        self.log_display.configure(state="normal")
        self.log_display._textbox.insert("end", str(entry) + "\n", entry.level)
        self.log_display.configure(state="disabled")
        self.log_display.see("end")

    def _on_clear(self):
        """Clear all logs."""
        get_logger().clear()
        self.log_display.configure(state="normal")
        self.log_display.delete("1.0", "end")
        self.log_display.configure(state="disabled")
