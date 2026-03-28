import customtkinter as ctk

from config import WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, APPEARANCE_MODE, COLOR_THEME
from app.ui.voice_panel import VoicePanel
from app.ui.face_panel import FacePanel
from app.ui.logs_panel import LogsPanel


class MainWindow(ctk.CTk):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()

        self._configure_window()
        self._build_layout()

    def _configure_window(self):
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(900, 600)

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabs()

    def _build_header(self):
        pass

    def _build_tabs(self):
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        voice_tab = self.tab_view.add("Voice Cloning")
        face_tab = self.tab_view.add("Face Swap")
        logs_tab = self.tab_view.add("Logs")

        voice_tab.grid_columnconfigure(0, weight=1)
        voice_tab.grid_rowconfigure(0, weight=1)
        self.voice_panel = VoicePanel(voice_tab)
        self.voice_panel.grid(row=0, column=0, sticky="nsew")

        face_tab.grid_columnconfigure(0, weight=1)
        face_tab.grid_rowconfigure(0, weight=1)
        self.face_panel = FacePanel(face_tab)
        self.face_panel.grid(row=0, column=0, sticky="nsew")

        logs_tab.grid_columnconfigure(0, weight=1)
        logs_tab.grid_rowconfigure(0, weight=1)
        self.logs_panel = LogsPanel(logs_tab)
        self.logs_panel.grid(row=0, column=0, sticky="nsew")

    def run(self):
        self.mainloop()
