import os
import tkinter as tk

from gui.controller import AppController
from gui.view import AppView


def test_app_view_initializes():
    root = tk.Tk()
    try:
        view = AppView(controller=AppController(view=None))
        assert view.title() == "Drowsiness Detection System"
    finally:
        root.destroy()
