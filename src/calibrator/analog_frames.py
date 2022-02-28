import cv2
import numpy as np
import tkinter.ttk as ttk
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.filedialog as fd
import tkinter.simpledialog as sd
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageTk as ImageTk
import time
import typer

from copy import copy
from pathlib import Path

import src.utils.image_editing as ie
import src.utils.point_math as point_math
import src.utils.circle_dectection as cd
import src.utils.envconfig as env
import src.utils.convert_xml as xmlr


class Toolbar(ttk.Frame):
    def __init__(self,
                 master):
        ttk.Frame.__init__(self, master)
        self.master = master


class circle_detection_frame(tk.Frame):
    def __init__(self,
                 master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title("Circle Detection")
