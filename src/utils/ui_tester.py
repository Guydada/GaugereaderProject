import cv2
import os
import numpy as np
import tkinter.ttk as ttk
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.filedialog as fd
import PIL.Image as Image
import PIL.ImageTk as ImageTk
import time
import typer

import src.utils.image_editing as ie
import src.utils.point_math as point_math
import src.utils.circle_dectection as cd
import src.utils.envconfig as env
import src.utils.convert_xml as xmlr


class TestUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('UI Tester')
        self.root.geometry('800x600')
        self.root.resizable(False, False)
        self.root.configure(background='white')

        self.frame = ttk.Frame(self.root, width=800, height=600)
        self.frame.pack(fill='both', expand=True)
        self.test_image = fd.askopenfilename(initialdir=env.PROJECT_ROOT,
                                             title="Select image",
                                             filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.test_image = cv2.imread(self.test_image)
        self.test_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        self.test_image = cv2.resize(self.test_image, (80, 60))
        self.test_image = Image.fromarray(self.test_image)
        self.test_image = ImageTk.PhotoImage(self.test_image)
        self.test_image_label = ttk.Label(self.frame, image=self.test_image)
        self.test_image_label.pack(fill='both', expand=True)
        self.test_button = ttk.Button(self.frame, text='Test', command=self.test)
        self.test_button.pack(fill='both', expand=True)

        self.root.mainloop()

    def test(self):
        return


if __name__ == '__main__':
    test = TestUI()