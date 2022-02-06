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

from dataclasses import dataclass
from copy import copy
from pathlib import Path

import src.utils.image_editing as ie
import src.utils.point_math as point_math
import src.calibrator.circle_dectection as cd
import src.utils.envconfig as env
import src.utils.convert_xml as xmlr

width = env.WINDOW_SIZE[0]
height = env.WINDOW_SIZE[1]


@dataclass
class Point:
    x: int
    y: int
    angle_to_current: float


class Calibrator:  # TODO: add user input fo max angle and min_angle
    def __init__(self,
                 calibration_image: str,
                 index: str,
                 camera_id: str = None,
                 gauge_type: str = 'analog'):

        # Outer variables
        self.rotated_im = None
        self.camera_id = camera_id
        self.index = index
        self.calibration_image = calibration_image
        self.gauge_type = gauge_type

        # Inner variables
        self.directory, self.xml_file = env.dir_file_from_camera_gauge(camera_id, index)
        self.calibration_image_path = Path(self.directory).joinpath(calibration_image).as_posix()
        self.needle_image_path = Path(self.directory).joinpath(env.NEEDLE_IMAGE_NAME).as_posix()
        self.train_image_path = Path(self.directory).joinpath(env.TRAIN_IMAGE_NAME).as_posix()
        self.current_image_path = None
        self.orig_cv = None
        self.proc_cv = None
        self.orig_im = None
        self.proc_im = None
        self.train_image = None
        self.needle_image = None
        self.mask = None
        self.params = {}  # placeholder for params of image editing
        self.calibration = {}
        self.tags = ['crop', 'needle', 'max_line', 'min_line', 'center', 'gauge', 'no_circles']
        self.x = 0  # Gauge center x
        self.y = 0  # Gauge center y
        self.r = 0  # Gauge radius
        self.crop_detection = {'min_r': 0,
                               'max_r': 0,
                               'min_d': 0}
        self.cur_angle = 0
        self.step_angle = 0
        self.value_diff = 0
        self.angle_diff = 0
        self.current_reading = 0
        self.min_val = Point(0, 0, 0)
        self.max_val = Point(0, 0, 360)

        # BOOL variables for idiot proofing
        self.cropped = False
        self.needle_found = False
        self.max_line_found = False
        self.min_line_found = False
        self.circles_found = False
        self.params_set = False
        self.masked = False
        self.rotate = False
        self.test_rotation = False

        # Image variables
        self.w, self.h = None, None

        # Window settings
        self.window = tk.Tk()
        self.window.title("Calibrator App")
        self.window.resizable(width=True, height=True)
        self.window.configure(background='#ffffff')
        self.canvas = None

        # Menubar
        menubar = tk.Menu(self.window)
        file = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file)
        file.add_command(label='Change Calibration Image', command=self.open_img)
        file.add_command(label='Save to XML', command=self.save_calibration_data)
        file.add_command(label='Exit', command=self.window.destroy)
        menubar.add_cascade(label='Help', command=lambda: None)
        self.window.config(menu=menubar)

        # Image Editor Frame
        self.editor_frame = tk.Frame(self.window)
        self.editor_frame.pack(side=tk.TOP, fill=tk.BOTH)

        # Image Frame settings
        self.image_frame = tk.Frame(master=self.window, width=width, height=height)
        self.image_frame.configure(background='gray')

        # Tweak Frame
        self.tweak_frame = None

        # Define Frame (finding the needle and the calibration circle)
        self.define_frame = None
        self.mark_needle_button = None
        self.brush_size_bar = None
        self.mark_min_line_button = None
        self.mark_max_line_button = None
        self.set_define_parameters_button = None
        self.detect_circles_button = None
        self.define_frame_button = None
        self.show_needle_button = None
        self.test_needle_rotation_button = None
        self.read_value_button = None

        # Text Frame
        self.text_frame = None

        # Image Editor Frame - Buttons
        # TODO: add editor functions
        tk.Button(self.editor_frame, text='Reset', command=self.reset_all).grid(row=0, column=0)
        tk.Button(self.editor_frame, text='Edit', command=self.edit_top_frame).grid(row=0, column=1)
        self.crop_button = tk.Button(self.editor_frame, text='Crop', command=self.use_crop)
        self.crop_button.grid(row=0, column=2)
        crop_options = ['Auto Crop', 'Manual Crop']
        self.crop_mode = tk.StringVar(self.window)
        self.crop_mode.set(crop_options[0])
        crop_mode_menu = tk.OptionMenu(self.editor_frame, self.crop_mode, *crop_options)
        crop_mode_menu.grid(row=1, column=2)

        # Gauge Detection Steps Frame
        self.gauge_detection_steps_frame = tk.Frame(self.window)
        self.gauge_detection_steps_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        tk.Label(master=self.gauge_detection_steps_frame,
                 text='Tasks',
                 font=('Helvetica', 12)).grid(row=0, column=0, sticky=tk.W)
        tk.Label(master=self.gauge_detection_steps_frame,
                 text='Settings',
                 font=('Helvetica', 12)).grid(row=10, column=0, sticky=tk.W)

        # Image editing variables

        self.active_button = None
        self.active_shape = None
        self.brush_size = 10  # default brush size
        self.eraser = False

        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.drawn = None
        self.color = 'red'
        self.draw = None

        # ANALOG MODE #
        if self.gauge_type == 'analog':
            self.circle_frame = None
            # Circle detection Menu
            d_keys = ['min_radius', 'max_radius', 'min_dist', 'max_dist']
            d_text = 'Circle Detection'
            d_func = self.circle_detection_frame
            # Text Frame
            t_keys = ['min_value', 'max_value', 'units']
            t_text = 'Parameters'
            t_func = self.circles_text_input_frame
            self.auto_crop_radius_bar = None
            self.auto_crop_radius = 1.5

        # DIGITAL MODE #
        else:
            # Characters detection Menu
            d_keys = ['digit_type']
            d_text = 'Digits Detection'
            d_func = self.characters_detection_frame
            # Text Frame
            t_keys = ['min_value', 'max_value', 'step', 'units']
            t_text = 'Parameters'
            t_func = self.characters_text_input_frame

        self.d_text = d_text
        self.t_func = t_func
        self.d_func = d_func
        self.cd = dict.fromkeys(d_keys)
        self.tp = dict.fromkeys(t_keys)

        # Additional buttons post creation
        if self.gauge_type == 'analog':
            self.needle_rotation_frame = None
            self.needle_rotation_scale = None
            self.auto_crop_radius_bar = tk.Scale(self.editor_frame,
                                                 from_=1,
                                                 to=2,
                                                 orient=tk.HORIZONTAL,
                                                 resolution=0.025,
                                                 label='Auto Crop Radius')
            self.auto_crop_radius_bar.set(self.auto_crop_radius)
            self.auto_crop_radius_bar.grid(row=2, column=2)

        # Open Image
        if self.calibration_image is not None:
            self.open_img(image=calibration_image)

    # Image loading methods

    def open_img(self,
                 image: str = None,
                 keep_window: bool = False,
                 resize: bool = True):
        if image is None:
            path = fd.askopenfilename(initialdir=env.CALIBRATION_PATH,
                                      title="Select Calibration Image",
                                      filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        else:
            path = Path(self.directory).joinpath(image).as_posix()
        for button in [self.define_frame_button,
                       self.detect_circles_button,
                       self.tweak_frame,
                       self.show_needle_button,
                       self.test_needle_rotation_button,
                       self.read_value_button]:
            if button is not None:
                button.destroy()

        for top in [self.circle_frame, self.define_frame]:
            if top is not None:
                top.destroy()

        # check if top level window is open and destroy it
        self.current_image_path = path
        self.orig_cv = cv2.imread(self.current_image_path)
        if self.orig_cv is None:
            raise Exception("Image not found")
        self.proc_cv = self.orig_cv.copy()
        if resize:
            self.proc_cv = self._resize_cv(self.proc_cv)
        else:
            self.h = self.proc_cv.shape[0]
            self.w = self.proc_cv.shape[1]
        self.orig_im = self._cv_to_imagetk(self.proc_cv)
        self.proc_im = copy(self.orig_im)
        self.reset_crop_detection()
        self._create_canvas()
        w_w, w_h = int(self.w * 1.4), int(self.h * 1.1)
        if not keep_window:
            self.window.geometry(f'{w_w}x{w_h}')
        self.show_image()

    @staticmethod
    def _cv_to_imagetk(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        return image

    @staticmethod
    def cv_to_image(cv_image, show: bool = False):
        image = cv_image.copy()
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if show:
            image.show()
        return image

    def _resize_cv(self,
                   image):
        h = image.shape[0]
        w = image.shape[1]
        self.w = width
        self.h = int(h * width / w)
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return image

    def _create_canvas(self):
        if self.canvas is not None:
            self.canvas.destroy()
            self.canvas = None
        self.canvas = tk.Canvas(self.image_frame,
                                bg='black',
                                width=self.w,
                                height=self.h,
                                cursor="cross")
        self.canvas_image = self.canvas.create_image(0, 0, image=self.orig_im, anchor=tk.NW)
        self.canvas.place(anchor=tk.CENTER, relx=0.5, rely=0.5)
        self.canvas.pack(anchor=tk.CENTER, padx=10, pady=10)
        self.image_frame.place(anchor=tk.CENTER, relx=0.5, rely=0.5)
        self.image_frame.pack(side=tk.TOP, anchor=tk.CENTER, padx=10, pady=10, fill=tk.BOTH, expand=True)

    def canvas_to_proc_im(self):  # TODO: complete this
        self.draw = ImageDraw.Draw(self.draw)

    def activate_button(self,
                        button):
        if self.active_button is not None:
            try:
                self.active_button.config(relief=tk.RAISED)
            except tk.TclError:
                ''
        self.active_button = button
        self.active_button.config(relief=tk.SUNKEN)

    def draw_shape(self):
        self.canvas.bind("<ButtonPress-1>", self.on_start)
        self.canvas.bind("<B1-Motion>", self.on_grow)
        self.canvas.bind("<Double-1>", self.on_clear)
        self.canvas.bind("<ButtonRelease-3>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_stop)

    def on_start(self, event):
        if self.params['tag'] is 'crop':
            self.params['width'] = 3
        else:
            self.params['width'] = self.brush_size_bar.get()
        self.start_x, self.start_y = event.x, event.y
        self.drawn = None

    def on_grow(self, event):
        canvas = event.widget
        if self.drawn:
            canvas.delete(self.drawn)
        if 'tag' in self.params.keys():
            self.canvas.delete(self.params['tag'])
        shape = self.active_shape

        line = shape(self.start_x,
                     self.start_y,
                     event.x,
                     event.y,
                     **self.params)
        self.drawn = line

    def on_clear(self, event):
        event.widget.delete(self.params['tag'])

    def on_move(self, event):
        if self.drawn:
            canvas = event.widget
            diff_x, diff_y = event.x - self.start_x, event.y - self.start_y
            canvas.move(self.drawn, diff_x, diff_y)
            self.start_x, self.start_y = event.x, event.y

    def on_stop(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.calibration[self.params['tag']] = {'point1': (self.start_x, self.start_y),
                                                'point2': (self.end_x, self.end_y)}
        if self.params['tag'] == 'needle':
            self.calibration['needle']['width'] = self.params['width']
            if self.needle_found:
                self.mask_needle()
                if not self.masked:
                    self.masked = True
                    self.show_needle_button = tk.Button(self.editor_frame,
                                                        text='Show Masked Needle',
                                                        command=self.show_masked_needle)
                    self.show_needle_button.grid(row=0, column=6, sticky=tk.W)
        if self.params['tag'] == 'crop':
            self.crop_image()

    def set_point(self,
                  set_: str = 'min'):
        if set_ == 'min':
            x, y = point_math.get_closest_pt_to_center(self.x, self.y, self.start_x, self.start_y, self.end_x,
                                                       self.end_y)
            angle = point_math.angle_from_pts(self.current_val.x, self.current_val.y, x, y)
        else:
            x, y = point_math.get_further_pt_to_center(self.x, self.y, self.start_x, self.start_y, self.end_x,
                                                       self.end_y)
            angle = 0
        point = Point(x, y, angle)
        return point

    def draw_x(self, event):
        self.brush_size = self.brush_size_bar.get()
        self.start_x = event.x
        self.start_y = event.y
        self.canvas.create_text(self.start_x, self.start_y,
                                text='X')

    def use_mark_needle(self):

        self.activate_button(self.mark_needle_button)
        self.params = dict(tag='needle', fill='white')
        self.active_shape = self.canvas.create_line
        self.draw_shape()
        self.needle_found = True

    def use_mark_min_line(self):
        self.activate_button(self.mark_min_line_button)
        self.params = dict(tag='min_line', fill='red')
        self.active_shape = self.canvas.create_line
        self.draw_shape()
        self.min_line_found = True

    def use_mark_max_line(self):
        self.activate_button(self.mark_max_line_button)
        self.params = dict(tag='max_line', fill='green')
        self.active_shape = self.canvas.create_line
        self.draw_shape()
        self.max_line_found = True

    def use_crop(self):
        auto = self.check_auto_crop()
        if auto:
            self.crop_image()
        else:
            self.canvas.delete('crop')
            self.activate_button(self.crop_button)
            self.params = dict(tag='crop', outline='black')
            self.active_shape = self.canvas.create_rectangle
            self.draw_shape()

    def crop_image(self):
        """
        Crop the image to the rectangle defined by the start and end points, force square size for training
        :return:
        """
        auto = self.check_auto_crop()
        if auto:
            self.reset_crop_detection()
            self.auto_find_circles(param_source='auto')
            r_param = self.auto_crop_radius_bar.get()
            square_side = int(self.r * 2.2)
            x_origin, y_origin = point_math.point_pos(self.x, self.y, self.r * r_param, 225)
            x, y = x_origin, y_origin
            x_diff, y_diff = x_origin + square_side, y_origin + square_side
            res = True
        else:
            diff_x, diff_y = self.end_x - self.start_x, self.end_y - self.start_y
            diff = max(diff_x, diff_y)
            x, y = self.start_x, self.start_y
            x_diff, y_diff = self.start_x + diff, self.start_y + diff
            res = False
        cropped_image = self.proc_cv[y:y_diff, x:x_diff]
        cropped_image = cv2.resize(cropped_image, env.TRAIN_IMAGE_SIZE)
        cv2.imwrite(self.train_image_path, cropped_image)

        self.open_img(env.TRAIN_IMAGE_NAME,
                      keep_window=True,
                      resize=res)
        self.cropped = True
        self.define_frame_button = tk.Button(self.gauge_detection_steps_frame, text='Find Lines',
                                             command=self.create_define_frame)
        self.define_frame_button.grid(row=1, column=0)
        self.detect_circles_button = tk.Button(self.gauge_detection_steps_frame, text=self.d_text, command=self.d_func)
        self.detect_circles_button.grid(row=11, column=0)
        if auto:
            self.reset_crop_detection(2, 2, 2)
            self.auto_find_circles(param_source='auto')
            self.reset_crop_detection()

    def update_from_cv(self):
        """
        updates self.proc_im from self.proc_cv. For purposes of applying cv2 functions on proc_cv
        :return:
        """
        self.proc_im = self._cv_to_imagetk(self.proc_cv)

    def show_image(self,
                   image: ImageTk.PhotoImage = None):
        if image is None:
            self.canvas.itemconfig(self.canvas_image, image=self.proc_im, tag='canvas_image')
        else:
            self.canvas.itemconfig(self.canvas_image, image=image, tag='canvas_image')

    def reset_canvas(self):
        for shape in self.tags:
            self.canvas.delete(shape)

    def reset_all(self,
                  except_for: list = []):
        """
        Reset all the parameters to their default values - view original image
        :return:
        """
        cond_dict = {'needle': self.needle_found,
                     'min_line': self.min_line_found,
                     'max_line': self.max_line_found,
                     'cropped': self.cropped,
                     'test_rotation': self.test_rotation}
        for item in cond_dict.keys():
            if item not in except_for:
                cond_dict[item] = False

        self.open_img(self.calibration_image)

    def edit_top_frame(self):  # TODO: implement editing
        """
        Image editing top frame
        :return:
        """
        pass

    def man_find_circles(self,
                         tweak: bool = True,
                         param_source: str = 'man'):
        def _create_circle(self, x, y, r, **kwargs):
            return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)

        tk.Canvas.create_circle = _create_circle
        for tag in ['center', 'gauge', 'no_circles']:
            self.canvas.delete(tag)
        auto_crop = True if param_source == 'auto' else False
        if auto_crop:
            min_r = self.crop_detection['min_r']
            max_r = self.crop_detection['max_r']
            min_d = self.crop_detection['min_d']
        else:
            min_r = self.cd['min_radius'].get()
            max_r = self.cd['max_radius'].get()
            min_d = self.cd['min_dist'].get()

        circles = cd.find_circles(self.proc_cv,
                                  min_r,
                                  max_r,
                                  min_d)
        if not circles:
            self.canvas.create_text(150, 150,
                                    text='No circles found',
                                    font=('Arial', 20),
                                    fill='red',
                                    tag='no_circles')
            if tweak:
                self.tweak_circle_params(min_r, max_r, min_d, auto_crop)
            return False
        else:
            x, y, r = circles
            self.x, self.y, self.r = x, y, r
            self.circles_found = True
            self.tweak_circle_params(min_r, max_r, min_d, auto_crop)
            self.canvas.create_circle(x, y, r, tag='gauge', width=3, outline='green')
            self.canvas.create_circle(x, y, 5, fill='red', tag='center')
            self.update_from_cv()
            self.show_image()
            return x, y, r

    def tweak_circle_params(self,
                            min_r,
                            max_r,
                            min_d,  # TODO implement or remove
                            auto_crop: bool = False):
        if min_r < self.proc_cv.shape[0] / 2:
            if auto_crop:
                self.crop_detection['min_r'] = min_r + 1
            else:
                if self.cd['min_radius'] is not None:
                    self.cd['min_radius'].set(min_r + 1)
        if max_r > 1:
            if auto_crop:
                self.crop_detection['max_r'] = max_r - 1
                if self.cd['max_radius'] is not None:
                    self.cd['max_radius'].set(min_r + 1)

    def auto_find_circles(self,
                          param_source: str = 'man'):
        circles = False
        self.canvas.delete('circles')
        timeout = time.time() + 10
        while not circles:
            if time.time() > timeout:
                break
            circles = self.man_find_circles(tweak=True,
                                            param_source=param_source)
        if circles:
            self.x, self.y, self.r = circles

    def reset_crop_detection(self,
                             reset_all: bool = False,
                             m_r: int = 3,
                             x_r: int = 2,
                             d: int = 4):
        if not reset_all:
            self.crop_detection['min_r'] = self.w // m_r
            self.crop_detection['max_r'] = self.w // x_r
            self.crop_detection['min_d'] = self.w // d
        if reset_all:
            self.crop_detection['min_r'] = 1
            self.crop_detection['max_r'] = self.w // 3
            self.crop_detection['min_d'] = 1

    def set_circle_parameters(self):
        self.calibration['center'] = self.x, self.y
        self.calibration['radius'] = self.r
        for key in self.cd.keys():
            self.calibration[key] = self.cd[key].get()
        self.circle_frame.destroy()

    # Parameters methods

    def set_text_parameters(self):
        if self.find_needle_error_check():
            return
        for key in self.tp.keys():
            try:
                if key is not 'units':
                    self.calibration[key] = float(self.tp[key].get())  # TODO seperate strings and floats
                else:
                    self.calibration[key] = self.tp[key].get()
            except ValueError:
                self.params_entry_error_prompt()
                return
        if self.needle_found:
            self.create_needle_rotation_frame()
        self.params_set = True
        self.define_frame.destroy()

    def save_calibration_data(self):
        if self.crop_error_check():
            return
        if self.find_needle_error_check():
            return
        if self.mask_error_check():
            return
        if self.params_set:
            return
        for tag in self.tags:
            self.calibration[tag] = self.canvas.find_withtag(tag)
        self.calibration['width'] = self.w
        self.calibration['height'] = self.h
        xmlr.dict_to_xml(self.calibration, self.xml_file)
        typer.secho('Saved parameters to {}'.format(self.xml_file), fg='green')

    # Image detection/text input frame methods

    def create_define_frame(self):
        if self.crop_error_check():
            return
        self.define_frame = tk.Toplevel(master=self.window,
                                        width=200,
                                        height=300)
        dense = tk.TOP
        self.brush_size_bar = tk.Scale(self.define_frame, from_=1, to=50, orient=tk.HORIZONTAL, label='Line width')
        self.brush_size_bar.set(self.brush_size)
        self.brush_size_bar.pack(side=dense)
        self.mark_needle_button = tk.Button(self.define_frame, text='Find Needle', command=self.use_mark_needle)
        self.mark_needle_button.pack(side=dense)
        self.mark_min_line_button = tk.Button(self.define_frame, text='Mark Min Line', command=self.use_mark_min_line)
        self.mark_min_line_button.pack(side=dense)
        self.mark_max_line_button = tk.Button(self.define_frame, text='Mark Max Line', command=self.use_mark_max_line)
        self.mark_max_line_button.pack(side=dense)
        self.circles_text_input_frame()
        self.use_mark_needle()

    def characters_text_input_frame(self):
        """
        Creates a frame with text input for digit gauges.
        :return:
        """
        pass

    def characters_detection_frame(self):
        """
        Creates a frame with a button to detect characters and set parameters.
        :return:
        """
        pass

    def circles_text_input_frame(self):
        """
        Creates a frame with text input for analog round gauges.
        :return:
        """
        for key in self.tp:
            self.tp[key] = tk.Entry(self.define_frame,
                                    width=10,
                                    name=key,
                                    bd=5)
            temp = tk.Label(self.define_frame,
                            text=key)
            temp.pack(side=tk.TOP)
            self.tp[key].pack(side=tk.TOP)
        ttk.Button(self.define_frame, text='Set', command=self.set_text_parameters).pack(side=tk.TOP)

    def circle_detection_frame(self):
        """
        Creates a frame with a button to detect circles and set parameters.
        :return:
        """
        self.circle_frame = tk.Toplevel(self.window)
        self.circle_frame.title("Circle Detection")
        self.circle_frame.geometry('500x300')
        tk.Label(self.circle_frame, text="Circle Detection").pack(side=tk.TOP)
        tk.Button(self.circle_frame, text="Detect", command=self.man_find_circles).pack(side=tk.TOP)
        tk.Button(self.circle_frame, text="Auto Detect", command=self.auto_find_circles).pack(side=tk.TOP)
        tk.Label(self.circle_frame, text='Auto detect tweaks minimum radius and max radius \n'
                                         'If not found, try tweaking other parameters.').pack(side=tk.TOP)
        tk.Button(self.circle_frame, text="Set", command=self.set_circle_parameters).pack(side=tk.BOTTOM)

        for key in self.cd.keys():
            self.cd[key] = tk.Scale(self.circle_frame,
                                    from_=1,
                                    to=self.proc_cv.shape[0],
                                    orient=tk.VERTICAL,
                                    label=key,
                                    command=self.man_find_circles)
            self.cd[key].pack(side=tk.LEFT, anchor=tk.CENTER)
        self.cd['min_radius'].set(self.w / 3)
        self.cd['max_radius'].set(1)
        self.cd['min_dist'].set(1)

    def check_auto_crop(self):
        crop_func = self.crop_mode.get()
        if crop_func == 'Manual Crop':
            return False
        elif crop_func == 'Auto Crop':
            return True

    def mask_needle(self):
        if self.crop_error_check():
            return
        if self.find_needle_error_check():
            return
        self.mask = np.zeros(self.proc_cv.shape[:2], dtype=np.uint8)
        cv2.line(self.mask,
                 self.calibration['needle']['point1'],
                 self.calibration['needle']['point2'],
                 (255, 0, 0),
                 thickness=self.calibration['needle']['width'])
        self.needle_image = cv2.bitwise_and(self.proc_cv, self.proc_cv, mask=self.mask)
        cv2.imwrite(self.needle_image_path, self.needle_image)
        self.train_image = cv2.inpaint(self.proc_cv, self.mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(self.train_image_path, self.train_image)

    def show_masked_needle(self):
        self.cv_to_image(self.needle_image, show=True)
        self.cv_to_image(self.train_image, show=True)

    def create_needle_rotation_frame(self):

        if self.tweak_frame is None:
            self.tweak_frame = tk.Frame(self.window, width=width, height=height)
            self.tweak_frame.pack(side=tk.TOP, fill=tk.BOTH)

        if self.needle_rotation_scale is None:
            self.needle_rotation_scale = tk.Scale(self.tweak_frame,
                                                  from_=-360,
                                                  to=360,
                                                  orient=tk.HORIZONTAL,
                                                  label='Rotate Needle',
                                                  resolution=0.001,
                                                  command=self.rotate_needle,
                                                  length=self.w / 2)
        # make scale wider
        self.needle_rotation_scale.pack(side=tk.RIGHT)
        tk.Button(self.tweak_frame, text='Set Max', command=self.set_max_needle_rotation, name='max').pack(side=tk.LEFT)
        tk.Button(self.tweak_frame, text='Set Min', command=self.set_min_needle_rotation, name='set').pack(side=tk.LEFT)
        tk.Button(self.tweak_frame, text='Re-Config', command=self.reconfig_needle, name='re_config').pack(side=tk.LEFT)
        self.tweak_frame.pack(side=tk.TOP)

    def rotate_needle(self,
                      event,
                      angle=None):
        if self.crop_error_check():
            return
        if self.find_needle_error_check():
            return
        self.canvas.itemconfig('needle', state=tk.HIDDEN)
        if angle is None:
            angle = self.needle_rotation_scale.get()
        rotated = ie.rotate_needle(train_image=self.train_image,
                                   needle_image=self.needle_image,
                                   needle_center=(self.x, self.y),
                                   needle_angle=angle)
        self.rotated_im = self._cv_to_imagetk(rotated)
        if self.test_rotation and self.read_value_button is not None:
            self.update_reading()
        self.show_image(self.rotated_im)

    def set_max_needle_rotation(self):
        self.calibration['needle']['max_angle'] = self.needle_rotation_scale.get()
        if self.set_min_first_error_check():
            return
        else:
            self.calibration['needle']['max_cur_angle_diff'] = 360 - self.calibration['needle']['min_angle']
        self.needle_rotation_scale.config(from_=self.calibration['needle']['max_angle'])
        self.test_needle_rotation_button = tk.Button(self.gauge_detection_steps_frame,
                                                     text='Test Rotation',
                                                     command=self.create_test_needle_rotation_frame)
        self.test_needle_rotation_button.grid(row=3, column=0, sticky=tk.W)

    def set_min_needle_rotation(self):
        self.calibration['needle']['min_angle'] = self.needle_rotation_scale.get()
        self.needle_rotation_scale.config(to=self.calibration['needle']['min_angle'])

    def reconfig_needle(self):
        self.canvas.itemconfig('needle', state=tk.NORMAL)
        self.needle_rotation_scale.config(from_=-360, to=360)
        self.show_image(self.proc_im)

    def create_test_needle_rotation_frame(self):  # TODO scale values are wrong, fix angle abs diff
        self.needle_image = ie.rotate_needle(needle_image=self.needle_image,
                                             train_image=self.train_image,
                                             needle_angle=self.calibration['needle']['min_angle'],
                                             needle_center=(self.x, self.y),
                                             return_needle=True)
        self.angle_diff = -(self.calibration['needle']['min_angle'] - self.calibration['needle']['max_angle'])
        self.value_diff = self.calibration['max_value'] - self.calibration['min_value']
        self.needle_rotation_scale.config(from_=0, to=self.angle_diff)
        self.calibration['angle_diff'] = self.angle_diff
        self.calibration['value_diff'] = self.value_diff
        self.needle_rotation_scale.set(0)
        self.current_reading = self.calibration['min_value']
        reading_text = f'Current reading: {self.current_reading}'
        self.read_value_button = tk.Button(self.gauge_detection_steps_frame, text=reading_text)
        self.read_value_button.grid(row=12, column=0, sticky=tk.W)
        self.test_rotation = True

    def update_reading(self):
        self.current_reading = self.calibration['min_value'] + abs(self.needle_rotation_scale.get() *
                                                                   self.value_diff
                                                                   / self.angle_diff)
        self.read_value_button.config(text='Current Reading: \n '
                                           '{:.2f} {}'.format(self.current_reading,
                                                              self.calibration['units']))
        self.read_value_button.grid(row=13, column=0)

    def ask_value(self, value_name):
        value = tk.simpledialog.askstring(title='Enter ' + value_name,
                                          prompt='Enter ' + value_name + ':')
        if value is not None:
            try:
                value = float(value)
                return value
            except ValueError:
                mb.showerror('Error', 'Invalid value')
                return None
        else:
            return None

    # Create massage box for errors
    def circle_error_check(self):
        if not self.circles_found:
            message = "Detect circles first"
            mb.showerror('Error', message)
            return True
        return False

    def crop_error_check(self):
        if not self.cropped:
            message = "Please crop the first image before proceeding."
            mb.showerror('Error', message)
            return True
        return False

    def find_needle_error_check(self):
        if not self.needle_found:
            message = "Please find the needle first."
            mb.showerror('Error', message)
            return True

    def mask_error_check(self):
        if self.needle_image is None:
            message = "Please mask the needle first."
            mb.showerror('Error', message)

    @staticmethod
    def params_entry_error_prompt():
        message = "Please set the parameters first."
        mb.showerror('Error', message)

    def set_min_first_error_check(self):
        try:
            self.calibration['needle']['min_angle'] is None
        except KeyError:
            massage = "Please set the min first."
            mb.showerror('Error', massage)
            return True
        return False

    def find_needle_error_check(self):
        if not any([self.needle_found]):
            message = "Please find the needle before proceeding."
            mb.showerror('Error', message)
            return True
        return False

    # Run methods

    def run(self):
        """
        Runs the main loop.
        :return:
        """
        self.window.mainloop()
        return self.calibration


dev_app = Calibrator(calibration_image=env.DEV_CALIBRATION_PHOTO,
                     camera_id=env.DEV_CAM,
                     index=env.DEV_GAUGE).run()
