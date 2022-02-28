import cv2
import os
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


class Calibrator:
    def __init__(self,
                 calibration_image: str,
                 index: str,
                 camera_id: str = None):
        """
        A class that creates a GUI for the calibration process. This is the main class of the calibrator,
        Alone it does not do anything, uses as an archi-class for the AnalogCalibrator and the DigitalCalibrator.
        :param calibration_image:
        :param index:
        :param camera_id:
        """

        # Outer variables
        self.camera_id = camera_id
        self.index = index
        self.calibration_image = calibration_image

        # Inner variables
        # Paths
        self.directory, self.xml_file = env.dir_file_from_camera_gauge(camera_id, index)
        self.calibration_image_path = Path(self.directory).joinpath(calibration_image).as_posix()
        self.train_image_path = Path(self.directory).joinpath(env.TRAIN_IMAGE_NAME).as_posix()
        self.current_image_path = None

        # Images
        self.orig_cv = None
        self.proc_cv = None
        self.orig_im = None
        self.proc_im = None

        # Calibration data and parameters
        self.draw_params = {}  # placeholder for params of image editing
        self.calibration = {}
        self.current_reading = 0

        # BOOL variables for error checking
        self.error_flags = {'cropped': False,
                            'parameters': False}

        # Image variables
        self.w, self.h = None, None

        # Root Window settings
        self.window = tk.Tk()
        self.create_main_window()
        self.canvas = None
        self.canvas_image = None

        # Menubar Frame
        self.menubar = tk.Menu(self.window)
        self.create_menu_bar_frame()

        # Main toolbar Frame # TODO: add editor functions
        self.toolbar_frame = tk.Frame(self.window)
        self.crop_button = None
        self.brush_size_bar = None
        self.brush_size = 10  # default brush size
        self.create_toolbar_frame()

        # Image Frame settings
        self.image_frame = tk.Frame(master=self.window, width=env.WINDOW_SIZE[0], height=env.WINDOW_SIZE[1])
        self.image_frame.configure(background='gray')

        # Gauge specific steps
        self.gauge_steps_frame = None
        self.create_gauge_steps_frame()

        # Image editing variables
        self.active_button = None
        self.active_shape = None

        # Location Variables for drawing and locating items
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.drawn = None
        self.color = 'red'
        self.draw = None

        # Specific Buttons and top frames containers
        self.buttons = {}
        self.top_frames = {}

    def create_main_window(self):
        """
        Creates the main window of the calibrator.
        :return:
        """
        self.window.title("Calibrator App")
        self.window.resizable(width=True, height=True)
        self.window.configure(background='#ffffff')

    def create_menu_bar_frame(self):
        """
        Creates the menubar frame of the calibrator. Each subclass of the calibrator will have its
        own menubar changes done later
        :return:
        """
        file = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='File', menu=file)
        file.add_command(label='Change Calibration Image', command=self.change_calibration_image)
        file.add_command(label='Save to XML', command=self.save_calibration_data)
        file.add_command(label='Exit', command=self.window.destroy)
        self.menubar.add_cascade(label='Help', command=lambda: None)  # TODO: add help menu images for workflows
        self.window.config(menu=self.menubar)

    def create_toolbar_frame(self):
        """
        Creates the toolbar frame of the calibrator. Each subclass of the calibrator will have its
        :return:
        """
        self.toolbar_frame = tk.Frame(self.window)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.BOTH)
        tk.Button(self.toolbar_frame, text='Reset', command=self.reset_to_start).pack(side=tk.LEFT)
        tk.Button(self.toolbar_frame, text='Edit', command=self.create_image_edit_frame).pack(side=tk.LEFT)
        self.crop_button = tk.Button(self.toolbar_frame, text="Crop", command=self.use_crop)
        self.brush_size_bar = tk.Scale(self.toolbar_frame, from_=1, to=50, orient=tk.HORIZONTAL, label='Line width')
        self.brush_size_bar.set(self.brush_size)
        self.crop_button.pack(side=tk.LEFT)
        self.brush_size_bar.pack(side=tk.RIGHT)

    def create_image_edit_frame(self):  # TODO: implement color options, perspective correction, etc.
        """
        Creates the image editing frame of the calibrator. Save calibration data to calibration data
        :return:
        """
        pass

    def create_gauge_steps_frame(self):
        """
        Create the generic frame for gauge calibration steps
        :return:
        """
        self.gauge_steps_frame = tk.Frame(self.window)
        self.gauge_steps_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        tk.Label(self.gauge_steps_frame, text='Steps').pack(side=tk.TOP)

    def create_canvas(self):
        """
        Creates the canvas of the calibrator.
        :return:
        """
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

    # Image loading methods
    def update_main_image(self,
                          image: str = None,
                          keep_window: bool = False,
                          resize: bool = True):
        """
        Updates the main image of the calibrator. This will also update the canvas and the image frame,
        and the relevant size variables.
        :param image: string - image name (without path)
        :param keep_window:
        :param resize:
        :return:
        """
        path = Path(self.directory).joinpath(image).as_posix()
        for top in self.top_frames.values():
            if top is not None:
                top.destroy()
        self.current_image_path = path
        self.orig_cv = cv2.imread(self.current_image_path)
        if self.orig_cv is None:
            raise Exception("Image not found / not readable")
        self.proc_cv = self.orig_cv.copy()
        if resize:
            self.proc_cv = self.resize_cv(self.proc_cv)
        else:
            self.h = self.proc_cv.shape[0]
            self.w = self.proc_cv.shape[1]
        self.orig_im = ie.cv_to_imagetk(self.proc_cv)
        self.proc_im = copy(self.orig_im)
        self.create_canvas()
        w_w, w_h = int(self.w * 1.4), int(self.h * 1.1)
        if not keep_window:
            self.window.geometry(f'{w_w}x{w_h}')
        self.show_image()

    def change_calibration_image(self):
        """
        Opens the image from the file dialog. This will change the original calibration image.
        :return:
        """
        # Prompt warning
        tk.messagebox.showwarning('Warning', 'This will reset all calibration data and replace calibration image path')
        # Open file dialog
        self.calibration_image_path = fd.askopenfilename(initialdir=env.CALIBRATION_PATH,
                                                         title="Select Calibration Image",
                                                         filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.calibration_image = os.path.basename(self.calibration_image_path)
        self.update_main_image(self.calibration_image)

    def resize_cv(self,
                  image: np.ndarray):
        """
        Resizes the image to the size of the window.
        :param image: cv2 image (ndarray)
        :return: resized cv2 image (ndarray)
        """
        h = image.shape[0]
        w = image.shape[1]
        self.w = int(w * env.WINDOW_SIZE[0] / w)
        self.h = int(h * env.WINDOW_SIZE[1] / w)
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return image

    def draw_shape(self):
        """
        Draws the shape on the canvas. This is used to make the drawing "flexible" and allow less
        duplicated code for different shapes drawing.
        :return: None
        """
        self.canvas.bind("<ButtonPress-1>", self.on_start)
        self.canvas.bind("<B1-Motion>", self.on_grow)
        self.canvas.bind("<Double-1>", self.on_clear)
        self.canvas.bind("<ButtonRelease-3>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_stop)

    def on_start(self, event):
        """
        This function is called when the user clicks on the canvas.
        :param event: tkinter event
        :return: None
        """
        if self.draw_params['tag'] is 'crop':
            self.draw_params['width'] = 3
        else:
            self.draw_params['width'] = self.brush_size_bar.get()
        self.start_x, self.start_y = event.x, event.y
        self.drawn = None

    def on_grow(self, event):
        """
        This function is called when the user moves the mouse on the canvas.
        :param event:
        :return:
        """
        canvas = event.widget
        if self.drawn:
            canvas.delete(self.drawn)
        if 'tag' in self.draw_params.keys():
            self.canvas.delete(self.draw_params['tag'])
        shape = self.active_shape
        line = shape(self.start_x,
                     self.start_y,
                     event.x,
                     event.y,
                     **self.draw_params)
        self.drawn = line

    def on_clear(self, event):
        """
        This function is called when the user double-clicks on the canvas.
        :param event: tkinter event
        :return: None
        """
        event.widget.delete(self.draw_params['tag'])

    def on_move(self, event):
        """
        This function is called when the user releases the right mouse button on the canvas.
        :param event: tkinter event
        :return: None
        """
        if self.drawn:
            canvas = event.widget
            diff_x, diff_y = event.x - self.start_x, event.y - self.start_y
            canvas.move(self.drawn, diff_x, diff_y)
            self.start_x, self.start_y = event.x, event.y

    def on_stop(self, event):
        """
        This function is called when the user releases the left mouse button on the canvas.
        :param event: tkinter event
        :return: None
        """
        self.end_x, self.end_y = event.x, event.y
        if self.draw_params['tag'] == 'crop':
            self.crop_image()
        self.stop_actions()

    def stop_actions(self):
        """
        This function is called when the user stops drawing on the canvas. Specific implementation
        is needed for subclasses
        :return: None
        """
        pass

    def use_crop(self):
        """
        This function is called when the user clicks on the crop button.
        :return:
        """
        self.canvas.delete('crop')
        self.draw_params = dict(tag='crop', outline='black')
        self.active_shape = self.canvas.create_rectangle
        self.draw_shape()

    def crop_image(self):
        """
        Crop the image to the rectangle defined by the start and end points, force square size for training
        :return:
        """
        diff_x, diff_y = self.end_x - self.start_x, self.end_y - self.start_y
        diff = max(diff_x, diff_y)
        x, y = self.start_x, self.start_y
        x_diff, y_diff = self.start_x + diff, self.start_y + diff
        resize = False
        self.apply_crop(x, y, x_diff, y_diff, resize)

    def apply_crop(self,
                   x: int,
                   y: int,
                   x_diff: int,
                   y_diff: int,
                   resize: bool = False):
        """
        Crop the image to the rectangle defined by the start and end points, force square size for training,
        save the image and update the image frame
        :param x: int x coordinate of the top left corner of the rectangle
        :param y: int y coordinate of the top left corner of the rectangle
        :param x_diff: int x distance of the bottom right corner of the rectangle
        :param y_diff: int y distance of the bottom right corner of the rectangle
        :param resize: bool whether to resize the image to a square
        :return: None
        """
        cropped_image = self.proc_cv[y:y_diff, x:x_diff]
        cropped_image = cv2.resize(cropped_image, env.EDIT_IMAGE_SIZE)
        cv2.imwrite(self.train_image_path, cropped_image)
        self.update_main_image(image=env.TRAIN_IMAGE_NAME,
                               keep_window=True,
                               resize=resize)
        self.error_flags['cropped'] = True

    def update_from_cv(self):
        """
        updates self.proc_im from self.proc_cv. For purposes of applying cv2 functions on proc_cv
        :return:
        """
        self.proc_im = ie.cv_to_imagetk(self.proc_cv)

    def show_image(self,
                   image: ImageTk.PhotoImage = None):
        """
        Show the image in the main image frame. If image is None, show the image in self.proc_im
        :param image:
        :return: None
        """
        if image is None:
            self.canvas.itemconfig(self.canvas_image, image=self.proc_im, tag='canvas_image')
        else:
            self.canvas.itemconfig(self.canvas_image, image=image, tag='canvas_image')

    def reset_error_flags(self,
                          except_for: list or str = None):
        """
        Reset all the parameters to their default values - view original image
        :return:
        """
        if except_for is None:
            except_for = []
        elif isinstance(except_for, str):
            except_for = [except_for]
        for item in self.error_flags.keys():
            if item not in except_for:
                self.error_flags[item] = False

    def reset_to_start(self):  # TODO: add all reset fields
        msg = "Are you sure you want to reset the image? This will delete all edits and reset parameters."
        if tk.messagebox.askokcancel("Reset", msg):
            self.reset_error_flags()
            self.update_main_image(self.calibration_image)
        else:
            return

    def set_calibration_parameters(self):
        """
        Set the calibration parameters for the current image, specific method for Digital/Analog
        :return:
        """
        pass

    def save_calibration_data(self):
        """
        Save to XML the calibration data
        :return:
        """
        self.set_calibration_parameters()
        xmlr.dict_to_xml(self.calibration, self.xml_file, gauge=True)
        typer.secho('Saved parameters to {}'.format(self.xml_file), fg='green')

    def run(self):
        """
        Runs the main loop.
        :return:
        """
        self.window.mainloop()
        return self.calibration


class AnalogCalibrator(Calibrator):
    def __init__(self,
                 calibration_image: str,
                 index: str,
                 camera_id: str = None):
        super().__init__(calibration_image=calibration_image,
                         index=index,
                         camera_id=camera_id)

        # Paths
        self.needle_image_path = Path(self.directory).joinpath(env.NEEDLE_IMAGE_NAME).as_posix()

        # Images
        self.train_image = None
        self.needle_image = None
        self.centered_needle_image = None
        self.rotated_im = None
        self.mask = None

        # Calibration Data Parameters
        self.tags = ['crop',
                     'needle',
                     'max_line',
                     'min_line',
                     'center',
                     'gauge',
                     'no_circles']
        self.x = 0  # Gauge center x
        self.y = 0  # Gauge center y
        self.r = 0  # Gauge radius
        self.crop_detection = {'min_r': 0,  # parameters for auto crop detection
                               'max_r': 0,
                               'min_d': 0}
        self.cur_angle = 0
        self.step_angle = 0
        self.value_diff = 0
        self.angle_diff = 0

        # Crop Control Tools - GUI
        self.crop_mode = tk.StringVar(self.window)

        # Find Needle Frame (finding the needle and the calibration circle)
        self.find_needle_frame = None
        self.text_params = {'min_value': 0,
                            'max_value': 0,
                            'units': ''}

        # Needle rotation Frame
        self.needle_rotation_frame = None

        # Circle Detection and gauge text parameters Frame
        self.circle_frame = None
        self.circle_detection_scales = {key: None for key in ['min_r', 'max_r', 'min_d']}

        # Needle rotation frame
        self.needle_rotation_frame = None
        self.needle_rotation_scale = None

        # Add gauge steps
        self.add_toolbar_buttons()
        self.add_gauge_steps_frame()

        # Open Image and run
        if self.calibration_image is not None:
            self.update_main_image(image=calibration_image)

    # Specific Frames and Widgets for Analog Calibration
    def add_toolbar_buttons(self):
        """
        Add the buttons to the toolbar specific to the analog calibrator
        :return:
        """
        crop_options = ['Auto Crop', 'Manual Crop']
        self.crop_mode.set(crop_options[0])
        self.crop_button.config(command=self.set_crop_mode)
        self.buttons['crop_options'] = tk.OptionMenu(self.toolbar_frame,
                                                     self.crop_mode,
                                                     *crop_options)
        self.crop_button.pack(side=tk.LEFT)
        self.buttons['crop_options'].pack(side=tk.LEFT)

    def add_gauge_steps_frame(self):
        """
        Add the gauge steps frame to the calibrator
        :return:
        """

        button_width = 18
        self.buttons['circle_detection'] = tk.Button(self.gauge_steps_frame,
                                                     text='Circle Detection',
                                                     width=button_width,
                                                     command=self.create_circle_detection_frame)
        self.buttons['find_needle'] = tk.Button(self.gauge_steps_frame,
                                                text='Find Needle',
                                                width=button_width,
                                                command=self.create_find_needle_frame)
        self.buttons['show_masked'] = tk.Button(self.gauge_steps_frame,
                                                text='Show Masked Needle',
                                                width=button_width,
                                                command=self.show_masked_needle)

        self.buttons['circle_detection'].pack(side=tk.TOP)
        self.buttons['find_needle'].pack(side=tk.TOP)
        self.buttons['show_masked'].pack(side=tk.TOP)

    def create_find_needle_frame(self):
        """
        Creates the frame for the needle finding and calibration the gauge text parameters
        :return:
        """
        if self.find_needle_frame is None:
            self.find_needle_frame = tk.Toplevel(master=self.window,
                                                 width=200,
                                                 height=300)
        else:
            tk.messagebox.showinfo('Error', 'Find Needle Frame already exists')
            return
        dense = tk.TOP
        self.buttons['find_needle'] = tk.Button(self.find_needle_frame, text='Find Needle',
                                                command=self.use_mark_needle)
        self.buttons['find_needle'].pack(side=dense)
        for key in self.text_params:
            self.text_params[key] = tk.Entry(self.find_needle_frame,
                                             width=10,
                                             name=key,
                                             bd=5)
            temp = tk.Label(self.find_needle_frame,
                            text=key)
            temp.pack(side=tk.TOP)
            self.text_params[key].pack(side=tk.TOP)
        ttk.Button(self.find_needle_frame, text='Set', command=self.set_text_parameters).pack(side=tk.TOP)
        self.use_mark_needle()

    def create_circle_detection_frame(self):
        """
        Creates a frame with a button to detect circles and set parameters.
        :return:
        """
        self.circle_frame = tk.Toplevel(self.window)
        self.circle_frame.title("Circle Detection")
        self.circle_frame.geometry('500x300')
        tk.Label(self.circle_frame, text="Circle Detection").pack(side=tk.TOP)
        tk.Button(self.circle_frame, text="Auto Detect", command=self.auto_find_circles).pack(side=tk.TOP)
        tk.Label(self.circle_frame, text='Auto detect tweaks minimum radius and max radius \n'
                                         'If not found, try tweaking other parameters.').pack(side=tk.TOP)
        for key in self.circle_detection_scales.keys():
            self.circle_detection_scales[key] = tk.Scale(self.circle_frame,
                                                         from_=1,
                                                         to=self.proc_cv.shape[0],
                                                         orient=tk.VERTICAL,
                                                         label=key,
                                                         command=self.man_find_circles)
            self.circle_detection_scales[key].pack(side=tk.LEFT, anchor=tk.CENTER)
        self.circle_detection_scales['min_r'].set(self.w / 3)
        self.circle_detection_scales['max_r'].set(1)
        self.circle_detection_scales['min_d'].set(1)
        self.man_find_circles()

    # Specific stop_actions method for Analog Calibration
    def stop_actions(self):
        """
        Analog class specific stop method
        :return:
        """
        # Gather the needle coordinates (from the drawn line) to mask the train image (inpainting)
        self.calibration[self.draw_params['tag']] = {'point1': (self.start_x, self.start_y),
                                                     'point2': (self.end_x, self.end_y)}
        if self.draw_params['tag'] == 'needle':
            self.error_flags['needle_found'] = True
            self.calibration['needle']['width'] = self.draw_params['width']
            self.mask_needle()

    # Find needle methods
    def use_mark_needle(self):
        """
        Use the mark needle button to mark the needle
        :return: None
        """
        self.buttons['find_needle'].config(relief=tk.SUNKEN)
        self.draw_params = dict(tag='needle', fill='white')
        self.active_shape = self.canvas.create_line
        self.draw_shape()

    # Specific crop methods for Analog Calibration
    def set_crop_mode(self):
        """
        Change auto/manual crop
        :return:
        """
        if self.crop_mode.get() == 'Auto Crop':
            self.auto_circle_crop()
        elif self.crop_mode.get() == 'Manual Crop':
            self.use_crop()

    def auto_circle_crop(self):
        """
        Automatically crop the image from a circle
        :return:
        """
        self.reset_crop_detection()
        self.auto_find_circles(param_source='auto')
        r_param = 1.5  # arbitrary value that by trial and error seems to work
        square_side = int(self.r * 2.2)
        x_origin, y_origin = point_math.point_pos(self.x, self.y, self.r * r_param, 225)
        x, y = x_origin, y_origin
        x_diff, y_diff = x_origin + square_side, y_origin + square_side
        self.apply_crop(x, y, x_diff, y_diff)
        self.reset_crop_detection(min_r=2,
                                  max_r=2,
                                  min_d=2)
        self.auto_find_circles(param_source='auto')
        self.reset_crop_detection()

    def reset_crop_detection(self,
                             reset_all: bool = False,
                             min_r: int = 3,
                             max_r: int = 2,
                             min_d: int = 4):
        """
        Reset the crop detection parameters
        :param reset_all:
        :param min_r: minimum radius
        :param max_r: maximum radius
        :param min_d: minimum distance
        :return: None
        """
        if not reset_all:
            self.crop_detection['min_r'] = self.w // min_r
            self.crop_detection['max_r'] = self.w // max_r
            self.crop_detection['min_d'] = self.w // min_d
        if reset_all:
            self.crop_detection['min_r'] = 1
            self.crop_detection['max_r'] = self.w // 3
            self.crop_detection['min_d'] = 1

    # Circle Detection Methods
    def man_find_circles(self,
                         tweak: bool = True,
                         param_source: str = 'man'):
        """
        Manually detect circles with the current parameters
        :param tweak:
        :param param_source:
        :return:
        """
        tk.Canvas.create_circle = ie.create_circle
        for tag in ['center', 'gauge', 'no_circles']:
            self.canvas.delete(tag)
        auto_crop = True if param_source == 'auto' else False
        if auto_crop:
            min_r = self.crop_detection['min_r']
            max_r = self.crop_detection['max_r']
            min_d = self.crop_detection['min_d']
        else:
            min_r = self.circle_detection_scales['min_r'].get()
            max_r = self.circle_detection_scales['max_r'].get()
            min_d = self.circle_detection_scales['min_d'].get()

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
            self.error_flags['circles'] = True
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
                if self.circle_detection_scales['min_r'] is not None:
                    self.circle_detection_scales['min_r'].set(min_r + 1)
        if max_r > 1:
            if auto_crop:
                self.crop_detection['max_r'] = max_r - 1
                if self.circle_detection_scales['max_r'] is not None:
                    self.circle_detection_scales['max_r'].set(min_r + 1)

    def auto_find_circles(self,
                          param_source: str = 'man'):
        """
        Apply circle detection Automatically
        :param param_source: string: 'man' or 'auto'
        :return: None
        """
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

    def set_text_parameters(self):
        """
        Set the text parameters, gathered in circle detection frame.
        :return:  None
        """
        for key in self.text_params.keys():
            try:
                if key is not 'units':
                    self.calibration[key] = float(self.text_params[key].get())
                else:
                    self.calibration[key] = self.text_params[key].get()
            except ValueError:
                message = "Please set the parameters first."
                mb.showerror('Error', message)
                return
        self.error_flags['parameters_set'] = True
        self.canvas.unbind('<Button-1>')
        self.canvas.unbind('<Button-3>')
        self.buttons['find_needle'].config(relief=tk.RAISED)
        self.create_needle_rotation_frame()
        self.find_needle_frame.destroy()

    def mask_needle(self):
        """
        Creates two separate images: one with the needle and one without. The images
        are saved in the gauge's directory in a 'jpeg' format.
        :return: None
        """
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
        """
        Show the masked needle image and the train image in a separate window.
        :return: None
        """
        if self.mask_needle is None or self.train_image is None:
            message = "Please Find the needle and set parameters first"
            mb.showerror('Error', message)
            return
        ie.cv_to_image(self.needle_image, show=True)
        ie.cv_to_image(self.train_image, show=True)

    def create_needle_rotation_frame(self):
        """
        Create the needle rotation frame.
        :return:
        """
        reading_text = f'Value: {self.current_reading}'
        self.needle_rotation_frame = tk.Frame(self.window,
                                              width=env.WINDOW_SIZE[0],
                                              height=env.WINDOW_SIZE[1])
        self.needle_rotation_frame.pack(side=tk.BOTTOM, fill=tk.BOTH)
        self.needle_rotation_scale = tk.Scale(self.needle_rotation_frame,
                                              from_=-360,
                                              to=360,
                                              orient=tk.HORIZONTAL,
                                              label='Rotate Needle',
                                              resolution=0.0001,
                                              command=self.rotate_needle,
                                              length=self.w / 2)
        self.buttons['current_value'] = tk.Button(self.needle_rotation_frame,
                                                  text=reading_text)
        self.buttons['set_max'] = tk.Button(self.needle_rotation_frame,
                                            text='Set Max',
                                            command=self.set_max_needle_rotation,
                                            name='max')
        self.buttons['set_min'] = tk.Button(self.needle_rotation_frame,
                                            text='Set Min',
                                            command=self.set_min_needle_rotation,
                                            name='set')
        self.buttons['re_config'] = tk.Button(self.needle_rotation_frame,
                                              text='ReConfig',
                                              command=self.reconfig_needle,
                                              name='re_config')
        self.needle_rotation_scale.pack(side=tk.RIGHT)
        self.buttons['current_value'].pack(side=tk.LEFT)
        self.buttons['set_max'].pack(side=tk.LEFT)
        self.buttons['set_min'].pack(side=tk.LEFT)
        self.buttons['re_config'].pack(side=tk.LEFT)

    def rotate_needle(self,
                      event,
                      angle=None):
        """
        Rotate the needle image and show it in the canvas.
        :param event: tkinter event (not used)
        :param angle: angle relative to the needle's center
        :return: None
        """
        if not self.error_flags['needle_found']:
            message = "Please find the needle first and set gauge text parameters."
            mb.showerror('Error', message)
            return
        self.canvas.itemconfig('needle', state=tk.HIDDEN)
        if angle is None:
            angle = self.needle_rotation_scale.get()
        rotated = ie.rotate_needle(train_image=self.train_image,
                                   needle_image=self.needle_image,
                                   needle_center=(self.x, self.y),
                                   needle_angle=angle)
        self.rotated_im = ie.cv_to_imagetk(rotated)
        # if self.error_flags['parameters_set'] and self.error_flags['needle_found']:
        #     self.update_reading()
        self.show_image(self.rotated_im)

    def set_max_needle_rotation(self):
        """
        Set the max angle for needle rotation.
        :return: None
        """
        if self.flag_error_check('needle_found'):
            return
        elif self.flag_error_check('min_angle_set'):
            return
        else:
            self.calibration['needle']['max_angle'] = self.needle_rotation_scale.get()
            self.calibration['needle']['max_cur_angle_diff'] = 360 - self.calibration['needle']['min_angle']
            self.needle_rotation_scale.config(from_=self.calibration['needle']['max_angle'])
            self.test_reading()

    def set_min_needle_rotation(self):
        """
        Set the min angle for needle rotation.
        :return: None
        """
        if self.flag_error_check('needle_found'):
            return
        self.error_flags['min_angle_set'] = True
        self.calibration['needle']['min_angle'] = self.needle_rotation_scale.get()
        self.needle_rotation_scale.config(to=self.calibration['needle']['min_angle'])

    def reconfig_needle(self):
        """
        Reconfigure the needle image.
        :return:
        """
        if self.flag_error_check('needle_found'):
            return
        self.canvas.itemconfig('needle', state=tk.NORMAL)
        self.needle_rotation_scale.config(from_=-360, to=360)
        self.error_flags['min_angle_set'] = False
        self.show_image(self.proc_im)

    def test_reading(self):  # TODO scale values are wrong, fix angle abs diff
        """
        Use the parameters and needle location to get the reading.
        :return: None
        """
        self.angle_diff = abs(self.calibration['needle']['min_angle'] - self.calibration['needle']['max_angle'])
        self.value_diff = self.calibration['max_value'] - self.calibration['min_value']

        # All the bellow angles are relative to calibration image needle location
        max_angle = self.calibration['needle']['max_angle']
        min_angle = self.calibration['needle']['min_angle']
        center_angle = (abs(max_angle) + abs(min_angle)) / 2
        # Angles calculation to transform to absolute angle - relative to 0 at 90 degrees to needle center

        # TODO: add cases for needle rotation (after center, before center, etc.)
        beta = abs(min_angle)
        theta = abs(center_angle) - abs(beta)
        alpha = beta - theta
        min_angle = theta - alpha

        # Rotate needle to be aligned with the absolute center
        self.centered_needle_image = ie.rotate_image(self.needle_image,
                                                     angle=alpha,
                                                     pivot=(self.x, self.y))
        cv2.imwrite('centered_needle.jpg', self.centered_needle_image)

        self.needle_rotation_scale.config(from_=max_angle, to=min_angle)

        self.calibration['value_diff'] = self.value_diff
        self.calibration['angle_diff'] = self.angle_diff
        self.calibration['max_angle'] = max_angle
        self.calibration['min_angle'] = min_angle

    def update_reading(self):
        value = abs(self.needle_rotation_scale.get() * self.value_diff / self.angle_diff)
        self.current_reading = self.calibration['min_value'] + value
        self.buttons['read_value'].config(text='Current Reading: \n '
                                               '{:.2f} {}'.format(self.current_reading,
                                                                  self.calibration['units']))
        self.buttons['read_value'].grid(row=13, column=0)

    def set_calibration_parameters(self):
        """
        Set the calibration parameters, gathered in calibration frame. Specific for Analog gauge calibrator.
        :return:
        """
        self.calibration['center'] = (self.x, self.y)
        self.calibration['radius'] = self.r
        self.calibration['width'] = self.w
        self.calibration['height'] = self.h

    def flag_error_check(self,
                         flag_name: str):
        """
        Check if flag value is True, else prompt warning and return False.
        :return: True if error, False if no error
        """
        if flag_name == 'needle_found':
            message = "Please find the needle first and set gauge text parameters."
        elif flag_name == 'min_angle_set':
            message = "Please set the minimum needle rotation angle."
        else:
            message = "Some error occurred."
        if self.error_flags[flag_name]:
            final_message = message
            mb.showerror('Error', final_message)
            return True
        return False


class DigitalCalibrator(Calibrator):
    def __init__(self,
                 calibration_image: str,
                 index: str,
                 camera_id: str = None):
        super().__init__(calibration_image=calibration_image,
                         index=index,
                         camera_id=camera_id)
