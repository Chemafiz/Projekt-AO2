

class Menu:
    def __init__(self, window_size, background_color, fps, title):
        self.window_size = window_size
        self.background_color = background_color
        self.fps = fps
        self.title = title

        self.reset_button = 0
        self.predict_button = 0
        self.reset_button_pos = 0
        self.predict_button_pos = 0
