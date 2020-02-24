import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.player import AudioPlayer


class PlotDiar:
    """
    A viewer of segmentation
    """

    def __init__(
        self,
        wav=None,
        title="",
        gui=False,
        pick=False,
        vgrid=False,
        size=(18, 9),
        maxx=0,
        df_tracks=None,
        df_speakers=None,
    ):
        self.rect_picked = None
        self.rect_color = (0.0, 0.6, 1.0, 1.0)  # '#0099FF'
        self.rect_selected_color = (0.75, 0.75, 0, 1.0)  # 'y'
        self.df_tracks = df_tracks
        self.df_speakers = df_speakers
        plt.rcParams["keymap.fullscreen"] = "ctrl+f"
        plt.rcParams["keymap.home"] = ""
        plt.rcParams["keymap.back"] = ""
        plt.rcParams["keymap.forward"] = ""
        plt.rcParams["keymap.pan"] = ""
        plt.rcParams["keymap.zoom"] = "ctrl+z"
        plt.rcParams["keymap.quit"] = "ctrl+q"
        plt.rcParams["keymap.grid"] = ""
        plt.rcParams["keymap.yscale"] = ""
        plt.rcParams["keymap.xscale"] = ""
        plt.rcParams["keymap.all_axes"] = ""
        plt.rcParams["toolbar"] = "None"
        plt.rcParams["keymap.save"] = "ctrl+s"
        # plt.rcParams.update({'font.family': 'courrier'})

        self.pick = pick
        self.gui = gui
        self.vgrid = vgrid
        self.fig = plt.figure(figsize=size, facecolor="white", tight_layout=True)
        self.plot = plt
        self.title = title

        self.ax = self.fig.add_subplot(1, 1, 1)
        cids = list()
        if self.gui:
            cids.append(
                self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
            )
            cids.append(
                self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            )
            if pick:
                cids.append(self.fig.canvas.mpl_connect("pick_event", self._on_pick))
        self.height = 5
        self.maxx = maxx
        self.maxy = 0
        self.end_play = 0
        self.wav = wav
        self.audio = None
        if self.wav is not None and self.gui:
            self.audio = AudioPlayer(wav)
            self.timer = self.fig.canvas.new_timer(interval=10)
            self.timer.add_callback(self._update_timeline)
            self.timer.start()

        self.timeline = self.ax.plot([0, 0], [0, 0], color="r")[-1]
        self.time_stamp = list()
        self.time_stamp_idx = 0

    def _draw_timeline(self, t):
        """
        Draw the timeline a position t
        :param t: in second, a float
        """
        min, max = self.ax.get_ylim()
        self.timeline.set_data([t, t], [min, max])
        self._draw_info(t)

    def _update_timeline(self):
        """
        Update the timeline given the position in the audio player
        """
        if self.audio is not None and self.audio.playing():
            t = self.audio.time()
            min, max = self.ax.get_xlim()
            if t > self.end_play and self.rect_picked is not None:
                self.audio.pause()
                self.end_play = self.maxx
            self._draw_timeline(t)
            if t > max:
                self._dec_right(min, max)
            if t < min:
                self._dec_left(min, max)
            self.fig.canvas.draw()

    def _draw_info(self, t):
        """
        Draw information on segment and timestamp
        :param t: a float
        :return:
        """
        ch = "time:{:s} ({:.3f} sec {:d} frame)".format(self._hms(t), t, int(t * 100))
        ch2 = "\n\n\n"
        if self.rect_picked is not None:
            s = self.rect_picked.get_x()
            w = self.rect_picked.get_width()
            e = s + w
            ch2 = "segment  start: {:20s} ({:8.2f} sec {:8d} frame)\n".format(
                self._hms(s), s, int(s * 100)
            )
            ch2 += "segment   stop: {:20s} ({:8.2f} sec {:8d} frame)\n".format(
                self._hms(e), e, int(e * 100)
            )
            ch2 += "segment length: {:20s} ({:8.2f} sec {:8d} frame)\n".format(
                self._hms(w), w, int(w * 100)
            )

        plt.xlabel(ch + "\n" + ch2)

    def draw(self):
        """
        Draw the segmentation
        """
        y = 4
        labels_pos = []
        labels = []
        # self.df_tracks['amplitude'].plot(ax=self.ax)
        self.ax.plot(self.df_tracks["time"], self.df_tracks["amplitude"])
        for i, row in self.df_speakers.iterrows():
            self.ax.plot([row["start"], row["end"]], [row["id"], row["id"]], "-r")

        if self.gui:
            plt.xlim([0, min(600, self.maxx)])
        else:
            plt.xlim([0, self.maxx])

        plt.ylim([-y, y])
        plt.yticks(labels_pos, labels)
        self.maxy = y
        self.end_play = self.maxx

        plt.title(self.title + " (last frame: " + str(self.maxx) + ")")
        if self.gui:
            self._draw_info(0)
        plt.tight_layout()
        self.time_stamp = list(set(self.time_stamp))
        self.time_stamp.sort()

        if self.vgrid:
            for x in self.time_stamp:
                self.ax.plot([x, x], [0, self.maxy], linestyle=":", color="#AAAAAA")

    def _dec_right(self, min, max):
        """
        Move right
        :param min: a float
        :param max: a float
        """
        dec = (max - min) // 10
        diff = max - min
        if max + dec <= self.maxx:
            # print('** 1 ', min, max, dec)
            plt.xlim(min + dec, max + dec)
        else:
            # print('** 2 ', min, max, dec, diff)
            plt.xlim(self.maxx - diff, self.maxx)

    def _dec_left(self, min, max):
        """
        Move left
        :param min: a float
        :param max: a float
        """
        dec = (max - min) // 10
        diff = max - min
        if min - dec >= 0:
            plt.xlim(min - dec, max - dec)
        else:
            plt.xlim(0, diff)

    def _on_keypress(self, event):
        """
        manage the keypress event
        :param event: a key event
        """
        hmin, hmax = self.ax.get_xlim()
        diff = hmax - hmin
        if event.key == "ctrl++" or event.key == "ctrl+=":
            plt.xlim(hmin * 1.5, hmax * 1.5)
        elif event.key == "ctrl+-":
            plt.xlim(hmin / 1.5, hmax / 1.5)
        elif event.key == "escape":
            plt.xlim(0, self.maxx)
            plt.ylim(0, self.maxy)
        elif event.key == "right":
            self._dec_right(hmin, hmax)
        elif event.key == "left":
            self._dec_left(hmin, hmax)
        elif event.key == "ctrl+right":
            plt.xlim(self.maxx - diff, self.maxx)
        elif event.key == "ctrl+left":
            plt.xlim(0, diff)
        elif event.key == "alt+right":
            self.time_stamp_idx = min(len(self.time_stamp) - 1, self.time_stamp_idx + 1)
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(self.time_stamp[self.time_stamp_idx])
            self._draw_timeline(self.time_stamp[self.time_stamp_idx])
        elif event.key == "alt+left":
            self.time_stamp_idx = max(0, self.time_stamp_idx - 1)
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(self.time_stamp[self.time_stamp_idx])
            self._draw_timeline(self.time_stamp[self.time_stamp_idx])
        elif event.key is None and self.audio is not None:
            self.audio.play()
        elif event.key == " " and self.audio is not None:
            if self.audio.playing():
                self.audio.pause()
            else:
                self.audio.play()

        self.fig.canvas.draw()

    def _on_click(self, event):
        """
        manage the mouse event
        :param event: a mouse event
        """
        if event.xdata is not None and self.rect_picked is None:
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(event.xdata)
            self._draw_timeline(event.xdata)
            self.fig.canvas.draw()

    def _on_pick(self, event):
        """
        manage the selection of a segment
        :param event: a picked event
        """
        if isinstance(event.artist, Rectangle) and event.mouseevent.dblclick:
            print("on pick dbclick")
            rect = event.artist
            x, y = rect.get_xy()
            w = rect.get_width()
            c = rect.get_fc()
            if self.rect_picked is not None:
                if self._colors_are_equal(c, self.rect_selected_color):
                    rect.set_color(self.rect_color)
                    self.rect_picked = None
                    self.end_play = self.maxx
                else:
                    self.rect_picked.set_color(self.rect_color)
                    rect.set_color(self.rect_selected_color)
                    self.rect_picked = rect
                    if self.audio is not None:
                        self.audio.pause()
                        self.audio.seek(x)
                    self.time_stamp_idx = self.time_stamp.index(x)
                    self.end_play = x + w
                    self._draw_timeline(x)
            else:
                rect.set_color(self.rect_selected_color)
                self.rect_picked = rect

                if self.audio is not None:
                    self.audio.pause()
                    self.audio.seek(x)
                self.time_stamp_idx = self.time_stamp.index(x)
                self.end_play = x + w
                self._draw_timeline(x)

            self.fig.canvas.draw()

    @classmethod
    def _colors_are_equal(cls, c1, c2):
        """
        Compare two colors
        """
        for i in range(4):
            if c1[i] != c2[i]:
                return False
        return True

    @classmethod
    def _hms(cls, s):
        """
        conversion of seconds into hours, minutes and secondes
        :param s:
        :return: int, int, float
        """
        h = int(s) // 3600
        s %= 3600
        m = int(s) // 60
        s %= 60
        return "{:d}:{:d}:{:.2f}".format(h, m, s)
