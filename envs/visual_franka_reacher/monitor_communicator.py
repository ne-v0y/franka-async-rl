import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import os
import signal
import logging
from senseact.communicator import Communicator
from senseact.sharedbuffer import SharedBuffer

class MonitorCommunicator(Communicator):

    def __init__(self, target_type='reaching', width=160, height=90, radius=7):
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, width), ylim=(0, height))
        self.target = plt.Circle((0, 0), radius, color='red')
        self.ax.add_patch(self.target)
        plt.axis('off')
        self.radius = radius
        self.width = width
        self.height = height
        self.target_type=target_type
        actuator_args = {
            'array_len': 1,
            'array_type': 'd',
            'np_array_type': 'd',
        }
        super(MonitorCommunicator, self).__init__(
            use_sensor=False,
            use_actuator=True,
            sensor_args={},
            actuator_args=actuator_args
        )
        self.reset()
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset(self):
        if self.target_type == 'static':
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = 0, 0
        if self.target_type == 'reaching':
            x, y = np.random.random(2)
            self.target.set_center(
                (self.radius + x * (self.width - 2 * self.radius),
                 self.radius + y * (self.height - 2 * self.radius))
            )
            self.velocity_x, self.velocity_y = 0, 0
        elif self.target_type == 'tracking':
            self.target.set_center((self.width / 2, self.height / 2))
            self.velocity_x, self.velocity_y = np.random.random(2) - 0.5
            velocity = np.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            self.velocity_x /= velocity
            self.velocity_y /= velocity

    def run(self):
        """Starts sensor and actuator related threads/processes if they exist."""
        # catching SIGTERM from terminate() call so that we can close thread
        # on this spawn-process
        # signal.signal(signal.SIGTERM, self._close)
        self._sensor_running = True
        self._actuator_running = True
        while self._sensor_running or self._actuator_running:
            # if parent pid is no longer the same (1 on Linux if re-parented to init), then
            # main process has been closed
            if os.getppid() != self._parent_pid:
                logging.info("Main environment process has been shutdown, closing communicator.")
                self._close()
                return

            if (self._sensor_thread is not None and not self._sensor_thread.is_alive()) or \
                    (self._actuator_thread is not None and not self._actuator_thread.is_alive()):
                logging.error("Sensor/Actuator thread has exited, closing communicator.")
                self._close()
                return
            self._actuator_handler()
            # time.sleep(1)

    def _sensor_handler(self):
        raise NotImplementedError()

    def _actuator_handler(self):
        if self.actuator_buffer.updated():
            content = self.actuator_buffer.read_update()
            # print(content)
            self.reset()
        x, y = self.target.get_center()
        if x + self.velocity_x + self.radius > self.width or \
           x + self.velocity_x - self.radius < 0:
            self.velocity_x = -self.velocity_x
        if y + self.velocity_y + self.radius > self.height or \
           y + self.velocity_y - self.radius < 0:
            self.velocity_y = -self.velocity_y
        self.target.set_center((x + self.velocity_x, y + self.velocity_y))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)


if __name__ == '__main__':
   p = MonitorCommunicator(target_type='tracking')
   p.start()
   while True:
       time.sleep(5)
       p.actuator_buffer.write(0)





