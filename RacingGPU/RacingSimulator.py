import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import scipy as sp


class Racetrack:
    def __init__(self,
                 filepath: str,
                 x_start: float,
                 y_start: float,
                 starting_angle: float,
                 points_per_line: int = 300,
                 grid_resolution: int = 2000,
                 offset: int = 20):
        self.filepath = filepath

        self.x_start = x_start
        self.y_start = y_start
        self.starting_angle = starting_angle

        self.grid_resolution = grid_resolution
        self.points_per_line = points_per_line
        self.offset = offset

        self.grid = np.zeros((self.grid_resolution, self.grid_resolution))
        self.pieces = []
        self.points = None
        self.scaling_factor = None

        self.load_racetrack()

    def load_racetrack(self):
        with open(self.filepath, 'r') as file:
            pieces = json.load(file)['pieces']

        for piece in pieces:
            pos = np.array(piece['pos'])
            pos[1] *= -1  # this is necessary since the racetrack loader has a flipped y-axis
            rotation_angle = -1 * piece['rotation']  # same here -> rotations have to be transformed aswell

            match piece['type']:
                case 'rect_obstacle':
                    width, height = piece['params']['w'], piece['params']['h']

                    top = np.stack([np.linspace(-width / 2, width / 2, self.points_per_line),
                                    np.full(self.points_per_line, height / 2)], axis=1)
                    bottom = np.stack([np.linspace(-width / 2, width / 2, self.points_per_line),
                                       np.full(self.points_per_line, -height / 2)], axis=1)
                    left = np.stack([np.full(self.points_per_line, -width / 2),
                                     np.linspace(-height / 2, height / 2, self.points_per_line)], axis=1)
                    right = np.stack([np.full(self.points_per_line, width / 2),
                                      np.linspace(-height / 2, height / 2, self.points_per_line)], axis=1)

                    points = np.vstack([top, bottom, left, right])

                case 'circle_obstacle':
                    radius = piece['params']['radius']

                    angles = np.linspace(0, 2 * np.pi, self.points_per_line, endpoint=False)
                    points = np.stack([radius * np.cos(angles),
                                       radius * np.sin(angles)], axis=1)

                case 'arc':
                    radius, span_deg = piece['params']['radius'], piece['params']['span_deg']

                    angles = np.linspace(0, np.deg2rad(span_deg), self.points_per_line)
                    points = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

                case 'line':
                    length = piece['params']['length']
                    x = np.linspace(0, length, self.points_per_line)
                    points = np.stack([x, np.zeros_like(x)], axis=1)

            theta_rad = np.deg2rad(rotation_angle)
            points = points @ np.array(
                [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]).T

            points += pos

            if type(self.points) == np.ndarray:
                self.points = np.vstack([self.points, points])
            else:
                self.points = points

        min_x, min_y = np.min(self.points[:, 0]), np.min(self.points[:, 1])
        self.points += np.abs(np.array([min_x, min_y])) + np.array([self.offset, self.offset])

        self.scaling_factor = min((self.grid_resolution - 2 * self.offset) / np.max(self.points[:, 0]),
                                  (self.grid_resolution - 2 * self.offset) / np.max(self.points[:, 1]))
        self.points *= self.scaling_factor
        self.x_start *= self.scaling_factor
        self.y_start *= self.scaling_factor

        idx = np.floor(self.points).astype(int)
        mask = ((1 <= idx[:, 0]) & (idx[:, 0] < self.grid_resolution - 1) & (1 <= idx[:, 1]) & (
                    idx[:, 1] < self.grid_resolution - 1))

        self.grid[idx[mask, 0], idx[mask, 1]] = 1

        # mark more cells to make sure the lines are closed
        kernel = np.ones((3, 3))
        self.grid = np.clip(sp.signal.convolve2d(self.grid, kernel, mode='same', boundary='fill', fillvalue=1), 0, 1)

    def plot_track(self):
        fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=self.grid_resolution)

        ax.scatter(self.points[:, 0], self.points[:, 1], s=0.75)
        ax.set_axis_off()
        ax.scatter(self.x_start, self.y_start, marker='x', c='red')

        return fig


class Simulator:
    def __init__(self,
                 racetrack_ls: list,
                 racer_length: int = 50,
                 racer_width: int = 30,
                 max_velocity: float = 100,
                 acceleration: float = 20,
                 dt: float = 0.1):
        self.track_ls = racetrack_ls
        self.dt = dt
        self.time_elapsed = 0.0
        self.length = racer_length
        self.width = racer_width
        self.max_velocity = max_velocity
        self.acceleration = acceleration

        self.velocity = 0.0
        self.steering_angle = 0.0

        self.racetrack = self.track_ls[np.random.randint(0, len(self.track_ls))]
        self.x = self.racetrack.x_start
        self.y = self.racetrack.y_start
        self.angle = self.racetrack.starting_angle

    def reset(self):
        self.racetrack = self.track_ls[np.random.randint(0, len(self.track_ls))]
        self.x = self.racetrack.x_start
        self.y = self.racetrack.y_start
        self.angle = self.racetrack.starting_angle
        self.velocity = 0.0
        return self.sense()

    def get_racer_outline(self):
        c = np.cos(np.deg2rad(self.angle))
        s = np.sin(np.deg2rad(self.angle))
        dx = self.length / 2
        dy = self.width / 2

        corners = [(self.x + dx * c - dy * s, self.y + dx * s + dy * c),
                   (self.x + dx * c + dy * s, self.y + dx * s - dy * c),
                   (self.x - dx * c + dy * s, self.y - dx * s - dy * c),
                   (self.x - dx * c - dy * s, self.y - dx * s + dy * c)]
        return corners

    def step(self, velocity, steering_angle):
        if velocity < self.velocity:
            self.velocity -= self.acceleration * self.dt
        else:
            self.velocity += self.acceleration * self.dt

        if self.velocity < 0:
            self.velocity = 0
        elif self.max_velocity < self.velocity:
            self.velocity = self.max_velocity

        self.angle += steering_angle
        self.x += self.velocity * self.dt * np.cos(np.deg2rad(self.angle))
        self.y += self.velocity * self.dt * np.sin(np.deg2rad(self.angle))

        collision = self.check_collision()
        reward = self.velocity * 0.1
        done = False

        if self.x - (self.racetrack.x_start-50) < 10 and -100 <self.y - self.racetrack.y_start < 100:
            reward = 1000 / (self.time_elapsed + 1e-6)
            done = True

        if self.x - (self.racetrack.x_start-30) < 10 and -100 <self.y - self.racetrack.y_start < 100:
            reward = -100
            done = True

        if collision:
            reward = -100
            done = True

        obs = self.sense()
        return obs, reward, done

    def check_collision(self):
        corners = np.array(self.get_racer_outline())

        x_min = max(0, int(np.floor(corners[:, 0].min())))
        x_max = min(self.racetrack.grid_resolution, int(np.ceil(corners[:, 0].max())) + 1)
        y_min = max(0, int(np.floor(corners[:, 1].min())))
        y_max = min(self.racetrack.grid_resolution, int(np.ceil(corners[:, 1].max())) + 1)

        if x_min >= x_max or y_min >= y_max:
            return False

        xs, ys = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')

        occupied = self.racetrack.grid[xs, ys] == 1
        if not np.any(occupied):
            return False
        else:
            return True

    def sense(self, sensor_opening_angle: float = 30, sensor_pixels: int = 60, max_distance: float = 450):
        readings = []
        max_distance *= self.racetrack.scaling_factor

        for a in np.linspace(-sensor_opening_angle / 2, sensor_opening_angle / 2, sensor_pixels):
            rad = np.deg2rad(self.angle + a)

            line_x = self.x + np.linspace(0, max_distance, 500) * np.cos(rad)
            line_y = self.y + np.linspace(0, max_distance, 500) * np.sin(rad)

            gx = np.floor(line_x).astype(int)
            gy = np.floor(line_y).astype(int)

            mask = (gx >= 0) & (gx < self.racetrack.grid_resolution) & (gy >= 0) & (gy < self.racetrack.grid_resolution)

            collision_idx = np.where(self.racetrack.grid[gx[mask], gy[mask]] == 1)[0]
            if collision_idx.size > 0:
                first_idx = collision_idx[0]
                distance = np.hypot(line_x[first_idx] - self.x, line_y[first_idx] - self.y)
                readings.append(distance / max_distance)
            else:
                readings.append(1)  # when no collision is detected

        return np.array(readings)


class LiveVisualizer:
    def __init__(self,
                 simulator,
                 sensor_opening_angle=30,
                 sensor_pixels=60,
                 max_distance=450,
                 refresh_interval=1):

        self.sim = simulator
        self.track = simulator.racetrack

        self.sensor_opening_angle = sensor_opening_angle
        self.sensor_pixels = sensor_pixels
        self.max_distance = max_distance * self.track.scaling_factor

        self.refresh_interval = refresh_interval
        self.step_counter = 0

        plt.ion()
        self._init_figure()

    def _init_figure(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[20, 1])

        self.ax_track = self.fig.add_subplot(gs[0])
        self.ax_track.scatter(self.track.points[:, 0], self.track.points[:, 1], s=0.75)
        self.ax_track.set_title('Racetrack')
        self.ax_track.set_axis_off()

        self.car_patch = Polygon(self.sim.get_racer_outline(),
                                 closed=True,
                                 facecolor='red',
                                 edgecolor='black',
                                 alpha=0.8)
        self.ax_track.add_patch(self.car_patch)

        self.sensor_lines = LineCollection([], colors='cyan', linewidths=1)
        self.ax_track.add_collection(self.sensor_lines)

        self.sensor_hits = self.ax_track.scatter([], [], c='yellow', s=8)

        self.ax_sensor = self.fig.add_subplot(gs[1])
        self.sensor_bars = self.ax_sensor.bar(np.arange(self.sensor_pixels),
                                              np.ones(self.sensor_pixels))
        self.ax_sensor.set_ylim(0, 1)
        self.ax_sensor.set_title('Sensor Readings')

        self.info_text = self.ax_sensor.text(0.02, 0.95, '', transform=self.ax_sensor.transAxes,
                                             verticalalignment='top')

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.show()

    def update(self, obs, reward, done):
        self.step_counter += 1
        if self.step_counter % self.refresh_interval != 0:
            return

        corners = self.sim.get_racer_outline()
        self.car_patch.set_xy(corners)

        rays, hits = self._compute_sensor_geometry(obs)
        self.sensor_lines.set_segments(rays)

        if hits:
            xs, ys = zip(*hits)
            self.sensor_hits.set_offsets(np.c_[xs, ys])
        else:
            self.sensor_hits.set_offsets(np.empty((0, 2)))

        for bar, v in zip(self.sensor_bars, obs):
            bar.set_height(v)

        self.info_text.set_text(f'x={self.sim.x:.1f}\n'
                                f'y={self.sim.y:.1f}\n'
                                f'angle={self.sim.angle:.1f}\n'
                                f'reward={reward:.2f}\n'
                                f'done={done}')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _compute_sensor_geometry(self, obs):
        rays = []
        hits = []

        angles = np.linspace(-self.sensor_opening_angle / 2, self.sensor_opening_angle / 2, self.sensor_pixels)

        for a, d_normed in zip(angles, obs):
            rad = np.deg2rad(self.sim.angle + a)
            start = np.array([self.sim.x, self.sim.y])

            if d_normed < 1.0:
                d = d_normed * self.max_distance
                end = start + d * np.array([np.cos(rad), np.sin(rad)])
                hits.append(end)
            else:
                end = start + self.max_distance * np.array([np.cos(rad), np.sin(rad)])

            rays.append([start, end])

        return rays, hits


if __name__ == '__main__':
    test_track = Racetrack(os.path.join(os.getcwd(), 'track1.json'), 500, 100, 0)
    test_simulator = Simulator([test_track])
    test_visualizer = LiveVisualizer(test_simulator)

    for i in range(0, 500):
        obs, reward, done = test_simulator.step(10 + 10 * i, 0.2 * i)
        test_visualizer.update(obs, reward, done)
        time.sleep(0.3)
        if done:
            test_simulator.reset()