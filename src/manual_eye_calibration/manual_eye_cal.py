import mysql.connector

from functools import partial
from numpy import array, searchsorted, unique, stack, asarray, arange, argmin
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Qt5Agg")
db = mysql.connector.connect(
    user='xper_rw',
    password='up2nite',
    host='172.30.6.59',
    database='test',
)

start_time = "2025-03-18 15:50:00"
stop_time = "2025-03-18 18:00:00"
eye = "right"
iscan_id = eye + "Iscan"
calibration_degree = 10

slide_on_delay = 20  # I.e. only display data starting 20 ms AFTER slide on time

_DEFAULT_FIX_VOLTS = array([
    [-1, 0],  # left
    [0, -1],  # down
    [0, 0],   # center
    [0, 1],   # up
    [1, 0],   # right
])

marker_size = 3
marker_alpha = 1
figsize = 25, 25


def get_voltage_data(db, iscan_id, start_time, stop_time):
    q = """
        SELECT tstamp,
               CAST(ExtractValue(msg, '/EyeDeviceMessage/volt/x') AS DOUBLE),
               CAST(ExtractValue(msg, '/EyeDeviceMessage/volt/y') AS DOUBLE)
        FROM   BehMsgEye 
        WHERE  type = "EyeDeviceMessage"
               AND tstamp BETWEEN (unix_timestamp(%s) * 1000000) 
                              AND (unix_timestamp(%s) * 1000000) 
               AND ExtractValue(msg, '/EyeDeviceMessage/id') = %s
        ORDER BY tstamp ASC
    """

    c = db.cursor()
    c.execute(q, (start_time, stop_time, iscan_id))
    eye_tstamp, volt_h, volt_v = map(array, zip(*c.fetchall()))
    c.close()
    volts = stack([volt_h, volt_v], axis=-1)
    return eye_tstamp, volts


def get_calibration_data(db, start_time, stop_time):
    q = """
        SELECT tstamp,
               CAST(ExtractValue(msg, '/CalibrationPointSetupMessage/fixationPosition/x') AS DOUBLE),
               CAST(ExtractValue(msg, '/CalibrationPointSetupMessage/fixationPosition/y') AS DOUBLE)
        FROM   BehMsg 
        WHERE  type = "CalibrationPointSetup"
               AND tstamp BETWEEN (unix_timestamp(%s) * 1000000) 
                              AND (unix_timestamp(%s) * 1000000) 
        ORDER BY tstamp ASC
    """
    c = db.cursor()
    c.execute(q, (start_time, stop_time))
    calpoint_tstamp, calpoint_x, calpoint_y = map(array, zip(*c.fetchall()))
    c.close()
    calpoints = stack([calpoint_x, calpoint_y], axis=1)
    return calpoint_tstamp, calpoints


def get_slide_times(db, start_time, stop_time):
    q = """
        WITH 
            on_times AS (SELECT tstamp FROM BehMsg WHERE type = "SlideOn"),
            off_times AS (SELECT tstamp FROM BehMsg WHERE type = "SlideOff")
        SELECT  on_times.tstamp slide_on, 
                MIN(off_times.tstamp) slide_off
        FROM 	on_times INNER JOIN off_times ON (on_times.tstamp < off_times.tstamp)
        WHERE 	on_times.tstamp BETWEEN (unix_timestamp(%s) * 1000000) 
                                      AND (unix_timestamp(%s) * 1000000) 
        GROUP BY 1
        ORDER BY on_times.tstamp ASC
    """
    c = db.cursor()
    c.execute(q, (start_time, stop_time))
    t_slide_on, t_slide_off = map(array, zip(*c.fetchall()))
    c.close()
    return t_slide_on, t_slide_off


def get_eye_cal_data(db, iscan_id, start_time, stop_time):
    """
    Returns
        volts: (n_samples, 2) array of voltage data (H, V)
        unq_cal_pos: (5, 2) array of calibration points (x, y)
        eye_cal_pos_idx: (n_samples,) array of indices into the 5 different calibration positions
    """
    t_eye, volts = get_voltage_data(db, iscan_id, start_time, stop_time)
    t_cal, calpoints = get_calibration_data(db, start_time, stop_time)

    # Assign eye data to calibration trial assuming eye timestamps immediately follow calibration timestamps
    cal_trial_idx = searchsorted(t_cal, t_eye) - 1
    keep = cal_trial_idx >= 0  # Some eye data precedes calibration timestamps
    t_eye, cal_trial_idx, volts = t_eye[keep], cal_trial_idx[keep], volts[keep]

    # Ditto slide on/off trial times
    t_on, t_off = get_slide_times(db, start_time, stop_time)
    slide_times_idx = searchsorted(t_on, t_eye) - 1
    keep = (
        (t_eye >= (t_on[slide_times_idx] + slide_on_delay * 1000))
        & (t_eye < t_off[slide_times_idx])
    )
    t_eye, cal_trial_idx, volts = t_eye[keep], cal_trial_idx[keep], volts[keep]

    unq_cal_pos, cal_pos_idx = unique(calpoints, axis=0, return_inverse=True)
    eye_cal_pos_idx = cal_pos_idx[cal_trial_idx]

    return volts, unq_cal_pos, eye_cal_pos_idx


def volt2degree(volts, sx, sy, center=None):
    if center is not None:
        volts = volts - center.reshape(1, 2)

    # degree.setY(((volt.getY() - eyeZero.getY()) * Sxh - (volt.getX() - eyeZero.getX()) * Sxv) / (Syv * Sxh - Syh * Sxv));
    # degree.setX(((volt.getX() - eyeZero.getX()) - degree.getY() * Syh) / Sxh);
    y = (volts[:, 1] * sx[0] - volts[:, 0] * sx[1]) / (sy[1] * sx[0] - sy[0] * sx[1])
    x = (volts[:, 0] - y * sy[0]) / sx[0]
    return stack([x, y], axis=1)


def plot_volt_data(ax, volts, unq_cal_pos, eye_cal_pos_idx):
    ax.set_xlabel('Volts')
    ax.set_ylabel('Volts')
    scatter = ax.scatter(volts[:, 0], volts[:, 1], s=marker_size, c=eye_cal_pos_idx, alpha=marker_alpha)
    items, _ = scatter.legend_elements(alpha=1)
    labels = [f'({x}, {y})' for (x, y) in unq_cal_pos]
    legend = ax.legend(items, labels, loc="lower left", title="Calibration position")
    ax.add_artist(legend)


def plot_degree_data(ax, degrees, eye_cal_pos_idx):
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Degrees')
    scatter = ax.scatter(degrees[:, 0], degrees[:, 1], s=marker_size, c=eye_cal_pos_idx, alpha=marker_alpha)
    return scatter


def calc_parameters(points):
    left, down, center, up, right = points
    sx_r = (right - center) / calibration_degree
    sx_l = (left - center) / -calibration_degree
    sy_u = (up - center) / calibration_degree
    sy_d = (down - center) / -calibration_degree

    sx = (sx_r + sx_l) / 2
    sy = (sy_u + sy_d) / 2
    return center, sx, sy


class DraggableMarkers:
    def __init__(
            self, fig, ax, points, eps=0.1,
            marker_size=24, marker_style='x',
            alpha=1.0, edge_color='black', callback=None
    ):
        self.fig = fig
        self.ax = ax
        self.scatter = ax.scatter(
            *points.T, s=marker_size, c=arange(len(points)),
            marker=marker_style, alpha=alpha, edgecolors=edge_color)
        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('button_release_event', self._on_release)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.eps = eps
        self._selected = None
        self.callback = callback

    def points(self):
        """Returns an (n, 2) array of current marker positions"""
        return self.scatter.get_offsets().data

    def _update_plot(self, event):
        self.points()[self._selected, :] = [event.xdata, event.ydata]
        self.fig.canvas.draw_idle()

    def _closest_point(self, event):
        pt = event.xdata, event.ydata
        pts = self.scatter.get_offsets().data
        dist = ((pts - pt) ** 2).sum(axis=1)
        i = argmin(dist)
        if dist[i] < self.eps ** 2:
            return i
        return None

    def _on_click(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            if (i := self._closest_point(event)) is not None:
                self._selected = i

    def _on_release(self, event):
        if event.button == 1 and event.inaxes == self.ax and self._selected is not None:
            self._update_plot(event)
            if self.callback:
                self.callback(self)
            self._selected = None

    def _on_motion(self, event):
        if self._selected is None or event.xdata is None or event.ydata is None:
            return
        self._update_plot(event)


def on_update_points(fix_markers: DraggableMarkers, volts, degree_ax, eye_cal_pos_idx):
    points = fix_markers.points()
    center, sx, sy = calc_parameters(points)
    print(f"""\n
INSERT INTO SystemVar (name, arr_ind, tstamp, val) VALUES 
    ("xper_{eye}_iscan_eye_zero", 0, unix_timestamp() * 1000000, {center[0]}),
    ("xper_{eye}_iscan_eye_zero", 1, unix_timestamp() * 1000000, {center[1]}),
    ("xper_{eye}_iscan_mapping_algorithm_parameter", 0, unix_timestamp() * 1000000, {sx[0]}),
    ("xper_{eye}_iscan_mapping_algorithm_parameter", 1, unix_timestamp() * 1000000, {sx[1]}),
    ("xper_{eye}_iscan_mapping_algorithm_parameter", 2, unix_timestamp() * 1000000, {sy[0]}),
    ("xper_{eye}_iscan_mapping_algorithm_parameter", 3, unix_timestamp() * 1000000, {sy[1]})
    """)

    lines = [f'    {p.tolist()}  # {dir}' for p, dir in zip(points, ('left', 'down', 'center', 'up', 'right'))]
    print(f"_DEFAULT_FIX_POINTS = array([\n" + '\n'.join(lines) + '\n])')

    degrees = volt2degree(volts, sx, sy, center=center)
    degree_ax.clear()
    plot_degree_data(degree_ax, degrees, eye_cal_pos_idx)


def main():
    fig, axs = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    volts, unq_cal_pos, eye_cal_pos_idx = get_eye_cal_data(
        db, iscan_id, start_time, stop_time)
    plot_volt_data(axs[0, 0], volts, unq_cal_pos, eye_cal_pos_idx)

    center, sx, sy = calc_parameters(_DEFAULT_FIX_VOLTS)
    degrees = volt2degree(volts, sx, sy, center=center)
    plot_degree_data(axs[0, 1], degrees, eye_cal_pos_idx)
    callback = partial(on_update_points, degree_ax=axs[0, 1], volts=volts, eye_cal_pos_idx=eye_cal_pos_idx)

    fix_markers = DraggableMarkers(
        fig=fig,
        ax=axs[0, 0],
        points=asarray(_DEFAULT_FIX_VOLTS),
        callback=callback,
        alpha=0.5,
        marker_size=128,
        marker_style='D',
    )

    plt.show()


if __name__ == '__main__':
    main()

