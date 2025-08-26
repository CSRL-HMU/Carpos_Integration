#!/usr/bin/env python3
import os
from datetime import datetime
import rospy
from std_msgs.msg import Bool, Float32MultiArray, String
import numpy as np
import time

try:
    from scipy.io import savemat
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class SmrLoggerNode:
    def __init__(self):
        # Params
        self.dir_out = rospy.get_param("~output_dir", os.path.expanduser("~/catkin_ws/src/logger/logs"))
        self.topic_start = rospy.get_param("~topic_start", "/command_topic")
        self.topic_confirm = rospy.get_param("~topic_confirm", "/ok_redetect")
        self.topic_completed = rospy.get_param("~topic_completed", "/stop_logger")
        self.topic_sensors = rospy.get_param("~topic_sensors", "/fsr_data")

        os.makedirs(self.dir_out, exist_ok=True)

        # State
        self.run_counter = 0
        self.logging_active = False
        self.samples = []
        self.sample_times = []
        self.t_start = None
        self.t_confirm = None
        self.t_done = None
        self.expecting = "start"

        # Subscribers
        rospy.Subscriber(self.topic_start, String, self.on_start, queue_size=1)
        rospy.Subscriber(self.topic_confirm, String, self.on_confirm, queue_size=1)
        rospy.Subscriber(self.topic_completed, Bool, self.on_completed, queue_size=1)
        rospy.Subscriber(self.topic_sensors, Float32MultiArray, self.on_sensors, queue_size=100)

        rospy.loginfo("SMR Logger ready. Waiting for %s", self.topic_start)

    # --- Callbacks ---
    def on_start(self, msg: String):
        rospy.loginfo("entered on_start callback ")
        if not msg.data:
            return
        if self.expecting != "start":
            rospy.loginfo("exiting on_start callback ")
            return
        if self.logging_active:
            rospy.logwarn("Received start while already logging; ignoring.")
            return
        self.run_counter += 1
        self.samples = []
        self.sample_times = []
        #self.t_start = rospy.Time.now().to_sec()
        self.t_start = time.time()
        self.t_confirm = None
        self.t_done = None
        self.logging_active = True
        
        self.expecting = "confirm"
        rospy.loginfo("Run %d: START at %.3f", self.run_counter, self.t_start)
        print("started logging")

    def on_confirm(self, msg: String):
        rospy.loginfo("entered on_confirm callback ")
        if not msg.data or not self.logging_active:
            rospy.loginfo("return 1 ")
            return
        if self.expecting != "confirm":
            rospy.loginfo("return 2")
            return
        #if self.t_confirm is None and (time.time() - self.t_start > 4):
            #self.t_confirm = rospy.Time.now().to_sec()
        self.t_confirm = time.time()
        rospy.loginfo("Run %d: CONFIRM at %.3f", self.run_counter, self.t_confirm)

    def on_completed(self, msg: Bool):
        if not msg.data or not self.logging_active:
            return
        #self.t_done = rospy.Time.now().to_sec()
        self.t_done = time.time()
        rospy.loginfo("Run %d: COMPLETED at %.3f", self.run_counter, self.t_done)
        self.logging_active = False
        self.expecting = "start"
        self.save_run()

    def on_sensors(self, msg: Float32MultiArray):
        if not self.logging_active:
            return
        if len(msg.data) < 2:
            rospy.logwarn_throttle(5.0, "Expected 2 ints, got len=%d", len(msg.data))
            return
        # t_now = rospy.Time.now().to_sec()
        t_now = time.time()
        self.samples.append([int(msg.data[0]), int(msg.data[1])])
        self.sample_times.append(t_now)

    # --- Save ---
    def save_run(self):
        vision_t = self.t_confirm - self.t_start if (self.t_start and self.t_confirm) else np.nan
        pick_t = self.t_done - self.t_confirm if (self.t_confirm and self.t_done) else np.nan

        ts = datetime.now().strftime("%d.%m_%H.%M")
        fname = f"{self.run_counter}_{ts}.mat"
        fpath = os.path.join(self.dir_out, fname)

        arr = np.array(self.samples, dtype=np.int32)
        t_arr = np.array(self.sample_times, dtype=np.float64)

        payload = {
            "counter": self.run_counter,
            "start_time": self.t_start or np.nan,
            "confirm_time": self.t_confirm or np.nan,
            "done_time": self.t_done or np.nan,
            "vision_t": vision_t,
            "pick_t": pick_t,
            "sensor_samples": arr,
            "sensor_sample_times": t_arr,
        }

        try:
            if not _HAS_SCIPY:
                raise RuntimeError("scipy not installed")
            savemat(fpath, payload, do_compression=True)
            rospy.loginfo("Saved run %d to %s", self.run_counter, fpath)
        except Exception as e:
            npz_path = fpath.replace(".mat", ".npz")
            np.savez_compressed(npz_path, **payload)
            rospy.logwarn("Could not save .mat (%s). Wrote NPZ: %s", e, npz_path)


def main():
    rospy.init_node("logger")
    SmrLoggerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
