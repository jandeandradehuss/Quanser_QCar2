# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
vehicle_control.py

Skills acivity code for vehicle control lab guide.
Students will implement a vehicle speed and steering controller.
Please review Lab Guide - vehicle control PDF

"""
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg
from enum import Enum

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images


#================ Experiment Configuration ================
class FrameOfRef(Enum):
    REAR = 0
    CENTER = 1
    FRONT = 2

# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_ref = 2
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
pureControl = False
frameOfRef = FrameOfRef.CENTER
K_stanley = 0.7
#nodeSequence = [9,7,14,20,22,9]
nodeSequence = [9,13,19,17,16,17,20,22,9]

# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [0,2,-pi/2]
# Comment out the one that is not used 
calibrationPose = [0,0,-np.pi/2]
#calibrationPose = [0,2,-np.pi/2]



#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
    # initialPose = initialPose - [0.2125, 0, 0]
    print("This is the initial pose",initialPose)
else:
    initialPose = [0, 0, 0]
  

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate=False
else:
    calibrate =  'y' in input('do you want to recalibrate?(y/n)')

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki

        self.errOld = 0

        self.uOld = 0

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):

        errNew = v_ref - v            
        dErr = errNew - self.errOld

        uNew = self.kp * (dErr) + self.ki * dt * errNew + self.uOld

        self.errOld = errNew
        self.uOld = uNew

        return uNew
    
class PureSteeringController:

    def __init__(self, waypoints, gain=1, cyclic=True, carLength = 0.425, frameOfRef= FrameOfRef.CENTER):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpI = 0

        self.gain = gain
        self.cyclic = cyclic
        self.steeringAngle = 0

        self.p_ref = (0, 0)
        self.th_ref = 0
        self.carLength = carLength

        self.frameOfRef = frameOfRef

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):

        if self.frameOfRef == FrameOfRef.CENTER:

            # # 1. Get current and next waypoint
            wp1 = self.wp[:, np.mod(self.wpI, self.N-1)]

            goalWp = wp1

            distToGoalWp = np.linalg.norm(goalWp - p)

            if speed != 0:
                while distToGoalWp < ( 0.5 * speed):
                    self.wpI += 1

                    wp1 = self.wp[:, np.mod(self.wpI, self.N-1)]
                    goalWp = wp1
                    distToGoalWp = np.linalg.norm(goalWp - p)

            vectP = goalWp - p        
            ksi = np.arctan2(vectP[1],vectP[0]) - th
            distP = np.linalg.norm(vectP)

            # Solution when using the Center as the reference Frame
            num = (2 * self.carLength * np.sin(ksi))
            dnom =  (distP**2) - ( self.carLength * np.sin(ksi))
            dnom = np.sqrt(dnom)
            delta = np.arctan2(num, dnom)
                
            return np.clip(wrap_to_pi(delta),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)

        elif self.frameOfRef == FrameOfRef.REAR:

            # # 1. Get current and next waypoint
            wp1 = self.wp[:, np.mod(self.wpI, self.N-1)]

            goalWp = wp1

            distToGoalWp = np.linalg.norm(goalWp - p)

            if speed != 0:
                while distToGoalWp < ( 0.75 * speed):
                    self.wpI += 1

                    wp1 = self.wp[:, np.mod(self.wpI, self.N-1)]
                    
                    goalWp = wp1

                    distToGoalWp = np.linalg.norm(goalWp - p)

        
            vectP = goalWp - p        
            ksi = np.arctan2(vectP[1],vectP[0]) - th
            distP = np.linalg.norm(vectP)

            # Solution when using the rear axe as the reference Frame
            num = (2 * self.carLength * np.sin(ksi))
            dnom =  distP
            delta = np.arctan2(num, dnom)
                
            return np.clip(wrap_to_pi(delta),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
        
        else:
            return 0

class StanleySteeringControl(PureSteeringController):

    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):

        
        # ----------------------------------------------------- #
        # Own implementation

        # 1. Get current and next waypoint
        wp = self.wp[:, np.mod(self.wpI, self.N-1)]
        wpOld = self.wp[:, np.mod(self.wpI-1, self.N-1)]

        distToGoalWp = np.linalg.norm(wp - p)

        if speed != 0:
            while distToGoalWp < 0.25 * speed :
                self.wpI += 1

                wp = self.wp[:, np.mod(self.wpI, self.N-1)]

                distToGoalWp = np.linalg.norm(wp - p)

        # 1. Vector from old waypoint to current waypoint (the path segment)
        path_vec = wp - wpOld
      
        ksiS = np.arctan2(path_vec[1],path_vec[0]) - th
   
        # 2. Vector from old waypoint to the vehicle position
        vec_wp_to_p = p - wpOld

        # 3. Calculate cross-track error (eS) 
        # We use the 2D cross product to find the perpendicular distance
        # |a x b| / |a|
        cross_product = path_vec[1] * vec_wp_to_p[0] - path_vec[0] * vec_wp_to_p[1]
        eS = cross_product / np.linalg.norm(path_vec)

        # Note: This eS is signed! Positive means one side, negative the other.
        # For Stanley, you usually want: deltaE = np.arctan2(k * eS, speed)

        deltaE = np.arctan2((1.13 * eS), (speed + 0.2))
        delta = ksiS + deltaE
    
        return np.clip(
            wrap_to_pi(delta),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
    
        # ----------------------------------------------------- #
        # Quanser implementation

        # wp_1 = self.wp[:, np.mod(self.wpI, self.N-1)]
        # wp_2 = self.wp[:, np.mod(self.wpI+1, self.N-1)]
        
        # v = wp_2 - wp_1
        # v_mag = np.linalg.norm(v)
        # try:
        #     v_uv = v / v_mag
        # except ZeroDivisionError:
        #     return 0

        # tangent = np.arctan2(v_uv[1], v_uv[0])

        # s = np.dot(p-wp_1, v_uv)

        # if s >= v_mag:
        #     if  self.cyclic or self.wpI < self.N-2:
        #         self.wpI += 1

        # ep = wp_1 + v_uv*s
        # ct = ep - p
        # dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        # ect = np.linalg.norm(ct) * np.sign(dir)
        # psi = wrap_to_pi(tangent-th)

        # self.p_ref = ep
        # self.th_ref = tangent

        # return np.clip(
        #     wrap_to_pi(psi + np.arctan2(0.7*ect, speed)),
        #     -self.maxSteeringAngle,
        #     self.maxSteeringAngle)
    
def controlLoop():
    #region controlLoop setup
    global KILL_THREAD
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl or calibrate:
        if not pureControl:
            initialPose[0:2] = initialPose[0:2] + (np.array([np.cos(initialPose[2]), np.sin(initialPose[2])]) * 0.225)
        else:
            if frameOfRef == FrameOfRef.REAR:
                initialPose[0:2] = initialPose[0:2] - (np.array([np.cos(initialPose[2]), np.sin(initialPose[2])]) * 0.225)

        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose,calibrate=calibrate)
    else:
        gps = memoryview(b'')
    #endregion

    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i
    )
    if enableSteeringControl:
        if pureControl:
            steeringController = PureSteeringController(
                waypoints=waypointSequence,
                gain=K_stanley
            )
        else :    
            steeringController = StanleySteeringControl(
                waypoints=waypointSequence,
                gain=K_stanley
            )
    #endregion

    with qcar, gps:
        t0 = time.time()
        t=0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl and t > startDelay:
                # if gps.readGPS():
                    
                #     y_gps = np.array([
                #         gps.position[0],
                #         gps.position[1],
                #         gps.orientation[2]
                #     ])
                #     ekf.update(
                #         [qcar.motorTach, delta],
                #         dt,
                #         y_gps,
                #         qcar.gyroscope[2],
                #     )
                # else:
                    
                #     ekf.update(
                #         [qcar.motorTach, delta],
                #         dt,
                #         None,
                #         qcar.gyroscope[2],
                #     )
                
                ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                
                y = ekf.x_hat[1,0]
                
                th = ekf.x_hat[2,0]
                
                if not pureControl:
                    p = ( np.array([x, y])
                        + np.array([np.cos(th), np.sin(th)]) * 0.225)
                else:        
                    p = ( np.array([x, y]))
                
            v = qcar.motorTach
            #endregion

            #region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                #region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                #endregion

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
                #endregion

            qcar.write(u, delta)
            #endregion

        qcar.read_write_std(throttle= 0, steering= 0)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Setup and run experiment
if __name__ == '__main__':

    #region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            #  MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
#endregion