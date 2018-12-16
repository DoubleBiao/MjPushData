from SimBase import ImiRob
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def randomact():
   
    actions = np.random.uniform(-np.pi, 
                                    np.pi,
                                    robot.low.shape)

    return actions

def genind(actiondim):
    return np.random.binomial(1, .5, actiondim)


savepath = "/home/xiaotian/CML/MjPush/data"

VideoNum = 100


rendersetting = {}
rendersetting["render_flg"] = False
rendersetting["screenwidth"] = 64
rendersetting["screenhight"] = 64
rendersetting["plt_switch"]  = False
robot = ImiRob(rendersetting,showarm=False)
robot.setcam(distance=2.0,azimuth=45,elevation=-60,lookat=[-0.8,0.4,0])

for i in range(VideoNum):
    contact = True
    while contact:
        robot.reset()
        robot.save_video(os.path.join(savepath,str(i)))
        contact = robot.check_contacts()
        if contact:
            print("cube contact occured")


