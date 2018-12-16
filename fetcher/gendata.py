from SimBase import FetchPushEnv
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

def randomact():
   
    actions = np.random.uniform(-np.pi, 
                                    np.pi,
                                    robot.low.shape)

    return actions

def genind(actiondim):
    return np.random.binomial(1, .5, actiondim)


savepath = "/home/cml/CML/fetcher/gendata"

VideoNum = 20000


rendersetting = {}
rendersetting["render_flg"] = False
rendersetting["screenwidth"] = 64
rendersetting["screenhight"] = 64
rendersetting["plt_switch"]  = False
rendersetting["distance"] = 1.2
rendersetting["azimuth"] = 180
rendersetting["elevation"] = -40
rendersetting["lookat"] = [1.2,0.75,0.2]
env = FetchPushEnv(rendersetting)

for i in tqdm(range(VideoNum)):
    env.reset()
    env.random_push(os.path.join(savepath,str(i)))


