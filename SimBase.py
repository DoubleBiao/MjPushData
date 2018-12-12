
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen,MjSimState
import matplotlib.pyplot as plt
import os
import numpy as np
import math

class TableObj:
    def __init__(self,joint_name,MjModel,scaling=0.55):
        self.scaling = scaling
        self.pos = [0,0]
        self.joint_address = [MjModel.get_joint_qpos_addr(joint_name[0]),MjModel.get_joint_qpos_addr(joint_name[1])]
    
    def set_pos(self,pos,mjqpos):
        pos = self.roundpos(pos)   
        self.pos = pos    
        return self.execute_pos(mjqpos)
    
    def move_cube(self,displacement,mjqpos):
        self.pos = [self.pos[i] + displacement[i] for i in range(len(self.pos))] 
        self.pos = self.roundpos(self.pos)

        return self.execute_pos(mjqpos)

    def roundpos(self,pos):
        pos = [min(1.0,val) for val in pos]
        pos = [max(-1.0,val)for val in pos]
        return pos

    def execute_pos(self,mjqpos):
        pos = [val*self.scaling for val in self.pos]
        mjqpos[self.joint_address] = pos
        return mjqpos



 

class ImiRob:
    """ imitate robot that imitates(replay) the actions extracted from other robots
        This robot simulator is of the quadruped robob which has 4 legs along with 2 hinge joints for each leg.
        This simulation is based on mujoco. All robots are derivation of OpenAI gym ant.
    """

    def __init__(self,rendersetting,frame_skip=4,objnum=4,showarm=False):
        if showarm:
            xmlsource = '/home/xiaotian/CML/MjPush/xmls/show_env.xml'
        else:
            xmlsource = '/home/xiaotian/CML/MjPush/xmls/lab_env.xml'

        self.model =  load_model_from_path(xmlsource)
        self.sim = MjSim(self.model)
        self.frame_skip = frame_skip
        self.rendermode = {}
        self.init_state = self.sim.get_state()

        self.viewer = None
        self.setrendermode(rendersetting)
        self.stopflag = False
        self.objnum = objnum
        self.showarm = showarm

   

        self.cubes = [TableObj(["free_x_"+str(i+1),"free_y_"+str(i+1)],self.model) for i in range(self.objnum)]
        self.move_dict = {
            0:[0.3,0],
            1:[-0.3,0],
            2:[0,0.3,],
            3:[0,-0.3,]
        }
  
    
    def init_cubes(self,mjqpos):
        for i in range(self.objnum):
            pos = [np.random.uniform(low=-1.0, high=1.0),np.random.uniform(low=-1.0, high=1.0)]
            # pos =[i*0.3,i*0.3]
            mjqpos = self.cubes[i].set_pos(pos,mjqpos)
        
        return mjqpos
    
    def mesh_init_cubes(self,mjqpos,gridsize=0.5):
        def mesh_random(pt_num,swing=0.5,seglow=-1.0,seghigh=1.0):
            res = []
            last_pt = seglow - swing
            for remain_pt_num in reversed(range(pt_num)):
                segrange = [last_pt+swing,seghigh - remain_pt_num*swing]
                last_pt = np.random.uniform(low=segrange[0],high=segrange[1])
                res.append(last_pt)
            return np.random.permutation(res)
        
        self.x_coord = mesh_random(self.objnum,swing=gridsize)
        self.y_coord = mesh_random(self.objnum,swing=gridsize)
        
        for i in range(self.objnum):
            pos = [self.x_coord[i],self.y_coord[i]]
            mjqpos = self.cubes[i].set_pos(pos,mjqpos)
        
        return mjqpos
    
    def check_contacts(self):
        def checkdistance(poses,scaling=0.6,eps=0.1):
            poses = sorted(poses)
            dis = np.array([np.abs(poses[i]-poses[i+1]) for i in range(len(poses)-1)])
            return np.any(dis < eps)

        cube_pos_x = [self.cubes[i].pos[0] for i in range(self.objnum)]
        cube_pos_y = [self.cubes[i].pos[1] for i in range(self.objnum)]
        return checkdistance(cube_pos_x) or checkdistance(cube_pos_y)

        





    def setrendermode(self,rendersetting):#render_flg, plt_switch, screenwidth = 500, screenhight = 500, interval = 5):
        self.rendermode["rendertype"]=rendersetting["render_flg"]
        if rendersetting["render_flg"] == True: # in-screen: ignore other mode parameters
            self.viewer = MjViewer(self.sim)
            return
        else:                  # off-screen:
            self.rendermode["W"] = rendersetting["screenwidth"]
            self.rendermode["H"] = rendersetting["screenhight"]
            self.rendermode["pltswitch"] = rendersetting["plt_switch"]
            self.viewer = MjRenderContextOffscreen(self.sim)
            if plt.get_fignums():
                plt.close()
            self.fig,self.ax = plt.subplots()

            return 
    
    
    
    def onscreenshow(self):
        self.viewer.render()
    
    def offscreenshow(self):
        self.viewer.render(self.rendermode["H"],self.rendermode["W"])
        im_src = self.viewer.read_pixels(self.rendermode["H"],self.rendermode["W"],depth=False)
        im_src = np.flip(im_src)
        im_src = np.flip(im_src, axis = 1)
        im_src = np.flip(im_src, axis = 2)


        if self.rendermode["pltswitch"]:
            self.ax.cla()
            self.ax.imshow(im_src)
            plt.pause(1e-10)

        return im_src, not plt.get_fignums()

    
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset(self):
        self.sim.set_state(self.init_state)
        self.stopflag = False

        self.qposrecrd = []

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
  
        if self.showarm:
            qpos = np.array([0.1, -50 / 180 * math.pi, 61.03 / 180 * math.pi,
               0, 0, 0, # robotic arm
               0, 0, 0, 0,
               0, 0, 0, 0, # two fingers
               -0.62, 0.32,
               -0.72, 0.38,
               -0.835, 0.425,
               -0.935, 0.46,]) # 4 cubes

 

        qpos = self.mesh_init_cubes(qpos)
        self.set_state(qpos, qvel)



    def setcam(self, distance=3, azimuth=0, elevation=-90, lookat=[-0.2,0,0], hideoverlay=True,trackbodyid=-1):
        self.viewer.cam.trackbodyid = trackbodyid
        self.viewer.cam.distance=distance
        self.viewer.cam.azimuth=azimuth
        self.viewer.cam.elevation=elevation
        self.viewer.cam.lookat[:]=lookat
        self.viewer._hide_overlay=hideoverlay

    def moverobot(self):
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos=self.cubes[0].move_cube([0.2,0],qpos)


        self.set_state(qpos, qvel)
    
    def random_move_cube(self):
        move_fashion, move_cube = np.random.randint(4,size=2)
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos=self.cubes[move_cube].move_cube(self.move_dict[move_fashion],qpos)
        self.set_state(qpos, qvel)

        return move_cube,move_fashion

    def save_video(self,savepath):
        video_data = np.zeros((2,self.rendermode["H"],self.rendermode["W"],3)).astype(np.uint8)
        im1,_ = self.offscreenshow()
        movecube,movefashion = self.random_move_cube()
        im2,_ = self.offscreenshow()
        video_data[0,...] = im1
        video_data[1,...] = im2

        np.savez_compressed(savepath,video=video_data,cubeindx=movecube,actionindx=movefashion)

    def __del__(self):
        del self.viewer
        del self.model
        del self.sim


if __name__ == "__main__":

        

    rendersetting = {}
    rendersetting["render_flg"] = False
    rendersetting["screenwidth"] = 128
    rendersetting["screenhight"] = 128
    rendersetting["plt_switch"]  = True
    robot = ImiRob('/home/xiaotian/CML/MjPush/xmls/lab_env.xml',rendersetting,showarm=False)
    robot.setcam(distance=2.0,azimuth=45,elevation=-60,lookat=[-0.8,0.4,0])
  

    while True:
        robot.reset()
        keepmoving = True
        for i in range(10):
            if i  == 5:
                robot.random_move_cube()
            robot.offscreenshow()
            
            if robot.check_contacts():
                from IPython import embed
                embed()
                assert False
        plt.pause(5e-1)

    



            
