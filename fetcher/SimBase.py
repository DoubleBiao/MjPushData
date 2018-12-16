import os
from gym import utils
from fetch_env import FetchEnv
from mujoco_py import MjRenderContextOffscreen,MjViewer
import matplotlib.pyplot as plt
import numpy as np 
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')

class TableObj:
    def __init__(self,joint_name,MjModel,scaling=0.28,origin=[1.2,0.75,0.4+0.025]):
        self.scaling = scaling
        self.pos = [0,0]
        self.joint_address = joint_name#[MjModel.get_joint_qpos_addr(joint_name[0]),MjModel.get_joint_qpos_addr(joint_name[1])]
        self.origin = origin
    
    def set_pos(self,pos,sim):
        pos = self.roundpos(pos)   
        self.pos = pos    
        self.execute_pos(sim)

    def get_pos(self):
        pos = [val*self.scaling for val in self.pos] + [0.0]
        pos = [pos[i] + self.origin[i] for i in range(len(pos))]
        return pos

    def roundpos(self,pos):
        pos = [min(1.0,val) for val in pos]
        pos = [max(-1.0,val)for val in pos]
        return pos

    def execute_pos(self,sim):
        pos = [val*self.scaling for val in self.pos] + [0.0]
        pos = [pos[i] + self.origin[i] for i in range(len(pos))]
        sim.data.set_joint_qpos(self.joint_address, pos+[1., 0., 0., 0.])





class FetchPushEnv(FetchEnv):
    def __init__(self,rendersetting, objnum = 4,reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
      
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        self.rendermode = {}
        self._viewer_setup(rendersetting)
        self.objnum = objnum

        self.objs = [TableObj("object{}:joint".format(i),self.model) for i in range(self.objnum)]


        self.movedict = {
            "dimidx":[np.array([1.0,0.0]),     #x - pos
                      np.array([-1.0,0.0]),    #x - neg
                      np.array([0.0,1.0]),   #y - pos,
                      np.array([0.0,-1.0])], #y - neg 
          }

    def _viewer_setup(self,rendersetting):
        self.rendermode["rendertype"]=rendersetting["render_flg"]



        if rendersetting["render_flg"] == True: # in-screen: ignore other mode parameters
            self.viewer = MjViewer(self.sim)
        else:                  # off-screen:
            self.rendermode["W"] = rendersetting["screenwidth"]
            self.rendermode["H"] = rendersetting["screenhight"]
            self.rendermode["pltswitch"] = rendersetting["plt_switch"]
            self.viewer = MjRenderContextOffscreen(self.sim)
            if plt.get_fignums():
                plt.close()
            self.fig,self.ax = plt.subplots()

        # body_id = self.sim.model.body_name2id('robot0:gripper_link')
        # lookat = self.sim.data.body_xpos[body_id]
        # for idx, value in enumerate(lookat):
        #     self.viewer.cam.lookat[idx] = value
        self.viewer.cam.lookat[:] = rendersetting["lookat"]
        self.viewer.cam.distance = rendersetting["distance"]
        self.viewer.cam.azimuth = rendersetting["azimuth"]
        self.viewer.cam.elevation = rendersetting["elevation"]



    def offscreenrender(self):
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

    def move_gripper(self,newpos):
        # Move end effector into position.
        gripper_target = np.array(newpos)
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
            self.viewer.render()
            # self.offscreenrender()



    def mesh_init_cubes(self,gridsize=0.4):
        def mesh_random(pt_num,swing=0.375,seglow=-0.7,seghigh=0.8):
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
            self.objs[i].set_pos(pos,self.sim)
        self.sim.forward()

    def random_push(self,savepath):
        video_data = np.zeros((2,self.rendermode["H"],self.rendermode["W"],3)).astype(np.uint8)
        move_fashion, move_cube = np.random.randint(4,size=2)

        im1,_ = self.offscreenrender()
        self.push_obj(move_fashion,move_cube)
        im2,_ = self.offscreenrender()


        video_data[0,...] = im1
        video_data[1,...] = im2
        np.savez_compressed(savepath,video=video_data,cubeindx=move_cube,actionindx=move_fashion)


    def push_obj(self, move_fashion,move_cube):
        

        objpos = np.array(self.objs[move_cube].get_pos())
        dimidx = self.movedict["dimidx"][move_fashion]
  
        objpos+= np.append(dimidx*0.06,[0.4])
        # objpos[1]+= 0.07
        # objpos[2]+= 0.4
        self.move_gripper(objpos)

        objpos[2] -= 0.38
        self.move_gripper(objpos)

        for _ in range(10):
            self._set_action(np.append(dimidx*-0.4,[0.0,0.0]))
            # self._set_action(np.array([0.0,-0.4,0,0]))
            self.sim.step()
            # self.viewer.render()
            # self.offscreenrender()
    
        for _ in range(10):
            self._set_action(np.append(dimidx*0.1,[0.0,0.0]))
            # self._set_action(np.array([0.0,0.1,0,0]))
            self.sim.step()
            self.viewer.render()
            # self.offscreenrender()

        objpos[2]+= 0.7
        self.move_gripper(objpos)


    def reset(self):
        self._reset_sim()
        self.mesh_init_cubes()
        self.move_gripper([1.2,0.75,0.8])


if __name__ == "__main__":
    rendersetting = {}
    rendersetting["render_flg"] = True
    rendersetting["screenwidth"] = 128
    rendersetting["screenhight"] = 128
    rendersetting["plt_switch"]  = True
    rendersetting["distance"] = 1.2
    rendersetting["azimuth"] = 180
    rendersetting["elevation"] = -40
    rendersetting["lookat"] = [1.2,0.75,0.2]
    env = FetchPushEnv(rendersetting)
    while True:
        env.reset()
        for i in range(1000):
            env.render()
        # env.push_obj()
