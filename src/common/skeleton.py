import numpy as np

class StickFigure:
    def __init__(self):
        # Define segment lengths (normalized to height ~ 1.0)
        self.lengths = {
            'torso': 0.25,
            'neck': 0.05, 
            'upper_arm': 0.12,
            'lower_arm': 0.12,
            'upper_leg': 0.15,
            'lower_leg': 0.15,
            'head_radius': 0.05
        }
        
        # Define constraints (min_angle, max_angle) in degrees
        self.constraints = {
            'torso': (-15, 15),       
            'head': (-20, 20),        
            'l_shoulder': (-160, 60), 
            'r_shoulder': (-160, 60),
            'l_elbow': (-150, 0),     
            'r_elbow': (-150, 0),
            'l_hip': (-45, 90),       
            'r_hip': (-45, 90),
            'l_knee': (0, 150),       
            'r_knee': (0, 150)
        }
        
    def get_random_pose(self):
        pose = {}
        for joint, (min_a, max_a) in self.constraints.items():
            pose[joint] = np.random.uniform(min_a, max_a)
        return pose

    def forward_kinematics(self, pose, center_pos=(0.5, 0.5), scale=1.0):
        rads = {k: np.radians(v) for k, v in pose.items()}
        
        hip_x, hip_y = center_pos
        
        torso_angle = np.pi/2 + rads['torso']
        neck_x = hip_x + np.cos(torso_angle) * self.lengths['torso'] * scale
        neck_y = hip_y - np.sin(torso_angle) * self.lengths['torso'] * scale 
        
        head_angle = torso_angle + rads['head']
        head_x = neck_x + np.cos(head_angle) * self.lengths['neck'] * scale
        head_y = neck_y - np.sin(head_angle) * self.lengths['neck'] * scale
        
        base_shoulder_angle = torso_angle + np.pi 
        
        l_uarm_angle = base_shoulder_angle + rads['l_shoulder'] 
        l_elbow_x = neck_x + np.cos(l_uarm_angle) * self.lengths['upper_arm'] * scale
        l_elbow_y = neck_y - np.sin(l_uarm_angle) * self.lengths['upper_arm'] * scale
        
        l_larm_angle = l_uarm_angle + rads['l_elbow']
        l_hand_x = l_elbow_x + np.cos(l_larm_angle) * self.lengths['lower_arm'] * scale
        l_hand_y = l_elbow_y - np.sin(l_larm_angle) * self.lengths['lower_arm'] * scale
        
        r_uarm_angle = base_shoulder_angle + rads['r_shoulder'] 
        r_elbow_x = neck_x + np.cos(r_uarm_angle) * self.lengths['upper_arm'] * scale
        r_elbow_y = neck_y - np.sin(r_uarm_angle) * self.lengths['upper_arm'] * scale
        
        r_larm_angle = r_uarm_angle + rads['r_elbow']
        r_hand_x = r_elbow_x + np.cos(r_larm_angle) * self.lengths['lower_arm'] * scale
        r_hand_y = r_elbow_y - np.sin(r_larm_angle) * self.lengths['lower_arm'] * scale
        
        base_hip_angle = torso_angle + np.pi
        
        l_uleg_angle = base_hip_angle + rads['l_hip']
        l_knee_x = hip_x + np.cos(l_uleg_angle) * self.lengths['upper_leg'] * scale
        l_knee_y = hip_y - np.sin(l_uleg_angle) * self.lengths['upper_leg'] * scale
        
        l_lleg_angle = l_uleg_angle + rads['l_knee']
        l_foot_x = l_knee_x + np.cos(l_lleg_angle) * self.lengths['lower_leg'] * scale
        l_foot_y = l_knee_y - np.sin(l_lleg_angle) * self.lengths['lower_leg'] * scale
        
        r_uleg_angle = base_hip_angle + rads['r_hip']
        r_knee_x = hip_x + np.cos(r_uleg_angle) * self.lengths['upper_leg'] * scale
        r_knee_y = hip_y - np.sin(r_uleg_angle) * self.lengths['upper_leg'] * scale
        
        r_lleg_angle = r_uleg_angle + rads['r_knee']
        r_foot_x = r_knee_x + np.cos(r_lleg_angle) * self.lengths['lower_leg'] * scale
        r_foot_y = r_knee_y - np.sin(r_lleg_angle) * self.lengths['lower_leg'] * scale
        
        joints = {
            'hip': (hip_x, hip_y),
            'neck': (neck_x, neck_y),
            'head': (head_x, head_y),
            'l_elbow': (l_elbow_x, l_elbow_y),
            'l_hand': (l_hand_x, l_hand_y),
            'r_elbow': (r_elbow_x, r_elbow_y),
            'r_hand': (r_hand_x, r_hand_y),
            'l_knee': (l_knee_x, l_knee_y),
            'l_foot': (l_foot_x, l_foot_y),
            'r_knee': (r_knee_x, r_knee_y),
            'r_foot': (r_foot_x, r_foot_y)
        }
        return joints
