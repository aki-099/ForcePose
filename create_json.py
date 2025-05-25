import pandas as pd
import numpy as np
import json
from config import KEYPOINTS

def main(
    cam1_2d: dict,
    cam2_2d: dict,
    df_3d: pd.DataFrame,
    cam1_bbox: pd.DataFrame,
    cam2_bbox: pd.DataFrame,
    subject: str,
    movement:str,
    total_frames:int,
    frame_width:int,
    frame_height:int,
    )-> dict:
    
    json_data = {}
    json_data['subject'] = subject
    json_data['movement'] = movement
    json_data['frames'] = frames_dict(
        np.array(cam1_2d['keypoints_visibility']),
        np.array(cam2_2d['keypoints_visibility']),
        df_3d,
        cam1_bbox,
        cam2_bbox
    )
    json_data['frame_width'] = frame_width
    json_data['frame_height'] = frame_height
    json_data['total_frames'] = total_frames
    json_data['mocap'] = mocap_dict(total_frames)
    json_data['grf'] = grf_dict(df_3d)
    return json_data

def frames_dict(
    cam1_2d: np.ndarray, 
    cam2_2d: np.ndarray, 
    df_3d: pd.DataFrame, 
    cam1_bbox: pd.DataFrame, 
    cam2_bbox: pd.DataFrame
    ) -> dict:
    
    total_frames = []
    min_frames = min(len(cam1_2d), len(cam2_2d), len(df_3d))
    for frame in range(min_frames):
        frame_dict = {}
        frame_dict['image_name'] = f"{frame:06d}.jpg"
        frame_dict['img_index'] = frame
        frame_dict['cam1'] = cam_dict(cam1_2d, cam1_bbox, frame)
        frame_dict['cam2'] = cam_dict(cam2_2d, cam2_bbox, frame)
        frame_dict['triangulated_pose'] = df_3d.loc[frame, df_3d.columns.str.startswith(tuple(KEYPOINTS.keys()))].values.flatten().tolist()
        total_frames.append(frame_dict)
    return total_frames

def cam_dict(kpts_2d: np.ndarray, bbox_df: pd.DataFrame, frame: int) -> dict:
    cam_data = {}
    cam_data['bbox'] = bbox_df.iloc[frame].tolist()
    cam_data['keypoints'] = kpts_2d[frame].flatten().tolist()
    return cam_data

def grf_dict(df_3d: pd.DataFrame) -> dict:
    COL_NAMES = {'time': 'time',
                'Force:X': 'ground_force1_vx',
                'Force:Y': 'ground_force1_vy',
                'Force:Z': 'ground_force1_vz',
                'Force2:X': 'ground_force2_vx',
                'Force2:Y': 'ground_force2_vy',
                'Force2:Z': 'ground_force2_vz'}
    grf_data = {}
    for col, new_col in COL_NAMES.items():
        grf_data[new_col] = df_3d[col].tolist()
    return grf_data

def mocap_dict(total_frames: int) -> dict:
    MOCAP_KPTS = ['CLAV', 'LACRM', 'LASIS', 'LHEEL', 'LLAK', 
     'LLEB', 'LLFARM', 'LLHND', 'LLKN', 'LLTOE', 
     'LLWR', 'LMAK', 'LMEB', 'LMFARM', 'LMHND', 
     'LMKN', 'LMTOE', 'LMWR', 'LPSI', 'LSHANK', 
     'LTHIGH', 'LUPARM', 'RACRM', 'RASIS', 'RHEEL', 
     'RLAK', 'RLEB', 'RLFARM', 'RLHND', 'RLKN', 
     'RLTOE', 'RLWR', 'RMAK', 'RMEB', 'RMFARM', 
     'RMHND', 'RMKN', 'RMTOE', 'RMWR', 'RPSI', 
     'RSHANK', 'RTHIGH', 'RUPARM', 'STRM', 'T1', 'T10', 'THEAD']
    mocap_data = {}
    for kpt in MOCAP_KPTS:
        mocap_data[kpt] = np.zeros((total_frames, 3)).tolist()
    return mocap_data