# %%
import numpy as np
import json
import pathlib
import torch

MAX_LENGTH = 2

"""
FILE IO
"""
def read_input(file_path):
    with open(file_path) as json_data:
        d = json.load(json_data)

    return d['positions'], d['rotations'], d['parents'], d['foot_contact']

def format_data(file_path):
    # reading all data
    all_orig_positions, all_orig_rotations, parents, all_foot_contact = read_input(file_path)
    all_orig_positions = np.array(all_orig_positions)
    all_orig_rotations = np.array(all_orig_rotations)
    print(f'FINSIHED IMPORTING DATA \n\noriginal positions shape: {all_orig_positions.shape}')

    all_global_positions, all_global_rotations = to_global(all_orig_positions, all_orig_rotations, parents)
    all_global_positions = to_blend_coords(all_global_positions)
    print(f'CONVERTED ORIGINAL DATA TO GLOBAL')
    return all_global_positions, all_global_rotations


"""
COORDINATE AND VECTOR MANIPULATION
"""
        
def swap_coordinate_axes(positions, one, two):
    new_pos = positions.copy()
    new_pos[..., [one, two]] = positions[..., [two, one]]
    return new_pos

def flip_coordinate_axis(positions, axis):
    positions[..., axis] *= -1
    return positions


def to_blend_coords(positions):
    positions = swap_coordinate_axes(positions, 1, 2)
    positions = swap_coordinate_axes(positions, 0, 1)
    #positions = flip_coordinate_axis(positions, 1)
    positions = positions * 0.01
    return positions
    
    
def from_blend_coords(positions):
    #positions = flip_coordinate_axis(positions, 1)
    positions = swap_coordinate_axes(positions, 0, 1)
    positions = swap_coordinate_axes(positions, 1, 2)
    positions = positions * 100
    return positions

def asser(b, c):
    assert(torch.allclose(torch.tensor(b), c))

def get_euler_from_vec_vectorized(vec, keep_length = False):
    dtype = torch.float64
    device = 'cpu'
    return get_euler_from_vec_vectorized_torch(torch.tensor(vec, dtype=dtype, device=device), dtype, device, keep_length).cpu().numpy()

def get_euler_from_vec_vectorized_torch(vec2, dtype, device, keep_length = False):
    length2 = torch.linalg.norm(vec2, axis=-1)
    vec2 = vec2 / length2[..., None]

    yaw2 = torch.atan2(vec2[..., 1], vec2[..., 0])
    pitch2 = torch.asin(-vec2[..., 2])
    roll2 = torch.zeros_like(yaw2, dtype=dtype, device=device)

    if keep_length:
        roll2 = length2 * (6 / MAX_LENGTH)

    return torch.stack([yaw2, pitch2, roll2], axis=-1)


def get_vec_from_euler(angle):
    length = angle[2] / (6 / MAX_LENGTH)
    
    if length == 0:
        length = 1
        
    y = np.cos(angle[1]) * np.sin(angle[0])
    z = np.sin(angle[1])
    x = np.cos(angle[1]) * np.cos(angle[0])
    
    x *= length
    y *= length
    z *= -length
    
    return np.array([x, y, z])

def get_vec_from_euler_vectorized(angle):
    '''
    angle: (..., 3)
    '''
    dtype = torch.float64
    device = 'cpu'
    return get_vec_from_euler_vectorized_torch(torch.tensor(angle, dtype=dtype, device=device), dtype, device).cpu().numpy()

def get_vec_from_euler_vectorized_torch(angle, dtype, device):
    '''
    angle: (..., 3)
    '''
    length = angle[..., 2] / (6 / MAX_LENGTH)
    
    mask = length == 0
    length[mask] = 1
        
    y = torch.cos(angle[..., 1]) * torch.sin(angle[..., 0])
    z = torch.sin(angle[..., 1])
    x = torch.cos(angle[..., 1]) * torch.cos(angle[..., 0])
    
    x *= length
    y *= length
    z *= -length
    
    return torch.stack([x, y, z], axis=-1)


def m9dtoeuler(m9d):
    dtype = torch.float64
    device = 'cpu'
    return m9dtoeuler_torch(torch.tensor(m9d, dtype=dtype, device=device), dtype, device).cpu().numpy()

def m9dtoeuler_torch(m9d, dtype, device):
    pitch = -1*torch.asin(m9d[:, :, 0, 2, 0])
    roll = torch.atan2(m9d[:, :, 0, 2, 1] / torch.cos(pitch) , m9d[:, :, 0, 2, 2] / torch.cos(pitch))
    yaw = torch.atan2(m9d[:, :, 0, 1, 0] / torch.cos(pitch) , m9d[:, :, 0, 0, 0] / torch.cos(pitch))
    return torch.stack((yaw, pitch, roll), axis=-1)

def euler_to_matrix_vectorized(euler_angles):
    """
    Convert multiple sets of Euler angles to rotation matrices using vectorized operations.

    Args:
        euler_angles (numpy.ndarray): Array of shape (n, 3) where each row contains
                                      yaw, pitch, and roll angles in radians.

    Returns:
        numpy.ndarray: Array of shape (n, 3, 3) containing rotation matrices.
    """
    dtype = torch.float64
    device = 'cpu'
    return euler_to_matrix_vectorized_torch(torch.tensor(euler_angles, dtype=dtype, device=device), dtype, device).cpu().numpy()

def euler_to_matrix_vectorized_torch(euler_angles, dtype, device):
    """
    Convert multiple sets of Euler angles to rotation matrices using vectorized operations.

    Args:
        euler_angles (torch.tensor): Array of shape (n, 3) where each row contains
                                      yaw, pitch, and roll angles in radians.

    Returns:
        torch.tensor : Array of shape (n, 3, 3) containing rotation matrices.
    """
    # Unpack Euler angles
    yaw, pitch, roll = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    # Trigonometric calculations
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # Component matrices, vectorized across the first dimension
    Rz = torch.stack([torch.stack([cy, -sy, torch.zeros_like(cy, dtype=dtype, device=device)], dim=0),
                       torch.stack([sy, cy, torch.zeros_like(cy, dtype=dtype, device=device)], dim=0),
                       torch.stack([torch.zeros_like(cy, dtype=dtype, device=device), torch.zeros_like(cy, dtype=dtype, device=device), torch.ones_like(cy, dtype=dtype, device=device)], dim=0)], dim=0)
    Ry = torch.stack([torch.stack([cp, torch.zeros_like(cp, dtype=dtype, device=device), sp], dim=0),
                       torch.stack([torch.zeros_like(cp, dtype=dtype, device=device), torch.ones_like(cp, dtype=dtype, device=device), torch.zeros_like(cp, dtype=dtype, device=device)], dim=0),
                       torch.stack([-sp, torch.zeros_like(cp, dtype=dtype, device=device), cp], dim=0)], dim=0)
    Rx = torch.stack([torch.stack([torch.ones_like(cr, dtype=dtype, device=device), torch.zeros_like(cr, dtype=dtype, device=device), torch.zeros_like(cr, dtype=dtype, device=device)], dim=0),
                       torch.stack([torch.zeros_like(cr, dtype=dtype, device=device), cr, -sr], dim=0),
                       torch.stack([torch.zeros_like(cr, dtype=dtype, device=device), sr, cr], dim=0)], dim=0)

    # Transpose to shape the matrices correctly for matrix multiplication
    Rz = Rz.permute(2, 0, 1)
    Ry = Ry.permute(2, 0, 1)
    Rx = Rx.permute(2, 0, 1)

    # Matrix multiplication for all sets, np.einsum can also be used for clarity
    R = Rz @ Ry @ Rx

    return R

"""
HELPERS
"""

def to_global(positions, rotations, parents):
    # takes in original positions and rotations and returns the new representation of positions and rotations
    global_rot, global_pos = fk(rotations, positions, parents)
    return global_pos, global_rot

def fk(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (Tensor): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (Tensor): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D int Tensor): Parent indices.

    Returns:
        Tensor, Tensor: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = np.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
            np.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)

def get_bone_mapping():
    '''
    IK controller returns three joints: root joint of the bone (1 joint) and the direction vector (2 joints)
    otherwise returns the single joint
    '''
    bone_mapping = {
        'left hand': 17,
        'left elbow': 16,
        'right hand': 21,
        'right elbow': 20,
        'left foot': 3,
        'left knee': 2,
        'left toe': 4,
        'right foot': 7,
        'right knee': 6,
        'right toe': 8,
        'left shoulder': 15,
        'right shoulder': 19,
        'left hip': 1,
        'right hip': 5,
        'head': 13,
        'spine top': 12,
        'root': 9
    }
    return bone_mapping

def get_representation1_mapping():
    rm1 = {
        'left hand': 0,
        'left elbow': 1,
        'right hand': 2,
        'right elbow': 3,
        'left foot': 4,
        'left knee': 5,
        'right foot': 6,
        'right knee': 7,
        'head': 8,
        'spine top': 9,
        'root': 10,
        'root rotation': 11
    }
    return rm1


def make_converted_json(file_path, save_path):
    all_global_positions, all_global_rotations = format_data(file_path)
    
    # creating new representation
    rep1 = representation1(all_global_positions, all_global_rotations)
    print(f'CONVERTED TO CUSTOM REPRESENTATION \n\nrep1 first instance: \nrot: \n{rep1[0, 0]}')
    
    rep1list = rep1.tolist()
    with open(save_path, 'w') as file:
        json.dump(rep1list, file)

    partial_back = representation1_backwards_partial(rep1)
    mask = representation1_partial_mask()
    original_masked = all_global_positions * mask[:, np.newaxis]
    total_error = np.sum(np.abs(original_masked - partial_back))
    print(f'total error: {total_error}')

    root_rot = representation1_backwards_rot(rep1)
    total_error = np.sum(np.abs(all_global_rotations[:, :, 0, :, :] - root_rot))
    print(f'total error: {total_error}')

"""
MAIN CONVERSION MODULES
"""

def representation1(rep0, root_angles):
    '''
    rep0 represents the original global positions of all joints

    Representation1:
    - One loc/rot for hands and feet as IK control. (4, 4)
    - Two shoulder and two hip endpoints, location only. (4, 0)
    - One head location (1, 0)
    Final structure: (9 location, 4 rotation)
    condensed into 13 rotations
    '''
    dtype = torch.float64
    device = 'cpu'
    return representation1_torch(torch.tensor(rep0, dtype=dtype, device=device), torch.tensor(root_angles, dtype=dtype, device=device), dtype, device).cpu().numpy()


def representation1_torch(rep0, root_angles, dtype, device):
    '''
    rep0 represents the original global positions of all joints

    Representation1:
    - One loc/rot for hands and feet as IK control. (4, 4)
    - Two shoulder and two hip endpoints, location only. (4, 0)
    - One head location (1, 0)
    Final structure: (9 location, 4 rotation)
    condensed into 13 rotations
    '''
    INDICES_IN_USE = 22
    rep1 = torch.zeros((rep0.shape[0], rep0.shape[1], INDICES_IN_USE, 3), dtype=dtype, device=device)
    bm = get_bone_mapping()
    rm1 = get_representation1_mapping()
    
    pull_target_dist_multiplier = 4

    root_location = rep0[:, :, bm['root']]

    # elbow and knee pull targets are stored relative to the hands/foot, not the root!
    rep1[:,:, rm1['left hand']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['left hand']] - root_location, dtype, device, True)
    left_arm_mid = 0.5 * rep0[:,:, bm['left hand']] + 0.5 * rep0[:,:, bm['left shoulder']]
    rep1[:,:, rm1['left elbow']] = get_euler_from_vec_vectorized_torch(pull_target_dist_multiplier * (rep0[:,:, bm['left elbow']] - left_arm_mid) + left_arm_mid - rep0[:,:, bm['left hand']], dtype, device, True)

    # hands
    rep1[:,:, rm1['right hand']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['right hand']] - root_location, dtype, device, True)
    right_arm_mid = 0.5 * rep0[:,:, bm['right hand']] + 0.5 * rep0[:,:, bm['right shoulder']]
    rep1[:,:, rm1['right elbow']] = get_euler_from_vec_vectorized_torch(pull_target_dist_multiplier * (rep0[:,:, bm['right elbow']] - right_arm_mid) + right_arm_mid - rep0[:,:, bm['right hand']], dtype, device, True)

    # feet
    rep1[:,:, rm1['left foot']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['left foot']] - root_location, dtype, device, True)
    left_leg_mid = 0.5 * rep0[:,:, bm['left foot']] + 0.5 * rep0[:,:, bm['left hip']]
    rep1[:,:, rm1['left knee']] = get_euler_from_vec_vectorized_torch(pull_target_dist_multiplier * (rep0[:,:, bm['left knee']] - left_leg_mid) + left_leg_mid - rep0[:,:, bm['left foot']], dtype, device, True)
    
    rep1[:,:, rm1['right foot']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['right foot']] - root_location, dtype, device, True)
    right_leg_mid = 0.5 * rep0[:,:, bm['right foot']] + 0.5 * rep0[:,:, bm['right hip']]
    rep1[:,:, rm1['right knee']] = get_euler_from_vec_vectorized_torch(pull_target_dist_multiplier * (rep0[:,:, bm['right knee']] - right_leg_mid) + right_leg_mid - rep0[:,:, bm['right foot']], dtype, device, True)

    # joints and root
    rep1[:,:, rm1['spine top']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['spine top']] - root_location, dtype, device, True)
    rep1[:,:, rm1['head']] = get_euler_from_vec_vectorized_torch(rep0[:,:, bm['head']] - root_location, dtype, device, True)
    rep1[:,:, rm1['root']] = get_euler_from_vec_vectorized_torch(root_location, dtype, device, True)
    rep1[:,:, rm1['root rotation']] = m9dtoeuler_torch(root_angles, dtype, device)
    return rep1

    
def representation1_backwards_partial(rep1):
    dtype = torch.float64
    device = 'cpu'
    return representation1_backwards_partial_torch(torch.tensor(rep1, dtype=dtype, device=device), dtype, device).cpu().numpy()

def representation1_backwards_partial_torch(rep1, dtype, device):
    INDICES_IN_USE = 22
    rep0 = torch.zeros((rep1.shape[0], rep1.shape[1], INDICES_IN_USE, 3), dtype=dtype, device=device)
    bm = get_bone_mapping()
    rm1 = get_representation1_mapping()
    
    root_location = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['root']], dtype, device)

    #hands
    rep0[:, :, bm['left hand']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['left hand']], dtype, device) + root_location
    rep0[:, :, bm['right hand']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['right hand']], dtype, device) + root_location

    rep0[:, :, bm['left foot']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['left foot']], dtype, device) + root_location
    rep0[:, :, bm['right foot']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['right foot']], dtype, device) + root_location

    rep0[:, :, bm['head']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['head']], dtype, device) + root_location
    rep0[:, :, bm['spine top']] = get_vec_from_euler_vectorized_torch(rep1[:, :, rm1['spine top']], dtype, device) + root_location
    rep0[:, :, bm['root']] = root_location
            
    return rep0

def representation1_backwards_rot(rep1):
    dtype = torch.float64
    device = 'cpu'
    return representation1_backwards_rot_torch(torch.tensor(rep1, dtype=dtype, device=device), dtype, device).cpu().numpy()

def representation1_backwards_rot_torch(rep1, dtype, device):
    rm1 = get_representation1_mapping()
    root_rot_euler = rep1[:, :, rm1["root rotation"]].reshape(-1, 3)
    root_rot_9D = euler_to_matrix_vectorized_torch(root_rot_euler, dtype, device).reshape(rep1.shape[0], rep1.shape[1], 3, 3)

    return root_rot_9D

def get_global_root_rot_from_rep1(rep1):
    rm1 = get_representation1_mapping()
    return rep1[:, :, rm1['root rotation']]

def representation1_partial_mask():
    bm = get_bone_mapping()
    mask = np.zeros((22,))
    mask[bm['left hand']] = 1
    mask[bm['right hand']] = 1
    mask[bm['left foot']] = 1
    mask[bm['right foot']] = 1
    mask[bm['head']] = 1
    mask[bm['spine top']] = 1
    mask[bm['root']] = 1
    return mask


if __name__ == "__main__":
    wkdir_path = str(pathlib.Path(__file__).parent.resolve())
    print(f'working directory path: {wkdir_path}')

    data_path = wkdir_path[:wkdir_path.rfind('\\')]
    data_path = data_path[:data_path.rfind('\\')]
    data_path += "\\motion_data\\"
    print(f'data path: {data_path}')

    file_name = "lafan1_detail_model_benchmark_5_0-2231.json"
    save_name = f"CONVERTED_{file_name}"

    data_path = "/home/tyler/Desktop/Github/motion_inbetweening/scripts/ignore/"
    make_converted_json(data_path + file_name, data_path + save_name)


# %%
