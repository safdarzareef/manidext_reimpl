# Contains functions to canonicalize and uncanonicalize the object and hand sequences. Also contains the batched versions of the uncanonicalized functions.
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix, matrix_to_axis_angle
from scipy.spatial.transform import Rotation as R


def canon_seq(mano_seq, obj_seq):
        """
        Compute Velocities and Canonicalize the object and hand sequence.
        """
        rhand_seq = mano_seq["right"]
        lhand_seq = mano_seq["left"]
        num_frames = obj_seq.shape[0]
        rhand_pose, rhand_global_orient, rhand_trans = rhand_seq["pose"], rhand_seq["rot"], rhand_seq["trans"]
        lhand_pose, lhand_global_orient, lhand_trans = lhand_seq["pose"], lhand_seq["rot"], lhand_seq["trans"]
        
        obj_angular_velocities = []
        obj_linear_velocities = []
        obj_artic_velocities = []

        new_rhand = []
        new_lhand = []
    
        for i in range(num_frames):
            curr_obj_seq = obj_seq[i]
            if i == 0:
                prev_obj_seq = curr_obj_seq.copy()
            else:
                prev_obj_seq = obj_seq[i-1]

            curr_obj_artic, curr_obj_rot, curr_obj_trans = curr_obj_seq[0], curr_obj_seq[1:4], curr_obj_seq[4:]
            prev_obj_artic, prev_obj_rot, prev_obj_trans = prev_obj_seq[0], prev_obj_seq[1:4], prev_obj_seq[4:]
            curr_obj_trans, prev_obj_trans = curr_obj_trans * 0.001, prev_obj_trans * 0.001

            # Compute angular velocity
            curr_obj_rot = R.from_rotvec(curr_obj_rot).as_matrix()
            prev_obj_rot = R.from_rotvec(prev_obj_rot).as_matrix()
            angular_vel = prev_obj_rot.T @ curr_obj_rot
            angular_vel = matrix_to_rotation_6d(torch.tensor(angular_vel)).numpy()
            obj_angular_velocities.append(angular_vel.flatten())

            # Compute linear velocity
            translational_vel = (curr_obj_trans - prev_obj_trans)
            translational_vel = prev_obj_rot.T @ translational_vel
            obj_linear_velocities.append(translational_vel.flatten())

            # Compute articulation velocity
            cur_artic_rot = np.array([0, 0, -curr_obj_artic]) # need to reverse angle
            cur_artic_rot = R.from_rotvec(cur_artic_rot).as_matrix()
            prev_artic_rot = np.array([0, 0, -prev_obj_artic]) # need to reverse angle
            prev_artic_rot = R.from_rotvec(prev_artic_rot).as_matrix()
            artic_vel = prev_artic_rot.T @ cur_artic_rot
            artic_vel = matrix_to_rotation_6d(torch.tensor(artic_vel)).numpy()
            obj_artic_velocities.append(artic_vel.flatten())

            curr_rhand_pose, curr_rhand_global_orient, curr_rhand_trans = rhand_pose[i], rhand_global_orient[i], rhand_trans[i]
            curr_lhand_pose, curr_lhand_global_orient, curr_lhand_trans = lhand_pose[i], lhand_global_orient[i], lhand_trans[i]
            
            curr_rhand_pose = torch.tensor(curr_rhand_pose).reshape(15, 3)
            curr_rhand_pose_mat = axis_angle_to_matrix(curr_rhand_pose)
            curr_rhand_pose = matrix_to_rotation_6d(curr_rhand_pose_mat).reshape(-1)

            tmp_rhand_pose = rotation_6d_to_matrix(curr_rhand_pose.reshape(-1, 15, 6))
            assert(np.linalg.norm(tmp_rhand_pose - curr_rhand_pose_mat) < 1e-5)

            curr_lhand_pose = torch.tensor(curr_lhand_pose).reshape(15, 3)
            curr_lhand_pose_mat = axis_angle_to_matrix(curr_lhand_pose)
            curr_lhand_pose = matrix_to_rotation_6d(curr_lhand_pose_mat).reshape(-1)

            tmp_lhand_pose = rotation_6d_to_matrix(curr_lhand_pose.reshape(-1, 15, 6))
            assert(np.linalg.norm(tmp_lhand_pose - curr_lhand_pose_mat) < 1e-5)

            curr_obj_rot = torch.tensor(curr_obj_rot).float()

            new_rhand_trans = curr_obj_rot.T @ (curr_rhand_trans - curr_obj_trans)
            new_lhand_trans = curr_obj_rot.T @ (curr_lhand_trans - curr_obj_trans)

            tmp_lhand_trans = (curr_obj_rot @ new_lhand_trans) + curr_obj_trans
            tmp_rhand_trans = (curr_obj_rot @ new_rhand_trans) + curr_obj_trans

            assert(np.linalg.norm(tmp_lhand_trans - curr_lhand_trans) < 1e-5)
            assert(np.linalg.norm(tmp_rhand_trans - curr_rhand_trans) < 1e-5)

            curr_rhand_global_orient = torch.tensor(curr_rhand_global_orient)
            curr_rhand_rot_mat = axis_angle_to_matrix(curr_rhand_global_orient)
            new_rhand_global_orient = curr_obj_rot.T @ curr_rhand_rot_mat
            new_rhand_global_orient = matrix_to_rotation_6d(new_rhand_global_orient)

            tmp_rhand_rot_mat = curr_obj_rot @ rotation_6d_to_matrix(new_rhand_global_orient)
            assert(np.linalg.norm(tmp_rhand_rot_mat - curr_rhand_rot_mat) < 1e-5)

            curr_lhand_global_orient = torch.tensor(curr_lhand_global_orient)
            curr_lhand_rot_mat = axis_angle_to_matrix(curr_lhand_global_orient)
            new_lhand_global_orient = curr_obj_rot.T @ curr_lhand_rot_mat
            new_lhand_global_orient = matrix_to_rotation_6d(new_lhand_global_orient)

            tmp_lhand_rot_mat = curr_obj_rot @ rotation_6d_to_matrix(new_lhand_global_orient)
            assert(np.linalg.norm(tmp_lhand_rot_mat - curr_lhand_rot_mat) < 1e-5)

            rhand_vec = np.zeros(99)
            rhand_vec[:3] = new_rhand_trans.flatten()
            rhand_vec[3:9] = new_rhand_global_orient.flatten()
            rhand_vec[9:] = curr_rhand_pose
            new_rhand.append(rhand_vec)

            lhand_vec = np.zeros(99)
            lhand_vec[:3] = new_lhand_trans.flatten()
            lhand_vec[3:9] = new_lhand_global_orient.flatten()
            lhand_vec[9:] = curr_lhand_pose
            new_lhand.append(lhand_vec)

            tmp_pose, tmp_rot, tmp_trans = convert_hand_vec_to_components(torch.tensor(rhand_vec).float(), curr_obj_rot, curr_obj_trans)
            assert(np.linalg.norm(tmp_trans - curr_rhand_trans) < 1e-5)
            tmp_rot = R.from_rotvec(tmp_rot).as_matrix()
            assert(np.linalg.norm(tmp_rot - curr_rhand_rot_mat.numpy()) < 1e-5)

            tmp_pose, tmp_rot, tmp_trans = convert_hand_vec_to_components(torch.tensor(lhand_vec).float(), curr_obj_rot, curr_obj_trans)
            assert(np.linalg.norm(tmp_trans - curr_lhand_trans) < 1e-5)
            tmp_rot = R.from_rotvec(tmp_rot).as_matrix()
            assert(np.linalg.norm(tmp_rot - curr_lhand_rot_mat.numpy()) < 1e-5)
            
        print("No assertion error in canon_seq!")
        
        obj_angular_velocities = torch.tensor(np.array(obj_angular_velocities)).float()
        obj_linear_velocities = torch.tensor(np.array(obj_linear_velocities)).float()
        obj_artic_velocities = torch.tensor(np.array(obj_artic_velocities)).float()
        new_rhand = torch.tensor(np.array(new_rhand)).unsqueeze(-2)
        new_lhand = torch.tensor(np.array(new_lhand)).unsqueeze(-2)
        hand_pose = torch.cat([new_rhand, new_lhand], dim=-2).float()

        obj_start_pose = obj_seq[0]
        start_obj_artic, start_obj_rot, start_obj_trans = obj_start_pose[0], obj_start_pose[1:4], obj_start_pose[4:]
        start_obj_trans = torch.tensor(start_obj_trans).float() * 0.001 # convert to m from mm
        start_obj_rot = R.from_rotvec(start_obj_rot).as_matrix()
        start_obj_rot = matrix_to_rotation_6d(torch.tensor(start_obj_rot)).float()

        start_obj_artic = R.from_rotvec(np.array([0, 0, -start_obj_artic])).as_matrix() # need to reverse angle
        start_obj_artic = matrix_to_rotation_6d(torch.tensor(start_obj_artic)).float()

        obj_linear_velocities = torch.cat([start_obj_trans.unsqueeze(0), obj_linear_velocities], dim=-2)
        obj_angular_velocities = torch.cat([start_obj_rot.unsqueeze(0), obj_angular_velocities], dim=-2)
        obj_artic_velocities = torch.cat([start_obj_artic.unsqueeze(0), obj_artic_velocities], dim=-2)

        return {
            "angular_v": obj_angular_velocities,
            "trans_v": obj_linear_velocities,
            "artic_v": obj_artic_velocities,
            "hand_pose": hand_pose,
        }


def convert_hand_vec_to_components(hand_vec, obj_rot, obj_trans):
    # Extract components from hand vector
    trans = hand_vec[:3]
    global_orient_6d = hand_vec[3:9]
    pose_6d = hand_vec[9:]
    
    # Convert translation back to global space
    trans = (obj_rot @ trans) + obj_trans
    
    # Convert 6D rotation to matrix then to rotation vector for global orientation
    global_orient_6d = global_orient_6d.reshape(-1, 6)
    global_orient_mat = obj_rot @ rotation_6d_to_matrix(global_orient_6d)
    global_orient = matrix_to_axis_angle(global_orient_mat).reshape(-1 , 3)
    
    # Convert 6D rotation to matrix then to rotation vector for pose
    pose_6d = pose_6d.reshape(-1, 6)
    pose_mat = rotation_6d_to_matrix(pose_6d)
    pose = matrix_to_axis_angle(pose_mat).reshape(-1, 3)
    return pose, global_orient, trans


def aa_to_rotmat(aa):
    aa = aa.unsqueeze(0)
    rotmat = axis_angle_to_matrix(aa).squeeze(0)
    return rotmat


def rot6d_to_rotmat(rot6d):
    rot6d = rot6d.unsqueeze(0)
    rotmat = rotation_6d_to_matrix(rot6d).squeeze(0)
    return rotmat


def rotmat_to_aa(rotmat):
    rotmat = rotmat.unsqueeze(0)
    aa = matrix_to_axis_angle(rotmat).squeeze(0)
    return aa


def uncanon_hand_pose(hand_pose, obj_seq, orig_mano_seq=None):
    num_frames = hand_pose.shape[0]
    assert(num_frames == obj_seq.shape[0])
    rhand_canon_vec, lhand_canon_vec = hand_pose[:, 0, :], hand_pose[:, 1, :]
    rhand_pose, rhand_global_orient, rhand_trans =  torch.zeros((rhand_canon_vec.shape[0], 15, 3)), \
                                                    torch.zeros((rhand_canon_vec.shape[0], 3)), \
                                                    torch.zeros((rhand_canon_vec.shape[0], 3))
    lhand_pose, lhand_global_orient, lhand_trans =  torch.zeros((lhand_canon_vec.shape[0], 15, 3)), \
                                                    torch.zeros((lhand_canon_vec.shape[0], 3)), \
                                                    torch.zeros((lhand_canon_vec.shape[0], 3))
    
    if orig_mano_seq is not None: # then check the conversion
        orig_rmano, orig_lmano = orig_mano_seq["right"], orig_mano_seq["left"]
        orig_rhand_pose, orig_rhand_rot, orig_rhand_trans = orig_rmano["pose"], orig_rmano["rot"], orig_rmano["trans"]
        orig_lhand_pose, orig_lhand_rot, orig_lhand_trans = orig_lmano["pose"], orig_lmano["rot"], orig_lmano["trans"]

    for i in range(num_frames):
        curr_obj_trans = torch.tensor(obj_seq[i][4:]).float() * 0.001
        curr_obj_rot = torch.tensor(aa_to_rotmat(obj_seq[i][1:4])).float()

        rhand_vec, lhand_vec = rhand_canon_vec[i], lhand_canon_vec[i]

        rhand_pose[i], rhand_global_orient[i], rhand_trans[i] = convert_hand_vec_to_components(rhand_vec, curr_obj_rot, curr_obj_trans)
        lhand_pose[i], lhand_global_orient[i], lhand_trans[i] = convert_hand_vec_to_components(lhand_vec, curr_obj_rot, curr_obj_trans)

        if orig_mano_seq is not None:
            assert(np.linalg.norm(rhand_trans[i] - orig_rhand_trans[i]) < 1e-4)
            rhand_rotmat = aa_to_rotmat(rhand_global_orient[i]).numpy()
            orig_rhand_rotmat = aa_to_rotmat(orig_rhand_rot[i]).numpy()
            assert(np.linalg.norm(rhand_rotmat - orig_rhand_rotmat) < 1e-4)

            assert(np.linalg.norm(lhand_trans[i] - orig_lhand_trans[i]) < 1e-4)
            lhand_rotmat = aa_to_rotmat(lhand_global_orient[i]).numpy()
            orig_lhand_rotmat = aa_to_rotmat(orig_lhand_rot[i]).numpy()
            assert(np.linalg.norm(lhand_rotmat - orig_lhand_rotmat) < 1e-4)
        
    if orig_mano_seq is not None:
        print("No assertion error in uncanon_hand_pose!")

    new_mano_seq = {
        "right": {
                "pose": rhand_pose,
                "rot": rhand_global_orient,
                "trans": rhand_trans,
            },
        "left": {
                "pose": lhand_pose,
                "rot": lhand_global_orient,
                "trans": lhand_trans,
            }
    }
    return new_mano_seq


def uncanon_obj_pose(trans_v, angular_v, artic_v, orig_obj_seq=None):
    init_pos, init_rot, init_artic = trans_v[0], rot6d_to_rotmat(angular_v[0]), rot6d_to_rotmat(artic_v[0])

    trans_v, angular_v, artic_v = trans_v[1:], angular_v[1:], artic_v[1:]
    obj_pos = torch.zeros((trans_v.shape[0], 3))
    obj_rot = torch.zeros((trans_v.shape[0], 3, 3))
    obj_artic = torch.zeros((trans_v.shape[0], 3, 3))
    new_obj_seq = torch.zeros((trans_v.shape[0], 7))
   
    for i in range(trans_v.shape[0]):
        if i != 0:
            obj_pos[i] = (obj_pos[i-1] + (obj_rot[i-1] @ trans_v[i]))
            obj_rot[i] = obj_rot[i-1] @ rot6d_to_rotmat(angular_v[i])
            obj_artic[i] = obj_artic[i-1] @ rot6d_to_rotmat(artic_v[i])
        else:
            obj_pos[i] = init_pos
            obj_rot[i] = init_rot
            obj_artic[i] = init_artic

        tmp_artic_rotvec = rotmat_to_aa(obj_artic[i])[-1] * -1
        tmp_obj_rotvec = rotmat_to_aa(obj_rot[i]).flatten()

        new_obj_seq[i] = torch.tensor([
                    tmp_artic_rotvec, 
                    tmp_obj_rotvec[0], tmp_obj_rotvec[1], tmp_obj_rotvec[2],
                    obj_pos[i][0] * 1000, obj_pos[i][1] * 1000, obj_pos[i][2] * 1000
                    ])
        
        if orig_obj_seq is not None:
            orig_artic, orig_rot, orig_pos = orig_obj_seq[i][0], orig_obj_seq[i][1:4], orig_obj_seq[i][4:]
            orig_pos = torch.tensor(orig_pos).float() * 0.001
            orig_rot = R.from_rotvec(orig_rot).as_matrix()
            assert(np.linalg.norm(obj_pos[i] - orig_pos) < 1e-4)
            assert(np.linalg.norm(obj_rot[i] - orig_rot) < 1e-4)
            assert(np.linalg.norm(new_obj_seq[i][0] - orig_artic) < 1e-4)
    return new_obj_seq


def uncanon_seq(hand_pose, trans_v, angular_v, artic_v, orig_obj_seq=None, orig_mano_seq=None):
    new_obj_seq = uncanon_obj_pose(
        trans_v=trans_v, 
        angular_v=angular_v, 
        artic_v=artic_v,
        orig_obj_seq=orig_obj_seq,
    )

    new_mano_seq = uncanon_hand_pose(
        hand_pose=hand_pose,
        obj_seq=new_obj_seq,
        orig_mano_seq=orig_mano_seq,
    )
    return new_obj_seq, new_mano_seq



def uncanon_seq_batch(hand_pose, trans_v, angular_v, artic_v):
    new_obj_seq = uncanon_obj_pose_batch(
        trans_v=trans_v, 
        angular_v=angular_v, 
        artic_v=artic_v,
    )

    new_mano_seq = uncanon_hand_pose_batch(
        hand_pose=hand_pose,
        obj_seq=new_obj_seq,
    )
    return new_obj_seq, new_mano_seq


def uncanon_obj_pose_batch(trans_v, angular_v, artic_v, orig_obj_seq=None):
    bs, seqlen, _ = trans_v.shape
    init_pos, init_rot, init_artic = trans_v[:, 0], rotation_6d_to_matrix(angular_v[:, 0]), rotation_6d_to_matrix(artic_v[:, 0])

    trans_v, angular_v, artic_v = trans_v[:, 1:], angular_v[:, 1:], artic_v[:, 1:]
    seqlen -= 1
    obj_pos = torch.zeros((bs, seqlen, 3)).to(trans_v.device)
    obj_rot = torch.zeros((bs, seqlen, 3, 3)).to(trans_v.device)
    obj_artic = torch.zeros((bs, seqlen, 3, 3)).to(trans_v.device)
    new_obj_seq = torch.zeros((bs, seqlen, 7)).to(trans_v.device)
   
    for i in range(seqlen):
        if i != 0:
            expanded_trans_v = trans_v[:, i].unsqueeze(-1)
            obj_pos[:, i] = (obj_pos[:, i-1] + torch.bmm(obj_rot[:, i-1], expanded_trans_v).squeeze(-1))
            obj_rot[:, i] = torch.bmm(obj_rot[:, i-1], rotation_6d_to_matrix(angular_v[:, i]))
            obj_artic[:, i] = torch.bmm(obj_artic[:, i-1], rotation_6d_to_matrix(artic_v[:, i]))
        else:
            obj_pos[:, i] = init_pos
            obj_rot[:, i] = init_rot
            obj_artic[:, i] = init_artic

        tmp_artic_rotvec = matrix_to_axis_angle(obj_artic[:, i])[:, -1] * -1
        tmp_obj_rotvec = matrix_to_axis_angle(obj_rot[:, i])

        new_obj_seq[:, i] = torch.cat([tmp_artic_rotvec.unsqueeze(-1), tmp_obj_rotvec, obj_pos[:, i] * 1000], dim=-1)
    return new_obj_seq


def convert_hand_vec_to_components_batch(hand_vec, obj_rot, obj_trans):
    # Extract components from hand vector
    bs = hand_vec.shape[0]
    trans = hand_vec[:, :3]
    global_orient_6d = hand_vec[:, 3:9]
    pose_6d = hand_vec[:, 9:]
    
    # Convert translation back to global space
    expanded_trans = trans.unsqueeze(-1)
    trans = torch.bmm(obj_rot, expanded_trans).squeeze(-1) + obj_trans
    
    # Convert 6D rotation to matrix then to rotation vector for global orientation
    global_orient_6d = global_orient_6d.reshape(bs, 6)
    global_orient_mat = torch.bmm(obj_rot, rotation_6d_to_matrix(global_orient_6d))
    global_orient = matrix_to_axis_angle(global_orient_mat).reshape(bs , 3)
    
    # Convert 6D rotation to matrix then to rotation vector for pose
    pose_6d = pose_6d.reshape(bs * 15, 6)
    pose_mat = rotation_6d_to_matrix(pose_6d)
    pose = matrix_to_axis_angle(pose_mat).reshape(bs, 15, 3)
    return pose, global_orient, trans


def uncanon_hand_pose_batch(hand_pose, obj_seq, orig_mano_seq=None):
    bs, seqlen, _, _ = hand_pose.shape

    rhand_canon_vec, lhand_canon_vec = hand_pose[:, :, 0, :], hand_pose[:, :, 1, :]
    rhand_pose, rhand_global_orient, rhand_trans =  torch.zeros((bs, seqlen, 15, 3)).to(hand_pose.device), \
                                                    torch.zeros((bs, seqlen, 3)).to(hand_pose.device), \
                                                    torch.zeros((bs, seqlen, 3)).to(hand_pose.device)
    lhand_pose, lhand_global_orient, lhand_trans =  torch.zeros((bs, seqlen, 15, 3)).to(hand_pose.device), \
                                                    torch.zeros((bs, seqlen, 3)).to(hand_pose.device), \
                                                    torch.zeros((bs, seqlen, 3)).to(hand_pose.device)

    for i in range(seqlen):
        curr_obj_trans = obj_seq[:, i, 4:] * 0.001
        curr_obj_rot = axis_angle_to_matrix(obj_seq[:, i, 1:4])

        rhand_vec, lhand_vec = rhand_canon_vec[:, i], lhand_canon_vec[:, i]

        rhand_pose[:, i], rhand_global_orient[:, i], rhand_trans[:, i] = convert_hand_vec_to_components_batch(rhand_vec, curr_obj_rot, curr_obj_trans)
        lhand_pose[:, i], lhand_global_orient[:, i], lhand_trans[:, i] = convert_hand_vec_to_components_batch(lhand_vec, curr_obj_rot, curr_obj_trans)

    new_mano_seq = {
        "right": {
                "pose": rhand_pose,
                "rot": rhand_global_orient,
                "trans": rhand_trans,
            },
        "left": {
                "pose": lhand_pose,
                "rot": lhand_global_orient,
                "trans": lhand_trans,
            }
    }
    return new_mano_seq