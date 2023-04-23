import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import roma
import helper

bones = json.load(open("model/bones_neutral.json"))
bone_names = list(bones.keys())

obj_template = helper.read_obj(r"model/000_generic_neutral_mesh_newuv.obj")

for name in bones:

    bone = bones[name]
    parent = bone["parent"]

    L2P_rotation = np.array(bone["matrix"])
    head_in_p = (np.array(bone["head"]) + ([0, bones[parent]["length"], 0] if parent is not None else 0))
    L2P_transformation = np.identity(4)
    L2P_transformation[:3, :3] = L2P_rotation
    L2P_transformation[:3, 3] = head_in_p

    bone["L2P_transformation"] = torch.tensor(L2P_transformation, dtype=torch.float32)


def update_L2W_transformation(name):
    bone = bones[name]

    transformation_name = "L2W_transformation"
    local_pose_transformation = torch.eye(4)[None]

    if transformation_name in bone:
        return bone[transformation_name]

    parent = bone["parent"]

    L2W_transformation = bone["L2P_transformation"] @ local_pose_transformation

    if parent is not None:
        L2W_transformation = update_L2W_transformation(parent) @ L2W_transformation

    bone[transformation_name] = L2W_transformation

    return L2W_transformation


def update_L2W_transformation_pose(L2W_transformation_pose, L2P_transformation, ith_bone, pose_matrix):
    """
    L2W_transformation_pose: list of N_bones
    """
    name = bone_names[ith_bone]
    bone = bones[name]

    if L2W_transformation_pose[ith_bone] is not None:
        return L2W_transformation_pose[ith_bone]

    local_pose_transformation = torch.eye(4, device=pose_matrix.device)[None].repeat(pose_matrix.shape[0], 1, 1)
    local_pose_transformation[:, :3, :3] = pose_matrix[:, bone_names.index(name)]

    L2W_transformation = L2P_transformation[ith_bone] @ local_pose_transformation

    parent = bone["parent"]
    if parent is not None:
        ith_parent = bone_names.index(parent)
        L2W_transformation = update_L2W_transformation_pose(L2W_transformation_pose, L2P_transformation, ith_parent, pose_matrix) @ L2W_transformation

    L2W_transformation_pose[ith_bone] = L2W_transformation
    return L2W_transformation_pose[ith_bone]


N_bones = 8

L2P_transformation = torch.stack([bones[bone_names[i]]["L2P_transformation"] for i in range(len(bone_names))])

L2W_transformation = torch.zeros(1, N_bones, 4, 4)  # [1, Nb, 4, 4]

for name in bones:
    L2W_transformation[:, bone_names.index(name)] = update_L2W_transformation(name)

W2L_transformation = torch.linalg.inv(L2W_transformation)


"""
^^^ CONSTANT ^^
"""


def uv1d_construct_delta(uv1d, tau):
    """
    uv1d: [1, 1, 256, 256]
    tau: [B, 1]
    return: [B, 14062, 1]
    """

    grid = getattr(uv1d_construct_delta, "grid", None)
    if grid is None:
        obj = obj_template
        uv = obj.vts
        uv[:, 1] = 1 - uv[:, 1]
        uv = uv * 2 - 1
        fv = obj.fvs
        fvt = obj.fvts
        grid = np.ones((1, 1, 14062, 2)) * 2
        for i in range(len(fv)):
            for j in range(4):
                if grid[0][0][fv[i][j]][0] == 2:
                    grid[0][0][fv[i][j]] = uv[fvt[i][j]]
                else:
                    continue

        grid = torch.tensor(grid).to(uv1d)
        setattr(uv1d_construct_delta, "grid", grid)

    grid = grid + F.pad(tau * 2, [1, 0])[:, None, None, :]

    output = torch.nn.functional.grid_sample(uv1d.expand(grid.shape[0], -1, -1, -1), grid, mode='bilinear', padding_mode="border", align_corners=True)
    return output[:, 0, 0, :, None]


class PCA(nn.Module):
    def __init__(self, mean, diff):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean[None]).to(torch.float32))
        self.register_buffer("diff", torch.tensor(diff[None]).to(torch.float32))

    def forward(self, a=None, clip=999):
        if a is None:
            return self.mean
        return self.mean + (a.reshape([a.shape[0], a.shape[1]] + [1] * (len(self.diff.shape) - 2)) * self.diff)[:, :clip].sum(dim=1)


def load_pca(path):
    pca = np.load(path, allow_pickle=True).item()
    mean = pca["mean"]
    VT_std = pca["VT_std"]
    pca = PCA(mean, VT_std)
    return pca


class HACK(nn.Module):

    def __init__(self):
        super().__init__()

        W = torch.tensor(np.load("model/weight_map_smooth.npy"), dtype=torch.float32)  # [Nb, 14062]
        W = W / W.sum(axis=0, keepdims=True)
        self.register_buffer("W", W, persistent=False)

        T = torch.tensor(obj_template.vs, dtype=torch.float32)  # [14062, 3]
        self.register_buffer("T", T)

        P = torch.zeros(N_bones, 3, 3, 14062, 3)  # [N_bones, 3, 3, 14062, 3]
        self.register_buffer("P", P)

        L = torch.tensor(cv2.imread("model/Lc_mid.png", cv2.IMREAD_GRAYSCALE) / 255, dtype=torch.float32)[None, None]  # [1, 1, 256, 256]
        self.register_buffer("L", L, persistent=False)

        ts = torch.tensor(np.load("model/ts_larynx.npy"), dtype=torch.float32)  # [3]
        self.register_buffer("ts", ts, persistent=False)

        self.register_buffer("L2P_transformation", L2P_transformation, persistent=False)
        self.register_buffer("W2L_transformation", W2L_transformation, persistent=False)

        blendshapes = torch.tensor(np.load("model/blendshape.npy"), dtype=torch.float32)
        neutral = blendshapes[:1]
        blendshapes = blendshapes[1:] - neutral
        self.register_buffer("E", blendshapes, persistent=False)

    def get_L_tau(self, tau):
        """
        tau>0 means upper
        """
        dist = uv1d_construct_delta(self.L, tau)
        L_tau = dist * self.ts
        return L_tau

    def forward(self, theta, tau, alpha, bsw, T=None, P=None, E=None):
        """
        theta: [B, Nb, 3]
        tau: [B, 1]
        alpha: [B, 1]
        bsw: [B, 55]

        return: [B, Nv, 3]
        """
        B = theta.shape[0]
        theta_matrix = roma.rotvec_to_rotmat(theta)  # [B, Nb, 3, 3]
        theta_matrix_zero = theta_matrix - torch.cat([theta_matrix[:, :1], (torch.eye(3).to(theta)[None, None]).expand(B, N_bones - 1, 3, 3)], dim=1)

        P = self.P if P is None else P
        P_theta = (theta_matrix_zero[:, :, :, :, None, None] * P).sum(dim=(1, 2, 3))

        L2W_transformation_pose = [None] * N_bones
        for ith_bone in range(len(bone_names)):
            update_L2W_transformation_pose(L2W_transformation_pose, self.L2P_transformation, ith_bone, theta_matrix)
        L2W_transformation_pose = torch.stack(L2W_transformation_pose, dim=1)  # [B, Nb, 4, 4]

        W2L2pWs = L2W_transformation_pose @ self.W2L_transformation  # [B, Nb, 4, 4]
        W2L2pW_weighted = (W2L2pWs[:, :, None, :, :] * self.W[None, :, :, None, None]).sum(axis=1)  # [B, 14062, 4, 4]

        T = self.T if T is None else T
        E = self.E if E is None else E
        T_theta = T + P_theta + (E[:, :, :] * bsw[:, :, None, None]).sum(dim=1) + self.get_L_tau(tau) * alpha[:, :, None]

        T_transformed = (W2L2pW_weighted @ F.pad(T_theta, [0, 1], value=1)[:, :, :, None])[:, :, :3, 0]  # [B, 14062, 3]

        data = {
            "T_transformed": T_transformed,
        }

        return data
