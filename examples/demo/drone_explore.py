from gibson.envs.camera_env import VirtualCameraEnv
from gibson.envs.env_bases import BaseEnv
from gibson.utils.play import play
from gibson.core.render.profiler import Profiler
from scipy.signal import savgol_filter
from multiprocessing import Pool, TimeoutError
import gibson
import argparse
import os
import trimesh
import numpy as np
import gym
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import time
import pybullet as p
import yaml
import peakutils
import imageio
'''
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass
'''

#import pyglet.window as pw

from collections import deque
#from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread


def load_obj(obj_path):
    verts = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

def sample_faces(vertices, faces, n_samples=10**4):
    """
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    Parameters:
      vertices  - n x 3 matrix
      faces     - n x 3 matrix
      n_samples - positive integer

    Return:
      vertices - point cloud

    Reference :
      [1] Barycentric coordinate system

      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Contributed by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2);
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
        np.sqrt(r[:,0:1]) * r[:,1:] * C
    return P


def find_floors(obj):
    verts, faces = obj
    points = sample_faces(verts, faces, n_samples=10**4)
    mini = points[:,2].min()
    maxi = points[:,2].max()
    hist = np.histogram(points[:,2], bins=30)

    plt.hist(points[:,2], bins=30)

    #data = savgol_filter(np.hstack((np.zeros(1), hist[0], np.zeros(1))), 9, 3)
    data = np.hstack((np.zeros(1), hist[0], np.zeros(1)))
    peaks = peakutils.indexes(data, min_dist=5)
    peaks = peaks - 1
    print(peaks.shape)
    print(peaks)
    plt.scatter(hist[1][peaks], hist[0][peaks])

    plt.savefig('hist.png')
    plt.clf()

    peak_values = hist[1][peaks]
    return [(a - b) / 2 + b
            for a, b in zip(peak_values[:-1], peak_values[1:])]


def find_cameras(camera_path):
    cameras = []
    with open(camera_path) as f:
        for line in f:
            l = line.strip().split(',')
            uuid = l[0]
            xyz = list(map(float, l[1:4]))
            quat = list(map(float, l[4:8]))
            cameras.append(xyz)

    clusters = []
    for cam in cameras:
        found = False
        for i, (cc, v) in enumerate(clusters):
            if abs(cam[2] - cc[2]) < 0.1:
                found = True
                clusters[i][1] += 1
                break
        if not found:
            clusters.append([cam, 1])
    camera_pos = [c[2] for (c, v) in clusters
                  if v > 5]
    return camera_pos


def make_floorplan(mesh, height=0.5):
    z = height
    s = mesh.section(plane_origin=(0, 0, z),
                     plane_normal=[0,0,1])
    s, _ = s.to_planar()
    return s
    #verts, faces = obj
    #z = height
    #return meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))


def draw_floorplan(floorplan):
    floorplan.show()


def sample_floorplan(floorplan, num_samples, min_dist, boundary_dist):
    paths = []
    for i, entity in enumerate(floorplan.entities):
        discrete = entity.discrete(floorplan.vertices)
        paths.append(discrete)
    paths = np.vstack(paths)
    paths = floorplan.vertices
    path = mpath.Path(floorplan.vertices)

    min_x = np.min(paths[:,0])
    max_x = np.max(paths[:,0])
    min_y = np.min(paths[:,1])
    max_y = np.max(paths[:,1])

    used = 0
    used_points = np.zeros((num_samples, 2))
    timeout = 1000
    while used < num_samples:
        point = np.array(np.random.random(2))
        point[0] = point[0] * (max_x - min_x) + min_x
        point[1] = point[1] * (max_y - min_y) + min_y
        contains = path.contains_point(point, radius=boundary_dist)
        if not contains:
            continue
        dist = np.linalg.norm(used_points[:used,:] - point, axis=1)
        if np.all(dist >= min_dist) or used == 0:
            used_points[used,:] = point
            used += 1
            timeout = 1000
        timeout -= 1
        if timeout == 0:
            break
    used_points = used_points[:used,:]
    return used_points


def draw_points(points):
    plt.scatter(x=points[:,0], y=points[:,1])


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return w, x, y, z


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def render_model(config_path, model, output_directory):
    start = time.time()
    print('Rendering {:s}...'.format(model))
    # Load the config file for finding the model of interest
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    config['model_id'] = model
    # Get the path to the model directory
    model_path = gibson.data.datasets.get_model_path(config['model_id'])
    # mesh_z_up is the mesh where the z dimension is normal to the
    # floor/ceiling of the building
    mesh_path = os.path.join(model_path, 'mesh_z_up.obj')
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)
    # The camera poses used to scan the mesh.
    posefile = os.path.join(model_path, 'camera_poses.csv')
    heights = find_cameras(posefile)

    floorplans_directory = os.path.join(output_directory, 'floorplans')
    views_directory = os.path.join(output_directory, 'views')
    os.makedirs(floorplans_directory, exist_ok=True)
    os.makedirs(views_directory, exist_ok=True)

    sampled_cameras = {}
    for i, h in enumerate(heights):
        #print('Height: {:3f}'.format(h))
        #print('Making floorplan...')
        floorplan = make_floorplan(mesh, height=h)
        if len(floorplan.vertices) == 0:
        #    print('Skipped due to lack of floorplan')
            continue
        #print('Floorplan segments: {:d}'.format(len(floorplan)))
        #print('Sampling floorplan...')
        points = sample_floorplan(floorplan, 1000, 1.0, -0.5)
        #print('# of points sampled: {:d}'.format(points.shape[0]))
        sampled_cameras[i] = {
            'points': points,
            'z': h
        }
        plt.figure()
        draw_floorplan(floorplan)
        draw_points(points)
        path = os.path.join(floorplans_directory, 'floor{:02d}.png'.format(i))
        plt.savefig(path)
        plt.clf()

    # CURRENT HACKY STRATEGY:
    # For each floor, for each camera, randomly select 3 camera orientations

    # IDEAL STRATEGY:
    # For each floor, select a set of cameras and camera orientations such that
    # each triangle is visible by at least 3 cameras

    num_views_per_camera = 3
    pitch_range = (30.0 / 180) * np.pi

    env = VirtualCameraEnv(config = config)
    obs = env.reset()
    total_images = 0
    poses = []
    for floor, data in sampled_cameras.items():
        camera_positions = data['points']
        h = data['z']
        image_idx = 0
        for i in range(camera_positions.shape[0]):
            pos_x = camera_positions[i,0]
            pos_y = camera_positions[i,1]
            pos_z = h

            for n in range(num_views_per_camera):
                num_tries = 0
                while num_tries < 10:
                    pitch = (np.random.random(1)[0] - 0.5) * pitch_range
                    yaw = np.random.random(1)[0] * 2 * np.pi
                    roll = 0
                    # Reference vector
                    q = axisangle_to_q((1, 0, 0), 0)
                    # Rotate by yaw
                    q = q_mult(axisangle_to_q((1, 0, 0), yaw), q)
                    # Rotate by pitch
                    q = q_mult(axisangle_to_q((0, 1, 0), pitch), q)
                    view = np.array([pos_x, pos_y, pos_z, q[0], q[1], q[2], q[3]])
                    obs, _, _, _ = env.step(view) # x y z quat
                    depth = obs["depth"]
                    median_depth  = np.median(depth)
                    mx = np.max(depth)
                    mi = np.min(depth)
                    shape_size = depth.shape[0] * depth.shape[1]
                    if np.count_nonzero(depth < 0.1) < shape_size * 0.1:
                        break
                    num_tries += 1
                if num_tries == 10:
                    continue

                poses.append(view)
                filled = obs["rgb_filled"]
                prefilled = obs["rgb_prefilled"]
                path = os.path.join(
                    views_directory,
                    'floor{:02d}_{:06d}.png'.format(floor, image_idx))
                imageio.imwrite(path, filled)
                image_idx += 1
        total_images += image_idx
    env.reset()
    env.close()
    print('Done rendering {:s}. Images: {:d}. Time: {:2f}'.format(
        model, total_images, time.time() - start))


if __name__ == '__main__':
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play',
        'test_camera.yaml')

    train_models = [
        'Klickitat',
        'Marstons',
        'Hanson',
        'Lakeville',
        'Merom',
        'Lindenwood',
        'Pinesdale',
        'Forkland',
        'Wainscott',
        'Newfields',
        'Ranchester',
        'Hiteman',
        'Leonardo',
        'Onaga',
        'Pomaria',
        'Stockman',
        'Tolstoy',
        'Cosmos',
        'Benevolence',
        'Woodbine',
        'Beechwood',
        'Shelbyville',
        'Mifflinburg',
        'Coffeen',
        'Allensville',
    ]
    val_models = [
        'Darden',
        'Markleeville',
        'Wiconisco',
        'Corozal',
        'Collierville',
    ]
    test_models = [
        'Uvalda',
        'Muleshoe',
        'Noxapater',
        'McDade',
        'Ihlen',
    ]
    all_models = train_models + val_models + test_models

    pool = Pool(processes=1)
    start = time.time()
    for model in all_models:
        output_directory = '/n/scanner/datasets/gibson_scene/{:s}'.format(model)
        #pool.apply_async(render_model, (config_path, model, output_directory))
        try:
            render_model(config_path, model, output_directory)
        except:
            print('Failed')
    pool.close()
    pool.join()
    print('Total time: {.2f} seconds'.format(time.time() - start))
