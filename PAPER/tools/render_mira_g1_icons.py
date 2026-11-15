#!/usr/bin/env python3
"""Render genuine local Unitree G1 meshes for the schematic, on CPU.

The robot pictures are illustrative icons, not experimental rollout evidence.
MuJoCo's OSMesa backend is forced, so Isaac Lab capture can run independently.
"""
from __future__ import annotations
import os
os.environ['MUJOCO_GL']='osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
from pathlib import Path
import hashlib,json,pickle,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import mujoco

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'PAPER/figures/assets/g1'
XML=ROOT/'data/assets/g1/g1.xml'
MOTION=ROOT/'data/motions/g1/g1_walk.pkl'

def render(kind):
    tree=ET.fromstring(XML.read_text())
    compiler=tree.find('compiler'); compiler.set('meshdir',str(XML.parent))
    for geom in tree.iter('geom'):
        if geom.get('type')!='mesh': geom.set('group','5')
        else:
            original=np.array([float(v) for v in geom.get('rgba','0.7 0.7 0.7 1').split()])
            if original[:3].mean()>.4:
                geom.set('rgba','0.52 0.59 0.66 1' if kind=='reference' else '0.14 0.34 0.56 1')
            else: geom.set('rgba','0.10 0.13 0.17 1')
            geom.set('group','1')
    visual=ET.SubElement(tree,'visual')
    ET.SubElement(visual,'global',offwidth='1280',offheight='720')
    ET.SubElement(visual,'quality',shadowsize='4096',offsamples='8')
    ET.SubElement(visual,'headlight',ambient='0.38 0.38 0.38',diffuse='0.65 0.65 0.65',specular='0.25 0.25 0.25')
    ET.SubElement(visual,'rgba',haze='0.95 0.96 0.98 1')
    world=tree.find('worldbody')
    ET.SubElement(world,'light',name='key',pos='-1 -2 4',dir='0.2 0.4 -1',diffuse='0.75 0.76 0.80',specular='0.35 0.35 0.35',castshadow='true')
    ET.SubElement(world,'light',name='fill',pos='2 1 3',dir='-0.5 -0.2 -1',diffuse='0.35 0.38 0.42',castshadow='false')
    assets=tree.find('asset')
    ET.SubElement(assets,'texture',name='paper_sky',type='skybox',builtin='flat',rgb1='0.94 0.95 0.97',rgb2='0.94 0.95 0.97',width='512',height='512')
    ET.SubElement(assets,'texture',name='floor_checker',type='2d',builtin='checker',rgb1='0.91 0.93 0.95',rgb2='0.85 0.88 0.91',width='512',height='512')
    ET.SubElement(assets,'material',name='floor',texture='floor_checker',texrepeat='5 5',reflectance='0.05',specular='0.12',shininess='0.35')
    ET.SubElement(world,'geom',name='paper_floor',type='plane',size='20 20 .1',pos='0 0 0',material='floor',group='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    data=mujoco.MjData(model)
    frames=np.array(pickle.load(MOTION.open('rb'))['frames'])
    frame=frames[31].copy()
    assert model.nq==frame.size+1,(model.nq,frame.shape)
    data.qpos[:3]=frame[:3];data.qpos[:2]=0
    xyzw=Rotation.from_rotvec(frame[3:6]).as_quat()
    data.qpos[3:7]=xyzw[[3,0,1,2]]
    data.qpos[7:]=frame[6:]
    mujoco.mj_forward(model,data)
    # Adjust the kinematic pose to touch the ground without stepping physics.
    ankle_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    data.qpos[2] += .045-min(data.xpos[i,2] for i in ankle_ids)
    mujoco.mj_forward(model,data)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[0,0,.66];cam.distance=1.95;cam.azimuth=135;cam.elevation=-9
    opt=mujoco.MjvOption();opt.geomgroup[:]=False;opt.geomgroup[:2]=True
    renderer=mujoco.Renderer(model,height=720,width=1280)
    renderer.update_scene(data,camera=cam,scene_option=opt)
    image=renderer.render();renderer.close()
    Image.fromarray(image).save(OUT/f'g1_{kind}.png')
    return {'image':str(OUT/f'g1_{kind}.png'),'frame_index':31,'camera':{'lookat':cam.lookat.tolist(),'distance':cam.distance,'azimuth':cam.azimuth,'elevation':cam.elevation},'qpos':data.qpos.tolist()}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[render(k) for k in ['reference','simulated']]
    metadata={'purpose':'Illustrative method diagram only. These images are not experimental robot rollouts.','robot':'Unitree G1 29DOF, authentic local XML and STL assets','model_path':str(XML),'model_sha256':hashlib.sha256(XML.read_bytes()).hexdigest(),'motion_path':str(MOTION),'motion_sha256':hashlib.sha256(MOTION.read_bytes()).hexdigest(),'renderer':'MuJoCo 3.8 OSMesa / CPU, no simulation rollout','license_path':str(XML.parent/'LICENSE.txt'),'renders':records}
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/'UNITREE_LICENSE.txt').write_text((XML.parent/'LICENSE.txt').read_text())
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
