import bpy
import time
import numpy as np
import math
import mathutils
import bpy_extras

import json
import os
import sys


T1 = time.time()

# base_dir = 'D:/Semester 6/Master Thesis/master-thesis'
base_dir = 'Z:/master-thesis'
    
f = open(f'{base_dir}/code/scripts/assembly_steps.json', 'r') 
data = json.load(f)

scn = bpy.data.scenes[0]
scn.use_nodes = True

scn.render.engine = 'CYCLES'
scn.render.resolution_x = 1920
scn.render.resolution_y = 1080

scn.cycles.device = 'GPU'

scn.render.image_settings.color_mode = 'RGB'

scn.render.image_settings.file_format = 'JPEG'
scn.render.image_settings.quality = 100

print(scn.render.resolution_x)


for obj in scn.objects:
    obj.hide_render = False


def get_mask_image(part, assembled_parts, outputPath, filename):
    show_all(assembled_parts)
    
    indexPass = 1
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.pass_index = indexPass
            if obj.name.startswith(part):
                obj.pass_index = 0
        indexPass += 1

    scn.view_layers["main"].use_pass_object_index = True

    tree = bpy.data.scenes['Scene'].node_tree
    nodes = tree.nodes
    links = tree.links
    indexOBOutput = tree.get('IndexOB Output')

    fileOutput = nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = outputPath
    fileOutput.file_slots.remove(fileOutput.inputs[0])

    idNode = nodes.new(type='CompositorNodeIDMask')
    idNode.index = 0

    links.new(nodes.get('Render Layers').outputs.get('IndexOB'), idNode.inputs[0])
    fileOutput.file_slots.new(filename)
    links.new(idNode.outputs[0], fileOutput.inputs[0])

    scn.render.resolution_percentage = 50
    scn.cycles.samples = 1
    bpy.ops.render.render(use_viewport=False)

    nodes.remove(idNode)
    nodes.remove(fileOutput)


def get_base_mask_image(assembled_parts, outputPath, filename):
    show_all(assembled_parts)
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.startswith('Plane') and obj.name in assembled_parts:
        # if obj.type == 'MESH' and not obj.name.startswith('Plane'):
            obj.pass_index = 0
        else:
            obj.pass_index = 1

    scn.view_layers["main"].use_pass_object_index = True

    tree = bpy.data.scenes['Scene'].node_tree
    nodes = tree.nodes
    links = tree.links
    indexOBOutput = tree.get('IndexOB Output')

    fileOutput = nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = outputPath
    fileOutput.file_slots.remove(fileOutput.inputs[0])

    idNode = nodes.new(type='CompositorNodeIDMask')
    idNode.index = 0

    links.new(nodes.get('Render Layers').outputs.get('IndexOB'), idNode.inputs[0])
    fileOutput.file_slots.new(filename)
    links.new(idNode.outputs[0], fileOutput.inputs[0])

    scn.render.resolution_percentage = 50
    scn.cycles.samples = 1
    bpy.ops.render.render(use_viewport=False)

    nodes.remove(idNode)
    nodes.remove(fileOutput)



def get_assembled_parts(step):
    step_details = None
    assembled_parts = None
    for i in data['steps']: 
        _id = i['head']['Id']
        _type = i['head']['type']
        if _id == step:
            if _type == 'Bolting':
                step+=1
                continue
            else:
                step_details = i
            break
    assembled_parts = step_details['already_assembled_parts']
    assembled_parts.append('Plane')
    return assembled_parts

def get_parts_without_part(step):
    step_details = None
    assembled_parts = None
    for i in data['steps']: 
        _id = i['head']['Id']
        _type = i['head']['type']
        if _id == step:
            step_details = i
            break
    assembled_parts = step_details['already_assembled_parts']
    assembled_parts.append('Plane')
    return assembled_parts

def get_range(step):
    for i in data['steps']: 
        _id = i['head']['Id']
        _type = i['head']['type']
        if _id == step:
            _range = i['head']['visibilityRange']
            phi_start = _range[0]
            phi_end = _range[1]
            theta_start = _range[2]
            theta_end = _range[3]
            return phi_start, phi_end, theta_start, theta_end, [10.5, 13.5]
    
    return 0, 0, 0, 0, [0, 0]

def get_range_steps(phi_start, phi_end, theta_start, theta_end):
    phi_step = (phi_end - phi_start)/5
    theta_step = (theta_end - theta_start)/5
    return phi_step, theta_step


def get_next_part(step):
    for i in data['steps']: 
        id = i['head']['Id']
        if id == step:
            return i['head']['parentId']

def show_all(assembled_parts):
    for obj in scn.objects:
        obj.hide_viewport = False
        obj.hide_render = False
        if obj.type == "MESH" and not obj.name in assembled_parts:
            obj.hide_viewport = True
            obj.hide_render = True


def gen_perspective(assembled_parts=[], part='main', save_dir='', obj_tag="", theta_start=-50, theta_end=50, theta_step=20, \
                                                                    phi_start=50, phi_end=130, phi_step=20, \
                                                                    r_list=[5.5, 6.5]):
    img_count = -1
    theta_shift = 0
    phi_shift = 0
    
    for r in r_list:
        for theta in np.arange(theta_start, theta_end, theta_step):
            for phi in np.arange(phi_start, phi_end, phi_step):

                img_count+=1
                # if img_count < 25:
                #     continue
                    # return
                # if not 0 <= theta <= 90:
                #     continue
                x, y, z = get_loc(r, theta + theta_shift, phi + phi_shift, obj=part)
                print(r, theta + theta_shift, phi + phi_shift)
                scn.camera.location.x = x
                scn.camera.location.y = y
                scn.camera.location.z = z
                
                prep_for_render(assembled_parts)
                scn.render.resolution_percentage = 120
                scn.cycles.samples = 128
                scn.render.filepath = f"{save_dir}/{part}{obj_tag}_p{img_count}"
                bpy.context.window.view_layer = bpy.context.scene.view_layers["main"]

                bpy.ops.render.render(write_still=True)

                get_mask_image(part, assembled_parts, save_dir, f"{part}{obj_tag}_p{img_count}_ref")
                
                get_base_mask_image(assembled_parts, save_dir, f"{part}{obj_tag}_p{img_count}_ref_base")
                
        theta_shift+=theta_step/2
        phi_shift+=phi_step/2

def get_loc(r, theta, phi, obj=None):
    theta = theta * math.pi / 180
    phi = phi * math.pi / 180
    
    tx = scn.objects[f"Empty_{obj}"].location[0]
    ty = scn.objects[f"Empty_{obj}"].location[1]
    tz = scn.objects[f"Empty_{obj}"].location[2]
    

    # cx = scn.camera.location[0]
    # cy = scn.camera.location[1]
    # cz = scn.camera.location[2]

    x = r * math.cos(theta) * math.cos(phi) + tx
    y = r * math.cos(theta) * math.sin(phi) + ty
    z = r * math.sin(theta) + tz
    
    return x, y, z

def test_view():
    bpy.context.window.view_layer = bpy.context.scene.view_layers["main"]
    objects=[ob for ob in bpy.context.view_layer.objects if ob.visible_get()]
    print(len(objects))
    objects=[ob for ob in bpy.context.view_layer.objects]
    print(len(objects))
    coord = mathutils.Vector((-0.11923, -1.82, 0))
    a = bpy_extras.object_utils.world_to_camera_view(scn, scn.camera, coord)
    print(a)

def set_lighting_colour():
    for obj in scn.objects:
        if obj.type == 'Light' and obj.name.startswith('Light'):
            obj.data.color = (0.906752, 0.970782, 1)
            # obj.data.color = (1, 1, 1)
            # obj.data.color = (1, 0.943, 0.675)

def prep_for_render(objs_list):
    for obj in scn.objects:
        obj.hide_render = False
        obj.hide_viewport = False
    for obj in scn.objects:
        if not obj.type == "MESH":
            continue
        if not obj.name in objs_list:
            obj.hide_render = True
            # obj.hide_viewport = True

def render_only_part(part):
    for obj in scn.objects:
        if not obj.type == "MESH":
            continue
        if obj.name == part:
            obj.hide_render = False
            continue
        obj.hide_render = True


def position_camera(obj):
    scn.camera.constraints["Track To"].target = bpy.data.objects[f"Empty_{obj}"]


def start_object_rendering(step):
    t1 = time.time()
    part_to_assemble = get_next_part(step)
    position_camera(part_to_assemble)
    # scn.camera.constraints["Track To"].target = bpy.data.objects[f"Empty_Adapter"]

    assembled_parts = get_assembled_parts(step+1)
    print(part_to_assemble, "----------------------------")

    print(assembled_parts)

    save_dir = f"{base_dir}/volumes/cad_models/{part_to_assemble}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    phi_start, phi_end, theta_start, theta_end, r_list = get_range(step)
    phi_step, theta_step = get_range_steps(phi_start, phi_end, theta_start, theta_end)
    print(phi_start, phi_end, phi_step, theta_start, theta_end, theta_step, r_list)
    print(len(np.arange(theta_start, theta_end, theta_step)), len(np.arange(phi_start, phi_end, phi_step)))

    gen_perspective(assembled_parts, part_to_assemble, save_dir=save_dir, theta_start=theta_start, theta_end=theta_end, theta_step=theta_step, \
                                                                            phi_start=phi_start, phi_end=phi_end, phi_step=phi_step, \
                                                                            r_list=r_list)


    t2 = time.time()

    print('done!', 'time taken: ', (t2-t1)/60, 'm')


def start_object_absence_rendering(step):
    t1 = time.time()
    part_to_assemble = get_next_part(step)
    position_camera(part_to_assemble)
    # scn.camera.constraints["Track To"].target = bpy.data.objects[f"Empty_Adapter"]

    assembled_parts = get_parts_without_part(step)
    print(part_to_assemble, "----------------------------")
    print(assembled_parts)

    save_dir = f"{base_dir}/volumes/cad_models/{part_to_assemble}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    phi_start, phi_end, theta_start, theta_end, r_list = get_range(step)
    phi_step, theta_step = get_range_steps(phi_start, phi_end, theta_start, theta_end)
    print(phi_start, phi_end, phi_step, theta_start, theta_end, theta_step, r_list)
    print(len(np.arange(theta_start, theta_end, theta_step)), len(np.arange(phi_start, phi_end, phi_step)))

    gen_perspective(assembled_parts, part_to_assemble, save_dir=save_dir, obj_tag="_absent", theta_start=theta_start, theta_end=theta_end, theta_step=theta_step, \
                                                                            phi_start=phi_start, phi_end=phi_end, phi_step=phi_step, \
                                                                            r_list=r_list)


    t2 = time.time()

    print('done!', 'time taken: ', (t2-t1)/60, 'm')

for i in range(5):
    start_object_rendering(step=i)
    start_object_absence_rendering(step=i)

T2 = time.time()

print('done!', 'time taken (ALL): ', (T2-T1)/60, 'm')