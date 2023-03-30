import bpy
import time
import numpy as np
import math
import mathutils
import bpy_extras

import os
print(os.getcwd())
print(bpy.data.filepath)
#import operations

t1 = time.time()

scn = bpy.data.scenes[0]
print(scn.render.resolution_x)
scn.render.resolution_x = 1920
scn.render.resolution_y = 1080

bpy.context.scene.cycles.device = 'GPU'

bpy.context.scene.render.image_settings.color_mode = 'RGB'
#bpy.context.scene.render.image_settings.file_format = 'PNG'
#bpy.context.scene.render.image_settings.color_depth = '16'
#bpy.context.scene.render.image_settings.compression = 15

bpy.context.scene.render.image_settings.file_format = 'JPEG'
bpy.context.scene.render.image_settings.quality = 100


print(scn.render.resolution_x)

for obj in scn.objects:
    obj.hide_render = False

print('rendering everything')


def gen_perspective():
#    base_dir = 'D:/Semester 6/Master Thesis/master-thesis/volumes/cad_models'
    base_dir = 'Z:/master-thesis/volumes/cad_models'

    img_count = -1
    
    theta_shift = 0
    phi_shift = 0
    
    for r in (5.5,6.5):
        for theta in np.arange(-80, 80, 25):
            for phi in np.arange(-30, 225, 30):
                img_count+=1
                if not 0 <= theta <= 90:
                    continue
                x, y, z = get_loc(r, theta + theta_shift, phi + phi_shift)
                print(r, theta + theta_shift, phi + phi_shift)
#                x = 2
#                z = 2
#                y = 5
                # Set camera translation
                scn.camera.location.x = x
                scn.camera.location.y = y
                scn.camera.location.z = z
                
#                scn.render.resolution_percentage = 100
#                scn.render.filepath = f"{base_dir}/main_p{img_count}"
#                bpy.context.window.view_layer = bpy.context.scene.view_layers["main"]
#                bpy.ops.render.render(write_still=True)
                
#                return
                
#                scn.render.resolution_percentage = 10
#                scn.render.filepath = f"{base_dir}/topcover_p{img_count}_main"
#                bpy.context.window.view_layer = bpy.context.scene.view_layers["topcover"]
#                bpy.ops.render.render(write_still=True)
                
                scn.render.resolution_percentage = 10
                scn.render.filepath = f"{base_dir}/oilscrew1_p{img_count}_main"
                bpy.context.window.view_layer = bpy.context.scene.view_layers["oilscrew1"]
                bpy.ops.render.render(write_still=True)
#                
#                scn.render.resolution_percentage = 100
#                scn.render.filepath = f"{base_dir}/topunfixed_p{img_count}"
#                bpy.context.window.view_layer = bpy.context.scene.view_layers["topunfixed"]
#                bpy.ops.render.render(write_still=True)
                
                
        theta_shift+=10
        phi_shift+=15

def get_loc(r, theta, phi):
    theta = theta * math.pi / 180
    phi = phi * math.pi / 180

#    tx = scn.objects["Empty_mainbody"].location[0]
#    ty = scn.objects["Empty_mainbody"].location[1]
#    tz = scn.objects["Empty_mainbody"].location[2]
    
    tx = scn.objects["Empty_topcover"].location[0]
    ty = scn.objects["Empty_topcover"].location[1]
    tz = scn.objects["Empty_topcover"].location[2]
    
    
#    cx = scn.camera.location[0]
#    cy = scn.camera.location[1]
#    cz = scn.camera.location[2]
    
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

def exclude_render(objs_list):
#    bpy.context.window.view_layer = bpy.context.scene.view_layers["main"]
    for obj in scn.objects:
        if not obj.name in objs_list:
            obj.hide_render = True
#    for coll in bpy.data.collections:
#        print(coll.name)
#        children = coll.children
#        print(children)
#        for child in children:
#            print(child)
#        coll.hide_render = False
    
#    layer_collection = bpy.context.view_layer.layer_collection
#    print(layer_collection.children)
#    for child in layer_collection.children:
#        child.hide_render = False
#    collection.hide_viewport = True
#    for coll in layer_collection:
#        print(coll)

#print(get_loc(5.5, -55, 0))
#test_view()
#exclude_render([])
#gen_perspective()
t2 = time.time()

print('done!', 'time taken: ', (t2-t1)/60, 'm')

#r = x
#g = y
