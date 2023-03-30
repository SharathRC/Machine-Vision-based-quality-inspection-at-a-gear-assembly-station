import bpy


def show_all():
    for obj in scn.objects:
        if obj.type == "MESH":
            obj.hide_viewport = False
            obj.hide_render = False

scn = bpy.data.scenes[0]

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.use_nodes = True
bpy.context.scene.cycles.samples = 1

part = "Gehaeuseschraube"
perc = 12

show_all()

indexPass = 1
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.pass_index = indexPass
        if obj.name.startswith(part):
            obj.pass_index = 0
    indexPass += 1
    print(obj.name, obj.type, obj.pass_index)

# For 2.8x
bpy.context.scene.view_layers["main"].use_pass_object_index = True

tree = bpy.data.scenes['Scene'].node_tree
nodes = tree.nodes
links = tree.links
indexOBOutput = tree.get('IndexOB Output')

#idMaskList = []
#indexPass = 1  # 0 is the default pass

#outputPath = r'C:\Users\YourName\Desktop\trash\\'
#outputPath = 'D:/Semester 6/Master Thesis/master-thesis/volumes'
outputPath = 'Z:/master-thesis/volumes'


fileOutput = nodes.new(type="CompositorNodeOutputFile")
fileOutput.base_path = outputPath
fileOutput.file_slots.remove(fileOutput.inputs[0])


idNode = nodes.new(type='CompositorNodeIDMask')
idNode.index = 0


links.new(nodes.get('Render Layers').outputs.get('IndexOB'), idNode.inputs[0])
fileOutput.file_slots.new(f'{part}_{perc}')
links.new(idNode.outputs[0], fileOutput.inputs[0])


#for obj in bpy.data.objects:
#    print(obj.pass_index, obj.name)
#    if obj.type == 'MESH' and obj.name == 'Gehaeuseschraube1':
#        print('here', obj.name)
#        obj.pass_index = indexPass

#        idNode = nodes.new(type='CompositorNodeIDMask')
#        idNode.index = indexPass
#        print(indexPass)
#        links.new(nodes.get('Render Layers').outputs.get('IndexOB'), idNode.inputs[0])
#        fileOutput.file_slots.new('Object_{}'.format(indexPass))

#        links.new(idNode.outputs[0], fileOutput.inputs[indexPass - 1])

#        indexPass += 1


print('about to render---------------------------------------------')
scn.render.resolution_percentage = 10
bpy.ops.render.render(use_viewport=False)

nodes.remove(idNode)
nodes.remove(fileOutput)