import bpy
import math
import numpy as np

OBJ_PATH = r"C:\Users\shibe\Cpro\03\mesh.obj"
HDRI_PATH = r"C:\Users\shibe\Cpro\03\river_alcove_4k.exr"
OUT_PATH = r"C:\Users\shibe\Cpro\03\render_cycles_shadowcatcher_fixed.png"


bpy.ops.wm.read_factory_settings(use_empty=True)


bpy.ops.wm.obj_import(filepath=OBJ_PATH)
obj = bpy.context.selected_objects[0]

# Compute the lowest Z coordinate of the mesh.
bpy.context.view_layer.update()
min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)

# Create a plane for Shadow Catcher
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, min_z))
plane = bpy.context.active_object
plane.is_shadow_catcher = True

# World / HDRI lighting setup
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True

nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

env_tex = nodes.new("ShaderNodeTexEnvironment")
env_tex.image = bpy.data.images.load(HDRI_PATH)

bg = nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0

out = nodes.new("ShaderNodeOutputWorld")

links.new(env_tex.outputs["Color"], bg.inputs["Color"])
links.new(bg.outputs["Background"], out.inputs["Surface"])

# Estimate the main light direction from the HDRI.
# The brightest pixel is treated as the sun direction.
img = env_tex.image
pixels = np.array(img.pixels[:]).reshape((img.size[1], img.size[0], 4))
brightness = pixels[:, :, 0:3].sum(axis=2)
y, x = np.unravel_index(np.argmax(brightness), brightness.shape)
u = x / img.size[0]
v = y / img.size[1]

theta = u * 2 * math.pi
phi = v * math.pi

# Convert spherical direction to 3D vector
direction = (
    math.sin(phi) * math.cos(theta),
    math.sin(phi) * math.sin(theta),
    math.cos(phi),
)

# Create a directional sun light
sun_data = bpy.data.lights.new("Sun", type='SUN')
sun_data.energy = 3.0
sun = bpy.data.objects.new("Sun", sun_data)
bpy.context.collection.objects.link(sun)

sun.rotation_euler = (
    math.atan2(direction[2], math.sqrt(direction[0]**2 + direction[1]**2)),
    0,
    math.atan2(direction[1], direction[0]),
)

# Camera setup
cam_data = bpy.data.cameras.new("Camera")
camera = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(camera)
bpy.context.scene.camera = camera

# Place the camera in front of the object with a slight downward angle
camera.location = (0, -3.2, min_z + 1.55)
camera.rotation_euler = (1.3, 0, 0)

# for faster CPU rendering
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32 
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'

scene.render.filepath = OUT_PATH
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024

bpy.ops.render.render(write_still=True)
print("Saved:", OUT_PATH)