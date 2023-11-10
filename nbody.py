import bpy
from mathutils import Vector

G = 6.6740831e-11

class planet(bpy.ops.mesh):
    def __init__(self, radius, location, velocity, mass):
        self.object = bpy.ops.mesh.primitive_uv_sphere_add(radius = radius)
        self.object.location = location
        self.velocity = velocity
        self.mass = mass
        
    def update_path(s):
        self.velocity = self.velocity - s.velocity

    def direction(self, planet):
        dir = self.object.location - planet.location
        return dir
    
    def force():

    




#frame rate for calculation
frame_rate = 1 #24 (normally 24 but 1 is faster)
seconds_per_frame = 1 / frame_rate

def n_bodies(scene):
    #import globals in the function scope
    global G
    global seconds_per_frame
    
    #calc current direction between the objects
    direction = o1.location - o2.location
    
    #calc the squared distance 
    d_squared = direction.length_squared
    #keep the direction of the strength
    direction.normalize()
    
    #calc new speed vectors
    v1 = v1 - (direction * (G * m2 / d_squared) * seconds_per_frame)
    v2 = v2 + (direction * (G * m1 / d_squared) * seconds_per_frame)
    
    #calc new locations
    o1.location += v1 * seconds_per_frame
    o2.location += v2 * seconds_per_frame

#get rid of previously set frame_change_pre handlers (if not the same handler may be fired n times)
bpy.app.handlers.frame_change_pre.clear()
#install "two_bodies" as current handler
bpy.app.handlers.frame_change_pre.append(two_bodies)