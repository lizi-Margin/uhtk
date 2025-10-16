import math
import numpy as np

use_cpp = True
try:
    from .py_c3utils import Vector3, Vector2
except:
    use_cpp = False

if not use_cpp:
    class Vector2  : 
        def __init__(self,list2) -> None:
            self.vec = np.array(list2)

        def prod(self,x : float):
        
            self.vec[0] = self.vec[0] *  x
            self.vec[1] =self.vec[1] * x
            return self

        def add(self,v3):
            v3 = v3.vec
            self.vec[0] += v3[0]
            self.vec[1] += v3[1]
            return self
    

        def __mul__(self, other):
            if isinstance(other, Vector2):
                return Vector2([self.vec[0] * other.vec[0], self.vec[1] * other.vec[1]])
            else:
                return Vector2([self.vec[0] * other, self.vec[1] * other])
        
        def __add__(self, other):
            if isinstance(other, Vector2):
                return Vector2([self.vec[0] + other.vec[0], self.vec[1] + other.vec[1]])
            else:
                return Vector2([self.vec[0] + other, self.vec[1] + other])
        
        def get_np(self):
            return  np.array(self.vec)
    

    ########################################################################## 
        def get_list(self):
            return [self.vec[0],self.vec[1]]
        
        def get_prod(self,v3 ):
            v3 = v3.vec
            product = self.vec[0] *  v3[0]
            product +=self.vec[1] * v3[1]
            return product 
        def get_module(self,non_zero =False):
            mo = self.vec[0]* self.vec[0]
            mo += self.vec[1] * self.vec[1]
            if (non_zero == True and mo ==0 ) : mo = 0.0001

            return math.sqrt(mo)

        def get_angle (self,v3):
            ang =  (self.get_prod(v3))*(1/(self.get_module(non_zero=True) * v3.get_module(non_zero = True)))
            if ang >1 :ang =1
            if ang <-1 : ang =-1
            return math.acos(ang) 
        
        def get_Vector3(self,z=0):
            lis = self.vec.tolist()
            return Vector3([lis[0],lis[1],z])




    class Vector3 :
        def __init__(self,list3) -> None:
            # self.vec = np.array(list3, dtype=np.float64)
            self.vec = np.array(list3, dtype=float)

        def rotate_xyz_self(self,ax,ay,az):
            rotate_matrix =  np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)], [0,math.sin(ax),math.cos(ax)]])
            rotate_matrix = rotate_matrix @ np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
            rotate_matrix = rotate_matrix @ np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
            self.vec = rotate_matrix @ self.vec
            return self

        def rotate_zyx_self(self,ax,ay,az) :   
            mz = np.array([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])
            my =  np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
            mx =  np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)], [0,math.sin(ax),math.cos(ax)]])
            self.vec = np.dot(mz,np.dot(my,mx)) @ self.vec 
            return self

        def rev_rotate_zyx_self(self,ax,ay,az): 
            mxt =  np.array([[1,    0,                   0],
                                    [0,math.cos(ax),math.sin(ax)],
                                    [0,-math.sin(ax),math.cos(ax)]])
            myt =  np.array([[math.cos(ay),0,-math.sin(ay)],
                                                    [0,            1,           0],
                                                    [math.sin(ay),0,math.cos(ay)]])
            mzt = np.array([[math.cos(az),math.sin(az),0],
                                                    [-math.sin(az),math.cos(az),0],
                                                    [0,          0,            1]])
            self.vec = np.dot(mxt,np.dot(myt,mzt)) @ self.vec
            return self


        def rotate_xyz_fix(self,ax,ay,az):
            rotate_matrix =   np.array([[math.cos(az),-math.sin(az),0],
                                        [math.sin(az),math.cos(az),0],
                                        [0,0,1]])  
            rotate_matrix = rotate_matrix @ np.array([[math.cos(ay),0,math.sin(ay)],
                                                    [0,1,0],
                                                    [-math.sin(ay),0,math.cos(ay)]])
            rotate_matrix =  rotate_matrix @ np.array([[1,0,0],
                                                    [0,math.cos(ax),-math.sin(ax)],
                                                    [0,math.sin(ax),math.cos(ax)]])
            self.vec = rotate_matrix @ self.vec
            return self
        def rev_rotate_xyz_fix(self,ax,ay,az):
            rotate_matrix =   np.array([[1,0,0],
                                        [0,math.cos(ax),math.sin(ax)],
                                        [0,-math.sin(ax),math.cos(ax)]])
            rotate_matrix =   rotate_matrix @ np.array([[math.cos(ay),0,-math.sin(ay)],
                                                        [0            ,1,          0],
                                                        [math.sin(ay),0,math.cos(ay)]])
            rotate_matrix =   rotate_matrix @ np.array([[math.cos(az),math.sin(az),0],
                                                        [-math.sin(az),math.cos(az),0],
                                                        [0,           0,           1]])  
            self.vec = rotate_matrix @ self.vec
            return self
    
        def prod(self,x : float):
        
            self.vec[0] = self.vec[0] *  x
            self.vec[1] =self.vec[1] * x
            self.vec[2] = self.vec[2] *x
            return self
        def add(self,v3 ):
            v3 = v3.vec
            self.vec[0] += v3[0]
            self.vec[1] += v3[1]
            self.vec[2] += v3[2]
            return self

    ########################################################################## 
        def get_list(self):
            return [self.vec[0],self.vec[1],self.vec[2]]
        
        def get_prod(self,v3 ):
            v3 = v3.vec
            product = self.vec[0] *  v3[0]
            product +=self.vec[1] * v3[1]
            product += self.vec[2] * v3[2]
            return product 
        def get_module(self,non_zero =False):
            # mo = self.vec[0]* self.vec[0]
            # mo += self.vec[1] * self.vec[1]
            # mo += self.vec[2] * self.vec[2]
            # if (non_zero == True and mo ==0 ) : mo = 0.0001
            # return math.sqrt(mo)
            mo = np.linalg.norm(self.vec)
            if (non_zero == True and mo ==0 ) : mo = 1e-4
            return mo
        def get_angle(self, v3, pid_set_zero=-1, pid_sign_dim:int=None):
            if pid_set_zero == -1:
                dot_product = self.get_dot(v3)
                modules_product = self.get_module(non_zero=True) * v3.get_module(non_zero=True)
                ang = dot_product / modules_product
                if ang > 1.0: ang = 1.0
                if ang < -1.0: ang = -1.0
                return math.acos(ang)

            temp = Vector3(v3.get_list())

            if pid_set_zero == 0:
                set_zero_dim = 0
                sign_dim = 1
            elif pid_set_zero == 1:
                set_zero_dim = 1
                sign_dim = 2
            elif pid_set_zero == 2:
                set_zero_dim = 2
                sign_dim = 0
            else:
                raise ValueError("Invalid pid_set_zero value")

            if pid_sign_dim is not None:
                if 0 <= pid_sign_dim < 3:
                    sign_dim = pid_sign_dim
                else:
                    raise ValueError("Invalid pid_sign_dim value")

            temp.vec[set_zero_dim] = 0

            dot_product = self.get_prod(temp)
            modules_product = self.get_module(non_zero=True) * temp.get_module(non_zero=True)
            ang = dot_product / modules_product
            if ang > 1.0: ang = 1.0
            if ang < -1.0: ang = -1.0

            angle = math.acos(ang)
            if temp.vec[sign_dim] < 0:
                angle = -angle

            return angle

        def get_rotate_angle_fix(self):
            """
            Get the Euler angles (gamma, beta, alpha) needed to rotate this vector
            using fixed-axis rotations (Z-Y-X order).

            Returns:
                list: [gamma, beta, alpha] where:
                    - alpha: rotation around Z-axis (yaw)
                    - beta: rotation around Y-axis (pitch)
                    - gamma: rotation around X-axis (roll, always 0 in this method)
            """
            vx = self.vec[0]
            vy = self.vec[1]
            vz = self.vec[2]

            alpha = math.atan2(vy, vx)
            beta = math.atan2(-vz, math.sqrt(vx*vx + vy*vy))
            gamma = 0.0

            return [gamma, beta, alpha]

        def get_dot(self, v3):
            """Compute dot product with another vector (alias for get_prod)."""
            return self.get_prod(v3)

        def get_Vector2(self):
            lis = self.vec.tolist()
            return Vector2([lis[0],lis[1]])

        def __add__(self, other):
            """Add two vectors."""
            if isinstance(other, Vector3):
                return Vector3((self.vec + other.vec).tolist())
            else:
                raise TypeError("Unsupported operand type for +")

        def __sub__(self, other):
            """Subtract two vectors."""
            if isinstance(other, Vector3):
                return Vector3((self.vec - other.vec).tolist())
            else:
                raise TypeError("Unsupported operand type for -")

        def __eq__(self, other):
            """Check equality with another vector."""
            if not isinstance(other, Vector3):
                return False
            if np.array_equal(self.vec, other.vec):
                return True
            # Use approximate equality with absolute precision
            abs_precision = 1e-9
            return np.allclose(self.vec, other.vec, atol=abs_precision, rtol=0)

        def __getitem__(self, index):
            """Get vector component by index."""
            if 0 <= index < 3:
                return self.vec[index]
            else:
                raise IndexError("Index out of range")

        def __setitem__(self, index, value):
            """Set vector component by index."""
            if 0 <= index < 3:
                self.vec[index] = value
            else:
                raise IndexError("Index out of range")
        
        def __str__(self):
            """Return string representation of vector."""
            return f"Vector3({self.vec.tolist()})"
        
        def __repr__(self):
            """Return string representation of vector."""
            return f"Vector3({self.vec.tolist()})"
        
        def copy(self):
            """Return a copy of the vector."""
            return Vector3(self.vec.tolist())

        def normalize(self):
            """Normalize the vector to unit length."""
            module = self.get_module(non_zero=False)
            if module == 0:
                raise ValueError("Cannot normalize zero vector")
            self.vec = self.vec / module
            return self



def dict_obj(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(top, key, dict_obj(value))
        elif isinstance(value, seqs):
            setattr(top, key, type(value)(dict_obj(x) if isinstance(x, dict) else x for x in value))
        else:
            setattr(top, key, value)
    return top 


def get_NED_from_NEU(x,mid=np.array([0,0,5000])):
    if mid != np.array([0,0,5000]) : raise NotImplementedError
    h = x[2]
    z = -(h- mid[2])
    x[2] = z
    return x


def NEU_to_self(neu:list,roll,pitch,yaw):
    intent_heading = neu
    intent_heading[2] = -intent_heading[2]  #-------------

    intent_heading = Vector3(intent_heading)
    intent_heading.rev_rotate_zyx_self(roll,pitch,yaw)
    intent_heading = intent_heading.get_list()

    intent_heading[2] = -1* intent_heading[2]   # ----------------------
    return intent_heading


def float2(list):
    for i in range(len(list)):
        list[i] = round(list[i],2)
    return list

    
def has(list,x):
    for i in range(len(list)):
        if list[i] == x:
            return True
    return False


def has_index(dict :dict, index):
    for ind,inf in dict.items():
        if ind == index:
            return True
    return False


def abs_max(num,m):
    if num>m :
        num = m
    if num<-m:
        num = -m
    return num


def no_neg(num):
    if num <0 :
        num = 0
    return num


# def abs (x):
#     if x < 0:
#         return -x
#     return x


def pwr(x) :
    return x*x  


def norm(x: float, lower_side: float=-1.0, upper_side: float=1.0):
    if (x > upper_side): x = upper_side
    if (x < lower_side): x = lower_side
    return x

def meters_to_feet(meters):
    return meters * 3.28084
def feet_to_meters(feet):
    return feet / 3.28084

def nm_to_meters(nm):
    """Convert nautical miles to meters."""
    return nm * 1852

def meters_to_nm(meters):
    """Convert meters to nautical miles."""
    return meters / 1852
