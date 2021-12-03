import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np

class MyClass1():

    def __init__(self, a: np.array, b: int):

        self.a = a
        self.b = b

    def my_fun_2(self, x: jnp.array):

        return jnp.dot(x,jnp.array(self.a))

def my_fun(x: jnp.array, obj1: MyClass1):

    alpha = jit(obj1.my_fun_2)(x)

    return obj1.b * alpha

obj1 = MyClass1(a = np.arange(2),  b = 3)

derivative = grad(jit(my_fun, static_argnums=(1,)), argnums = 0)(jnp.array([2.0, 3.0]), obj1)
