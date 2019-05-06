"""
This module will serve as the base to optimize the loss functions of the implemented machine 
learning algorithms
"""
import types
import numpy as np
class GradientDescent():
    def __init__(self,func,grad_func,initial_value,nb_iters,alpha,accuracy):
        self.func=func
        self.grad_func=grad_func
        self.initial_value = initial_value
        self.nb_iters=nb_iters
        self.alpha=alpha
        self.accuracy=accuracy
    @property
    def func(self):
        """getter of the attribute func
        Returns:
            LambdaFunction -- lambda function that will be minimized
        """
        return self.__func
    @func.setter
    def func(self,func):
        if not isinstance(func,types.LambdaType):
            raise TypeError("function type is not a lambda function")
        else:
            self.__func=func
    @property
    def grad_func(self):
        """getter of the attribute grad_func
        Returns:
            LambdaFunction -- lambda function representing the gradient of the function to be minimized
        """
        return self.__grad_func
    @grad_func.setter
    def grad_func(self,grad_func):
        if not isinstance(grad_func,types.LambdaType):
            raise TypeError("gradient function type is not a lambda function")
        else:
            self.__grad_func=grad_func
    @property
    def initial_value(self):
        """getter of the attribute initial_value
        Retruns:
            float -- initial value to start the algorithm
        """
        return self.__initial_value
    @initial_value.setter
    def initial_value(self,initial_value):
        if not isinstance(initial_value,float):
            raise TypeError("initial value needs to be a float")
        else:
            self.__initial_value = initial_value

    @property
    def nb_iters(self):
        """getter of the attribute nb_iters
        Retruns:
            int -- number of iterations before stopping the gradient descent
        """
        return self.__nb_iters
    @nb_iters.setter
    def nb_iters(self,nb_iters):
        if not isinstance(nb_iters,int):
            raise TypeError("number of iterations needs to be an integer")
        else:
            self.__nb_iters = nb_iters
    @property
    def alpha(self):
        """getter of the attribute alpha
        Returns:
            float -- step of the descent
        """
        return self.__alpha
    @alpha.setter
    def alpha(self,alpha):
        if not isinstance(alpha,float):
            raise TypeError("alpha needs to be a float")
        else:
            self.__alpha = alpha
    @property
    def accuracy(self):
        """getter of the attribute accuracy
        Returns:
            float -- accuracy of the minimum
        """
        return self.__accuracy
    @accuracy.setter
    def accuracy(self,accuracy):
        if not isinstance(accuracy,float):
            raise TypeError("accuracy needs to be a float")
        else:
            self.__accuracy = accuracy
    def optimize(self):
        x = self.initial_value
        for i in range(self.nb_iters):
            previous_x = x
            x = x - self.alpha * self.grad_func(x)
            diff = x-previous_x
            if(np.linalg.norm(x-previous_x)<self.accuracy):
                break
        return x
