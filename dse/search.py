class SearchEngine:
    def __init__(self, object_function, space, variable_names) -> None:
        '''
        Design Space Exploration. Single object searching for discrete space. In default, brute force searching 
        
        object_function: the object function. Ex. function to compute performance.
        rangelist: range of variables. Ex. [[4, 32], [4, 32]] for array size XxY).
        variable_names: variable names for logging
        '''
        self.object_function = object_function
        self.space = space
        self.variable_names = variable_names

        self.optimum = None
        self.optimal_point = None
    
    def search(self):
        def nestloop(n, level, parameters):
            if level == n:
                metric = self.object_function(parameters)
                if metric < self.optimum:
                    self.optimum = metric
                    self.optimal_point = parameters
                return 
            for i in self.space[level]:
                parameters[level] = i
                nestloop(n, level+1, parameters)
        parameters = [p[0] for p in self.space]
        return nestloop(len(self.space), 0, parameters)
        