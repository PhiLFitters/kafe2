class ParametricModelBaseMixin(object):
    """
    Mixin class. Defines additional properties and methods to be 'mixed into' another class.
    """
    def __init__(self, model_func, model_parameters, *args, **kwargs):
        # print "ParametricModelBaseMixin.__init__(model_func=%r, model_parameters=%rb, args=%r, kwargs=%r)" % (model_func, model_parameters, args, kwargs)
        self._model_function_handle = model_func
        self.parameters = model_parameters
        super(ParametricModelBaseMixin, self).__init__(*args, **kwargs)

    @property
    def parameters(self):
        return self._model_parameters

    @parameters.setter
    def parameters(self, parameters):
        self._model_parameters = parameters
        self._pm_calculation_stale = True