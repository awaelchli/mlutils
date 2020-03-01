import time


def periodic_step(orig_func):
    """
    A decorator controlling the frequency of function calls
    based on the step and frequency attributes of the parent object.

    The behaviour changes depending on the mode. The parent class is required to implement
    a boolean property 'training'.

    - Training mode:
        This uses the 'current_step' property of the parent class as a counter variable.

    - Evaluation mode:
        In this mode, a separate counter for the wrapped function is incremented
        for every call. Incrementing the counter can be suppressed by passing
        the argument 'commit=False' to the wrapped function.

    In both modes, the call to the function can be forced by passing 'force=True'.

    """
    orig_func.counter = 0

    def wrapper(self, *args, **kwargs):
        forced = kwargs.get('force_log', False) or kwargs.get('force', False)
        commit = kwargs.get('commit', True)

        # In 'eval' mode, use the function counter
        # In 'train' mode, use the 'step'
        if not self.training and commit:
            orig_func.counter += 1

        counter = self.current_step if self.training else orig_func.counter

        if forced or counter % self.frequency == 0:
            orig_func.counter = 0
            return orig_func(self, *args, **kwargs)

    return wrapper


def periodic(orig_func):
    """ A decorator controlling the frequency of function calls. """
    orig_func.counter = 0

    def wrapper(self, *args, **kwargs):
        orig_func.counter += 1
        forced = 'force_log' in kwargs and kwargs['force_log']

        if forced or orig_func.counter % self.frequency == 0:
            return orig_func(self, *args, **kwargs)

    return wrapper


def timed(func):
    """ Decorator that returns the runtime of the wrapped function in seconds. """
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        return elapsed

    return wrapper


def dict2markdown(data, key_title='key', value_title='value'):
    """ Creates a formatted string of a markdown table with two columns (keys, values). """
    md = f'| {key_title} | {value_title} | \n'
    md += '| ' + '-' * 10 + '| ' + '-' * 10 + '| \n'
    for key in data.keys():
        value = data[key]
        md += f'| {key} | {value} | \n'
    return md
