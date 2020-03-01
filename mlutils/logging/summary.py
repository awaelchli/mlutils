from typing import Dict, Callable


class Summary(dict):
    """
    Useful to collect metrics in different places in e.g. the training/validation/test loop.
    All metrics get collected by callbacks into this dictionary which then can be passed
    to a logger.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__callbacks = []

    def collect(self) -> 'Summary':
        """ Collect the summary from the callbacks. """
        for callback in self.__callbacks:
            callback(self)
        return self

    def add_callback(self, fn: Callable[[Dict], None]) -> None:
        """ Adds a callback to the summary that gets invoked when the summary contents get collected. """
        self.__callbacks.append(fn)

    def clear(self) -> None:
        """ Clears the summary. Does not clear the list of callbacks. """
        super().clear()

    def reset(self) -> None:
        """ Clears the summary and the list of callbacks. """
        self.clear()
        self.__callbacks = []
