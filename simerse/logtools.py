
import enum
import abc
import warnings


@enum.unique
class LogVerbosity(enum.Enum):
    """
    Represents log verbosity levels. To be used in the log function.
    """

    # No messages will be logged
    SILENT = 0

    # Only errors will be logged
    ERRORS = 1

    # Only warnings and errors will be logged
    WARNINGS = 2

    # Only messages, warning, and errors wil be logged
    MESSAGES = 3

    # Everything will be logged
    EVERYTHING = 4

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class TempVerbosityLevel:
    """
    Context manager that manages a temporary LogVerbosity level for a Logger
    """

    def __init__(self, logger, verbosity):
        self.current_verbosity = verbosity
        self.old_verbosity = logger.verbosity
        self.logger = logger

    def __enter__(self):
        self.logger._verbosity = self.current_verbosity

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._verbosity = self.old_verbosity


class Logger:
    """
    Abstract base class for objects that log messages
    """

    def __init__(self):
        self.verbosity = LogVerbosity.ERRORS

    def verbosity_temp(self, contextual_verbosity):
        """
        :param contextual_verbosity: The temporary verbosity level to change this Logger's verbosity to
        :return: A context manager that sets the verbosity to contextual_verbosity when entered and restores the
        current verbosity when exited
        """
        return TempVerbosityLevel(self, contextual_verbosity)

    def log(self, message, message_importance=LogVerbosity.MESSAGES):
        """
        Logs a message with a given importance only if this Logger's current verbosity
        is greater than the message's importance.

        If an exception type that is a subclass of Warning is given as the message_importance, the message wil be
        logged as if WARNINGS were given **AND** via warnings.warn with the given Warning type as the warning
        category.

        If an exception type that is not a subclass of Warning is given as the message_importance, the message will be
        logged as if ERRORS were given **AND** an exception of the given type will be constructed and raised.

        If an exception object is given as the message_importance, the exception's message will be logged as if
        ERRORS were given and will have the exception's message appended to it **AND** the exception will be raised

        :param message: The message to log
        :param message_importance: A LogVerbosity level (please don't use SILENT because it doesn't make sense)
        """

        try:
            if self.verbosity >= message_importance:
                self(message, message_importance)
        except AttributeError:
            pass

        # noinspection PyTypeChecker
        if self.verbosity >= LogVerbosity.WARNINGS and issubclass(message_importance, Warning):
            self(message, message_importance)
            warnings.warn(message, message_importance)
        elif self.verbosity >= LogVerbosity.ERRORS and isinstance(message_importance, BaseException):
            self(message + f'\n\n{message_importance.__name__}: {str(message_importance)}', message_importance)
            raise message_importance
        elif self.verbosity >= LogVerbosity.ERRORS and issubclass(message_importance, BaseException):
            self(message, message_importance)
            # noinspection PyCallingNonCallable
            raise message_importance(message)

    @abc.abstractmethod
    def __call__(self, message, importance):
        """
        Actually logs a message
        :param message: The message to log
        :param importance: The message importance as passed to log (so it may be an Exception or Warning type)
        """
        pass


class DefaultLogger(Logger):
    """
    A Logger that logs MESSAGES via the print function, WARNINGS via warning.warn, and errors by raising them
    as an Exception if they were not passed as an exception to be raised
    """

    def __call__(self, message, importance):
        if isinstance(importance, LogVerbosity):
            if importance == LogVerbosity.ERRORS:
                raise Exception(message)
            elif importance == LogVerbosity.WARNINGS:
                warnings.warn(message)
            else:
                print(f'Simerse: {message}')


default_logger = DefaultLogger()
