import logging
from functools import wraps
from time import time


class LoggingInfo:

    @staticmethod
    def get_logger():

        # create logger
        logger = logging.getLogger('brain_logger')
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level=logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s - [%(module)s/%(funcName)s] - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

        return logger


def timeit(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))

        return result

    return wrap
