from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import tensorflow as tf

from toy_experiments.models import GAAN


def main(_):
    model = GAAN()
    model.fit()

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
