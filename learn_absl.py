from absl import flags
from absl import app
from main import FLAGS


def learn_absl(args):
    del args
    for i in range(FLAGS.nums):
        print("hello! %s" % FLAGS.name)


if __name__ == "__main__":
    app.run(learn_absl)
