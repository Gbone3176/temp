from absl import app
from absl import flags

FLAGS = flags.FLAGS  # 解析命令行参数，可帮助在不修改源码的情况下选择特定参数来运行程序

# 接受任何输入并将其解释为字符串
flags.DEFINE_string('name', None, 'Your name.')
# 将输入解释为整数型，可选的参数lower_bound和upper_bound用于规定最小值和最大值；如果命令行中的数值超过此范围，则产生FlagError
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
# 将输入解释为浮点型，其他同EFINE_integer
flags.DEFINE_float("weight", None, "Your weight in kg.", lower_bound=0)
# 通常不需要设置参数。True/Flase
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# 获取字符串列表，如果命令行中的值不在此列表中则报错。否则，会将此值赋值给FLAGS.flag
flags.DEFINE_enum('job', 'running', ['running', 'stopped'],
                  'Job status.')  # DEFINE_enum()函数()中各元素分别代表name,default,enum_values,help
# 接受命令行上以逗号分隔的字符串列表，并将它们存储在Python列表对象中
flags.DEFINE_list("food", None, "Your favorite food")

flags.DEFINE_integer("nums", None, "repeat times")

flags.mark_flag_as_required('name')
flags.mark_flag_as_required('nums')


def main(argv):
    if FLAGS.debug:
        print('non-flag arguments:', argv)
    print('Hi', FLAGS.name)
    if FLAGS.age is not None:
        print('You are %d years old, and your job is %s' % (FLAGS.age, FLAGS.job))
    if FLAGS.weight is not None:
        print('Your weight is %d kg' % FLAGS.weight)
    if FLAGS.food is not None:
        print("Your favorite food(s): %s" % FLAGS.food)


if __name__ == '__main__':
    app.run(main)
