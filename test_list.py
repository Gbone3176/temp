#
# None_lst = ['n','m','s','l']
# join_lst = '\n'.join(None_lst)
# for item in join_lst:
#     print(item)


def convert_str_to_num(s: str):
    try:
        # 尝试转换为整数
        if s.isdigit():
            return int(s)
        # 尝试转换为浮点数
        return float(s)
    except ValueError:
        # 如果转换失败，则返回原始字符串
        return s


print(convert_str_to_num("1.00"))
print(convert_str_to_num("iamsb"))
