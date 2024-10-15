import json

from schemas import UserSchema
from models import User

# 创建一个用户对象
user = User(id=1, name='John Doe', email='john@example.com')

# 创建序列化类实例
user_schema = UserSchema()

# 序列化用户对象
user_data = user_schema.dump(user)

# 打印序列化后的数据
print("序列化:",user_data)

# 反序列化 JSON 数据
with open('data.json', 'r') as file:
    data = json.load(file)

user_schema = UserSchema()
user_obj = user_schema.load(data)

# 打印反序列化后的对象
print("反序列化：",user_obj)  # 输出: {'id': 2, 'name': 'Jane Doe', 'email': 'jane@example.com'}




