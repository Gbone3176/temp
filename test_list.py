# from flask import Flask, jsonify
#
# app = Flask(__name__)
#
# @app.route('/users')
# def get_users():
#     users = [
#         {"name": "Alice", "age": 30},
#         {"name": "Bob", "age": 25},
#         {"name": "Charlie", "age": 35}
#     ]
#     return jsonify(users)  # 将Python对象转换为JSON格式的响应
#
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)
VIDEO_SUFFIXES = [".mp4", ".avi"]
data = {'suffix':'abc.jpg'}
suffix = data.get('suffix')
isImage = suffix not in VIDEO_SUFFIXES
print(isImage)