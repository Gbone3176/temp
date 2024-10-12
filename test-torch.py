import torch

def test_torch():
    # 创建两个张量
    tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])

    # 执行矩阵乘法
    result = torch.mm(tensor_a, tensor_b)

    # 计算预期结果
    expected_result = torch.tensor([[4.0, 4.0], [10.0, 8.0]])

    # 检查结果是否与预期相符
    if torch.equal(result, expected_result):
        print("PyTorch 测试通过，矩阵乘法结果正确。")
    else:
        print("PyTorch 测试失败，矩阵乘法结果不正确。")

if __name__ == "__main__":
    test_torch()
