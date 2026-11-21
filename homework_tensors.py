from typing import Dict, Tuple
import torch

# 0. Вспомогательные функции адаптивности и валидации

def get_device() -> torch.device:
    """
    Определяет доступное вычислительное устройство (CUDA или CPU).

    Returns:
        torch.device: Объект устройства ('cuda' при наличии GPU, иначе 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_ndim: int = None,
    expected_shape: Tuple[int, ...] = None,
) -> None:
    """
    Проверяет, что переданный объект является непустым тензором PyTorch,
    а также сверяет его размерность (ndim) и форму (shape) при необходимости.

    Args:
        tensor (torch.Tensor): Проверяемый тензор.
        name (str): Имя переменной для информативного сообщения об ошибке.
        expected_ndim (int, optional): Ожидаемое количество осей (размерность).
        expected_shape (Tuple[int, ...], optional): Ожидаемая форма тензора.

    Raises:
        TypeError: Если объект не является экземпляром torch.Tensor.
        ValueError: Если тензор пуст либо его размерность/форма не соответствуют ожидаемым.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Объект '{name}' должен быть torch.Tensor, получено: {type(tensor)}")

    if tensor.numel() == 0:
        raise ValueError(f"Тензор '{name}' не может быть пустым.")

    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(
            f"Неверная размерность тензора '{name}': ожидалось {expected_ndim}D, получено {tensor.ndim}D"
        )

    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Неверная форма тензора '{name}': ожидалось {expected_shape}, получено {tuple(tensor.shape)}"
        )


# 1.1 Создание тензоров (7 баллов)
def create_initial_tensors(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Создает базовые тензоры заданных размеров на указанном устройстве.

    Args:
        device (torch.device): Устройство для размещения тензоров (CPU или CUDA).

    Returns:
        Dict[str, torch.Tensor]: Словарь с созданными тензорами:
            - 'random': 3x4 случайные числа от 0 до 1;
            - 'zeros': 2x3x4 из нулей;
            - 'ones': 5x5 из единиц;
            - 'range': 4x4 числа от 0 до 15.
    """
    random_tensor = torch.rand(3, 4, device=device)
    zeros_tensor = torch.zeros(2, 3, 4, device=device)
    ones_tensor = torch.ones(5, 5, device=device)
    range_tensor = torch.arange(16, device=device).reshape(4, 4)

    return {
        "random": random_tensor,
        "zeros": zeros_tensor,
        "ones": ones_tensor,
        "range": range_tensor,
    }



# 1.2 Операции с тензорами (6 баллов)
def perform_tensor_operations(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Выполняет транспонирование, матричное умножение, поэлементное умножение
    и суммирование элементов для двух согласованных двумерных матриц.

    Args:
        tensor_a (torch.Tensor): Матрица размером (M, N).
        tensor_b (torch.Tensor): Матрица размером (N, M).

    Returns:
        Dict[str, torch.Tensor]: Словарь с результатами операций.

    Raises:
        ValueError: Если матрицы не 2D или их внутренние размерности не согласованы.
    """
    validate_tensor(tensor_a, "tensor_a", expected_ndim=2)
    validate_tensor(tensor_b, "tensor_b", expected_ndim=2)

    if tensor_a.device != tensor_b.device:
        raise ValueError(
            f"Тензоры находятся на разных устройствах: {tensor_a.device} и {tensor_b.device}"
        )

    if tensor_a.shape[1] != tensor_b.shape[0] or tensor_a.shape[0] != tensor_b.shape[1]:
        raise ValueError(
            f"Несогласованные формы матриц: A={tuple(tensor_a.shape)}, B={tuple(tensor_b.shape)}. "
            f"Ожидались (M, N) и (N, M)."
        )

    a_transposed = tensor_a.T
    matmul_ab = tensor_a @ tensor_b
    elementwise_mul = tensor_a * tensor_b.T
    sum_a = tensor_a.sum()

    return {
        "transposed": a_transposed,
        "matmul": matmul_ab,
        "elementwise_mul": elementwise_mul,
        "sum": sum_a,
    }



# 1.3 Индексация и срезы (6 баллов)

def extract_tensor_slices(tensor_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Извлекает срезы из трехмерного тензора размером 5x5x5:
    первую строку, последний столбец, центральную подматрицу 2x2 и элементы с четными индексами.

    Args:
        tensor_3d (torch.Tensor): Трехмерный тензор размером (5, 5, 5).

    Returns:
        Dict[str, torch.Tensor]: Словарь с извлеченными срезами.

    Raises:
        ValueError: Если форма входного тензора отличается от (5, 5, 5).
    """
    validate_tensor(tensor_3d, "tensor_3d", expected_ndim=3, expected_shape=(5, 5, 5))

    first_row = tensor_3d[0, 0, :]
    last_col = tensor_3d[:, :, -1]
    center_submatrix = tensor_3d[2, 1:3, 1:3]
    even_elements = tensor_3d[::2, ::2, ::2]

    return {
        "first_row": first_row,
        "last_col": last_col,
        "center_submatrix": center_submatrix,
        "even_elements": even_elements,
    }



# 1.4 Работа с формами (6 баллов)
def reshape_24_elements_tensor(tensor_1d: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Преобразует тензор из 24 элементов в набор заданных многомерных форм.

    Args:
        tensor_1d (torch.Tensor): Тензор, содержащий ровно 24 элемента.

    Returns:
        Dict[str, torch.Tensor]: Словарь вида {'описание_формы': преобразованный_тензор}.

    Raises:
        ValueError: Если общее количество элементов в тензоре не равно 24.
    """
    validate_tensor(tensor_1d, "tensor_1d")

    if tensor_1d.numel() != 24:
        raise ValueError(
            f"Для преобразования требуется тензор из 24 элементов, получено: {tensor_1d.numel()}"
        )

    target_shapes = {
        "2x12": (2, 12),
        "3x8": (3, 8),
        "4x6": (4, 6),
        "2x3x4": (2, 3, 4),
        "2x2x2x3": (2, 2, 2, 3),
    }

    return {name: tensor_1d.reshape(*shape) for name, shape in target_shapes.items()}


# Тестирование (проверка на простых примерах)
def run_unit_tests(device: torch.device) -> None:
    """
    Проверяет корректность работы всех функций Задания 1 и обработчиков ошибок.

    Args:
        device (torch.device): Устройство для выполнения тестов.
    """
    print("Запуск проверочных тестов для Задания 1...")

    # Тест 1.1: проверка форм и значений созданных тензоров
    created = create_initial_tensors(device)
    assert created["random"].shape == (3, 4)
    assert created["zeros"].shape == (2, 3, 4) and torch.all(created["zeros"] == 0)
    assert created["ones"].shape == (5, 5) and torch.all(created["ones"] == 1)
    assert created["range"].shape == (4, 4) and created["range"][-1, -1].item() == 15

    # Тест 1.2: проверка операций на простом примере 2x3 и 3x2
    a_test = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    b_test = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
    ops_res = perform_tensor_operations(a_test, b_test)

    expected_matmul = torch.tensor([[4.0, 5.0], [10.0, 11.0]], device=device)
    assert ops_res["transposed"].shape == (3, 2)
    assert torch.allclose(ops_res["matmul"], expected_matmul)
    assert ops_res["elementwise_mul"].shape == (2, 3)
    assert torch.isclose(ops_res["sum"], torch.tensor(21.0, device=device))

    # Тест 1.3: проверка размерностей срезов для тензора 5x5x5
    t_5x5x5 = torch.arange(125, device=device).reshape(5, 5, 5)
    slices = extract_tensor_slices(t_5x5x5)
    assert slices["first_row"].shape == (5,)
    assert slices["last_col"].shape == (5, 5)
    assert slices["center_submatrix"].shape == (2, 2)
    assert slices["even_elements"].shape == (3, 3, 3)

    # Тест 1.4: проверка преобразования форм
    reshaped = reshape_24_elements_tensor(torch.arange(24, device=device))
    assert reshaped["2x12"].shape == (2, 12)
    assert reshaped["2x2x2x3"].shape == (2, 2, 2, 3)

    # Тест обработки ошибок: несовпадение размеров в 1.2 и неверное число элементов в 1.4
    try:
        perform_tensor_operations(torch.randn(2, 3, device=device), torch.randn(2, 3, device=device))
        assert False, "Ожидалось исключение ValueError при несогласованных формах матриц!"
    except ValueError:
        pass

    try:
        reshape_24_elements_tensor(torch.arange(10, device=device))
        assert False, "Ожидалось исключение ValueError при количестве элементов != 24!"
    except ValueError:
        pass

    print("Все тесты успешно пройдены!\n")


# Основной блок выполнения

if __name__ == "__main__":
    target_device = get_device()
    print(f"Используемое устройство: {target_device}\n")

    # Запуск автотестов
    run_unit_tests(target_device)

    # 1.1 Создание тензоров
    print(" 1.1 Создание тензоров ")
    tensors_1_1 = create_initial_tensors(target_device)
    print("Случайный тензор (3x4):\n", tensors_1_1["random"], "\n")
    print("Тензор из нулей (2x3x4):\n", tensors_1_1["zeros"], "\n")
    print("Тензор из единиц (5x5):\n", tensors_1_1["ones"], "\n")
    print("Тензор от 0 до 15 (4x4):\n", tensors_1_1["range"], "\n")

    #  1.2 Операции с тензорами
    print("1.2 Операции с тензорами ")
    A = torch.randn(3, 4, device=target_device)
    B = torch.randn(4, 3, device=target_device)
    ops = perform_tensor_operations(A, B)
    print("Транспонированный A:\n", ops["transposed"])
    print("Матричное умножение A и B:\n", ops["matmul"])
    print("Поэлементное умножение A и B.T:\n", ops["elementwise_mul"])
    print("Сумма элементов A:\n", ops["sum"].item(), "\n")
#  1.3 Индексация и срезы 
    print("=== 1.3 Индексация и срезы ===")
    tensor_5 = torch.arange(125, device=target_device).reshape(5, 5, 5)
    slices_1_3 = extract_tensor_slices(tensor_5)
    print("Исходный тензор:\n", tensor_5)
    print(f"\nПервая строка (shape {slices_1_3['first_row'].shape}):\n", slices_1_3["first_row"])
    print(f"\nПоследний столбец (shape {slices_1_3['last_col'].shape}):\n", slices_1_3["last_col"])
    print(f"\nПодматрица 2x2 из центра (shape {slices_1_3['center_submatrix'].shape}):\n", slices_1_3["center_submatrix"])
    print(f"\nЭлементы с четными индексами (shape {slices_1_3['even_elements'].shape}):\n", slices_1_3["even_elements"], "\n")
#  1.4 Работа с формами
    print("=== 1.4 Работа с формами ===")
    tensor_24 = torch.arange(24, device=target_device)
    reshaped_1_4 = reshape_24_elements_tensor(tensor_24)
    print(f"Исходный тензор ({tensor_24.shape}):\n{tensor_24}\n")
    print("Формы после преобразования:")
    for shape_label, reshaped_t in reshaped_1_4.items():
        print(f"- {shape_label}: {reshaped_t.shape}")