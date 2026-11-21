from typing import Tuple
import torch


# 0. Вспомогательные функции валидации и адаптивности
def get_device() -> torch.device:
    """
    Определяет доступное вычислительное устройство (CUDA или CPU).

    Returns:
        torch.device: Объект устройства ('cuda' при наличии GPU, иначе 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_grad_tensor(tensor: torch.Tensor, name: str, require_grad: bool = True) -> None:
    """
    Проверяет тип данных, форму и наличие отслеживания градиентов у тензора.

    Args:
        tensor (torch.Tensor): Проверяемый тензор.
        name (str): Имя переменной для информативного сообщения об ошибке.
        require_grad (bool): Обязателен ли флаг requires_grad=True. По умолчанию True.

    Raises:
        TypeError: Если объект не является torch.Tensor или не имеет тип float.
        ValueError: Если тензор пуст или у него отключено отслеживание градиентов.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Параметр '{name}' должен быть torch.Tensor, получено: {type(tensor)}")

    if tensor.numel() == 0:
        raise ValueError(f"Тензор '{name}' не может быть пустым.")

    if not torch.is_floating_point(tensor):
        raise TypeError(f"Тензор '{name}' должен иметь тип с плавающей точкой, получено: {tensor.dtype}")

    if require_grad and not tensor.requires_grad:
        raise ValueError(f"Для тензора '{name}' должен быть установлен флаг requires_grad=True.")



# 2.1 Простые вычисления с градиентами (8 баллов)
def compute_polynomial_gradients(
    x_val: float, y_val: float, z_val: float, device: torch.device
) -> Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Вычисляет значение функции f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z,
    находит градиенты через PyTorch Autograd и проверяет их аналитически.

    Args:
        x_val (float): Значение переменной x.
        y_val (float): Значение переменной y.
        z_val (float): Значение переменной z.
        device (torch.device): Устройство для вычислений (CPU или CUDA).

    Returns:
        Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]:
            - значение функции f;
            - кортеж градиентов (df/dx, df/dy, df/dz) от PyTorch;
            - кортеж аналитических градиентов (df/dx, df/dy, df/dz).
    """
    # Создаем тензоры на целевом устройстве с отслеживанием градиентов
    x = torch.tensor(float(x_val), device=device, requires_grad=True)
    y = torch.tensor(float(y_val), device=device, requires_grad=True)
    z = torch.tensor(float(z_val), device=device, requires_grad=True)

    validate_grad_tensor(x, "x")
    validate_grad_tensor(y, "y")
    validate_grad_tensor(z, "z")

    # Вычисляем функцию и запускаем обратное распространение
    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    pytorch_grads = (x.grad.item(), y.grad.item(), z.grad.item())

    # Аналитическая проверка
    grad_x_analytic = 2 * x.item() + 2 * y.item() * z.item()
    grad_y_analytic = 2 * y.item() + 2 * x.item() * z.item()
    grad_z_analytic = 2 * z.item() + 2 * x.item() * y.item()
    analytic_grads = (grad_x_analytic, grad_y_analytic, grad_z_analytic)

    return f.item(), pytorch_grads, analytic_grads



# 2.2 Градиент функции потерь (9 баллов)
def compute_mse_gradients(
    x_data: torch.Tensor, y_true: torch.Tensor, w_init: float, b_init: float, device: torch.device
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Вычисляет MSE для линейной модели y_pred = w * x + b,
    находит градиенты по w и b через Autograd и проверяет их аналитически.

    Args:
        x_data (torch.Tensor): Вектор входных признаков.
        y_true (torch.Tensor): Вектор целевых значений.
        w_init (float): Начальное значение веса w.
        b_init (float): Начальное значение смещения b.
        device (torch.device): Устройство для вычислений (CPU или CUDA).

    Returns:
        Tuple[float, Tuple[float, float], Tuple[float, float]]:
            - значение функции потерь MSE;
            - кортеж градиентов (dMSE/dw, dMSE/db) от PyTorch;
            - кортеж аналитических градиентов (dMSE/dw, dMSE/db).

    Raises:
        ValueError: Если размерности x_data и y_true не совпадают или не являются 1D-векторами.
    """
    validate_grad_tensor(x_data, "x_data", require_grad=False)
    validate_grad_tensor(y_true, "y_true", require_grad=False)

    if x_data.ndim != 1 or y_true.ndim != 1:
        raise ValueError(f"Ожидались одномерные тензоры, получено: x={x_data.ndim}D, y={y_true.ndim}D")

    if x_data.shape != y_true.shape:
        raise ValueError(f"Размерности x_data и y_true не совпадают: {x_data.shape} != {y_true.shape}")

    # Переносим данные на целевое устройство и создаем обучаемые параметры
    x_dev = x_data.to(device)
    y_dev = y_true.to(device)
    w = torch.tensor(float(w_init), device=device, requires_grad=True)
    b = torch.tensor(float(b_init), device=device, requires_grad=True)

    # Прямой проход и расчет MSE = (1/n) * Σ(y_pred - y_true)^2
    y_pred = w * x_dev + b
    n = x_dev.shape[0]
    mse = torch.sum((y_pred - y_dev) ** 2) / n

    # Обратный проход
    mse.backward()

    pytorch_grads = (w.grad.item(), b.grad.item())

    # Аналитическая проверка
    dw_analytic = (2.0 / n) * torch.sum((y_pred - y_dev) * x_dev).item()
    db_analytic = (2.0 / n) * torch.sum(y_pred - y_dev).item()
    analytic_grads = (dw_analytic, db_analytic)

    return mse.item(), pytorch_grads, analytic_grads



# 2.3 Цепное правило (8 баллов)
def compute_chain_rule_gradient(
    x_val: float, device: torch.device
) -> Tuple[float, float, float, float]:
    """
    Вычисляет составную функцию f(x) = sin(x^2 + 1) и находит градиент df/dx
    двумя способами (torch.autograd.grad и .backward()), сверяя с аналитической производной.

    Args:
        x_val (float): Точка, в которой вычисляется функция и производная.
        device (torch.device): Устройство для вычислений (CPU или CUDA).

    Returns:
        Tuple[float, float, float, float]:
            (значение f(x), градиент autograd.grad, градиент backward, аналитический градиент).
    """
    x = torch.tensor(float(x_val), device=device, requires_grad=True)
    validate_grad_tensor(x, "x")

    f_x = torch.sin(x**2 + 1)

    # Проверяем через torch.autograd.grad (сохраняем граф для последующего .backward())
    grad_autograd = torch.autograd.grad(outputs=f_x, inputs=x, retain_graph=True)[0]
    f_x.backward()

    # Аналитическая проверка по цепному правилу: f'(x) = cos(x^2 + 1) * 2x
    grad_analytic = (torch.cos(x**2 + 1) * 2 * x).item()

    return f_x.item(), grad_autograd.item(), x.grad.item(), grad_analytic



# Тестирование (проверка на простых примерах)
def run_unit_tests(device: torch.device) -> None:
    """
    Запускает модульные тесты для проверки корректности вычисления градиентов
    и обработки некорректных входных данных.

    Args:
        device (torch.device): Устройство, на котором выполняются тесты.
    """
    print("Запуск проверочных тестов...")

    # Тест 2.1: для x=1, y=1, z=1 -> f = 1+1+1+2 = 5, градиенты = (4, 4, 4)
    f_val, pt_grads, an_grads = compute_polynomial_gradients(1.0, 1.0, 1.0, device)
    assert abs(f_val - 5.0) < 1e-5, "Ошибка значения функции в 2.1"
    for g_pt, g_an in zip(pt_grads, an_grads):
        assert abs(g_pt - g_an) < 1e-5, f"Несовпадение градиентов в 2.1: {g_pt} != {g_an}"

    # Тест 2.2: идеальное совпадение y_pred == y_true -> MSE = 0, градиенты = (0, 0)
    x_test = torch.tensor([1.0, 2.0])
    y_test = torch.tensor([2.0, 4.0])
    mse_val, (dw_pt, db_pt), (dw_an, db_an) = compute_mse_gradients(x_test, y_test, 2.0, 0.0, device)
    assert abs(mse_val) < 1e-5 and abs(dw_pt) < 1e-5 and abs(db_pt) < 1e-5, "Ошибка нулевого MSE в 2.2"
    assert abs(dw_pt - dw_an) < 1e-5 and abs(db_pt - db_an) < 1e-5, "Несовпадение градиентов в 2.2"

    # Тест 2.3: сравнение autograd.grad, backward и аналитики
    _, g_ag, g_bw, g_an = compute_chain_rule_gradient(1.0, device)
    assert abs(g_ag - g_bw) < 1e-5 and abs(g_bw - g_an) < 1e-5, "Несовпадение градиентов в 2.3"

    # Тест обработки ошибок: разные размеры векторов в 2.2
    try:
        compute_mse_gradients(torch.tensor([1.0, 2.0]), torch.tensor([1.0]), 0.5, 0.0, device)
        assert False, "Ожидалось исключение ValueError при несовпадении размерностей!"
    except ValueError:
        pass

    print("Все тесты успешно пройдены!\n")



# Основной блок выполнения
if __name__ == "__main__":
    target_device = get_device()
    print(f"Используемое устройство: {target_device}\n")

    # Запуск автотестов
    run_unit_tests(target_device)

    # Вывод по заданию 2.1 
    print("=== 2.1 Простые вычисления с градиентами ===")
    f_res, (gx_pt, gy_pt, gz_pt), (gx_an, gy_an, gz_an) = compute_polynomial_gradients(
        2.0, 3.0, 4.0, target_device
    )
    print(f"Значение функции f: {f_res}")
    print(f"df/dx: PyTorch = {gx_pt} | Аналитически = {gx_an}")
    print(f"df/dy: PyTorch = {gy_pt} | Аналитически = {gy_an}")
    print(f"df/dz: PyTorch = {gz_pt} | Аналитически = {gz_an}\n")

    #  Вывод по заданию 2.2 
    print("=== 2.2 Градиент функции потерь (MSE) ===")
    x_samples = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_targets = torch.tensor([3.0, 5.0, 7.0, 9.0])
    mse_res, (dw_pt, db_pt), (dw_an, db_an) = compute_mse_gradients(x_samples, y_targets, w_init=0.5, b_init=0.0, device=target_device
)
print(f"Значение MSE: {mse_res}")
print(f"dMSE/dw: PyTorch = {dw_pt} | Аналитически = {dw_an}")
print(f"dMSE/db: PyTorch = {db_pt} | Аналитически = {db_an}\n")
# --- Вывод по заданию 2.3 ---
print("=== 2.3 Цепное правило ===")
fx_res, grad_ag, grad_bw, grad_an = compute_chain_rule_gradient(2.0, target_device)
print(f"Значение функции f(x): {fx_res:.6f}")
print(f"df/dx через torch.autograd.grad: {grad_ag:.6f}")
print(f"df/dx через .backward():         {grad_bw:.6f}")
print(f"df/dx (аналитически):            {grad_an:.6f}")