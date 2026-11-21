import time
from typing import Callable, Tuple, Any
import torch

# 0. Вспомогательные функции валидации и тестирования
def validate_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor = None) -> None:
    """
    Проверяет корректность типов данных, размерностей и устройств входных тензоров.

    Args:
        tensor_a (torch.Tensor): Первый входной тензор.
        tensor_b (torch.Tensor, optional): Второй входной тензор для бинарных операций.

    Raises:
        TypeError: Если аргументы не являются объектами torch.Tensor или имеют нечисловой тип.
        ValueError: Если тензоры пустые, имеют разную форму или находятся на разных устройствах.
    """
    if not isinstance(tensor_a, torch.Tensor):
        raise TypeError(f"Ожидался torch.Tensor для tensor_a, получено: {type(tensor_a)}")

    if tensor_a.numel() == 0:
        raise ValueError("Передан пустой тензор tensor_a.")

    if not torch.is_floating_point(tensor_a):
        raise TypeError(f"Ожидался тензор с плавающей точкой (float), получено: {tensor_a.dtype}")

    if tensor_b is not None:
        if not isinstance(tensor_b, torch.Tensor):
            raise TypeError(f"Ожидался torch.Tensor для tensor_b, получено: {type(tensor_b)}")

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Несовпадение размерностей тензоров: {tensor_a.shape} != {tensor_b.shape}"
            )

        if tensor_a.dtype != tensor_b.dtype:
            raise TypeError(
                f"Несовпадение типов данных тензоров: {tensor_a.dtype} != {tensor_b.dtype}"
            )

        if tensor_a.device != tensor_b.device:
            raise ValueError(
                f"Тензоры находятся на разных устройствах: {tensor_a.device} и {tensor_b.device}"
            )



# 3.1 Подготовка данных (5 баллов)

def create_benchmark_tensors(dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """
    Создает набор трехмерных тензоров заданных размеров, заполненных случайными числами.

    Args:
        dtype (torch.dtype): Тип данных создаваемых тензоров. По умолчанию torch.float32.

    Returns:
        dict[str, torch.Tensor]: Словарь вида {'описание_размера': тензор}.

    Raises:
        TypeError: Если передан неподдерживаемый тип данных.
    """
    if dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"Неподдерживаемый тип данных для бенчмарка: {dtype}")

    shapes = {
        "64 x 1024 x 1024": (64, 1024, 1024),
        "128 x 512 x 512": (128, 512, 512),
        "256 x 256 x 256": (256, 256, 256),
    }

    return {name: torch.randn(*shape, dtype=dtype) for name, shape in shapes.items()}



# 3.2 Функция измерения времени (5 баллов)

def measure_time(func: Callable, *args: Any, device: str = "cpu") -> Tuple[float, Any]:
    """
    Измеряет время выполнения операции на CPU или GPU (CUDA).

    Для GPU используются события torch.cuda.Event() с синхронизацией потока.
    Для CPU используется стандартный таймер time.perf_counter() / time.time().

    Args:
        func (Callable): Выполняемая функция или лямбда-выражение.
        *args (Any): Аргументы, передаваемые в функцию func.
        device (str): Целевое устройство ('cpu' или 'cuda'). По умолчанию 'cpu'.

    Returns:
        Tuple[float, Any]: Кортеж (время_в_миллисекундах, результат_выполнения).

    Raises:
        ValueError: Если указано неизвестное устройство.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Неподдерживаемое устройство '{device}'. Используйте 'cpu' или 'cuda'.")

    # Адаптивность: если запрошена CUDA, но она недоступна, переключаемся на CPU
    use_cuda = (device == "cuda") and torch.cuda.is_available()

    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time_ms = float(start_event.elapsed_time(end_event))
    else:
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000.0

    return elapsed_time_ms, result



# Тестирование на простых примерах

def run_unit_tests() -> None:
    """
    Проверяет корректность работы операций, таймера и обработчиков ошибок на простых примерах.
    """
    print("Запуск проверочных тестов...")

    # 1. Проверка математической корректности операций на простом примере (1 x 2 x 2)
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    b = torch.tensor([[[2.0, 0.0], [1.0, 2.0]]])

    expected_matmul = torch.tensor([[[4.0, 4.0], [10.0, 8.0]]])
    expected_add = torch.tensor([[[3.0, 2.0], [4.0, 6.0]]])
    expected_mul = torch.tensor([[[2.0, 0.0], [3.0, 8.0]]])
    expected_transpose = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    expected_sum = torch.tensor(10.0)

    assert torch.allclose(torch.matmul(a, b), expected_matmul), "Ошибка в матричном умножении!"
    assert torch.allclose(a + b, expected_add), "Ошибка в поэлементном сложении!"
    assert torch.allclose(a * b, expected_mul), "Ошибка в поэлементном умножении!"
    assert torch.allclose(a.transpose(1, 2), expected_transpose), "Ошибка в транспонировании!"
    assert torch.isclose(torch.sum(a), expected_sum), "Ошибка в вычислении суммы элементов!"

    # 2. Проверка работы функции measure_time
    elapsed_ms, res = measure_time(lambda x, y: x + y, a, b, device="cpu")
    assert elapsed_ms >= 0.0, "Время выполнения не может быть отрицательным!"
    assert torch.allclose(res, expected_add), "measure_time вернула неверный результат!"

    # 3. Проверка обработки ошибок (несовпадение размерностей и неверный тип)
    try:
        validate_tensors(torch.randn(2, 2), torch.randn(3, 3))
        assert False, "Ожидалось исключение ValueError при разных размерностях!"
    except ValueError:
        pass

    try:
        validate_tensors([1, 2, 3])  # type: ignore
        assert False, "Ожидалось исключение TypeError при передаче списка вместо тензора!"
    except TypeError:
        pass

    print("Все тесты успешно пройдены!\n")


# 3.3 Сравнение операций (10 баллов)
def run_performance_benchmark() -> None:
    """
    Проводит сравнение производительности 5 операций на CPU и GPU (если доступен)
    для всех подготовленных матриц и выводит результаты в табличном виде.
    """
    has_cuda = torch.cuda.is_available()
    print(f"Статус CUDA: {'Доступна (' + torch.cuda.get_device_name(0) + ')' if has_cuda else 'Недоступна (работаем в режиме CPU)'}")

    tensors_cpu = create_benchmark_tensors(dtype=torch.float32)

    # Прогрев GPU (warm-up) перед основными замерами
    if has_cuda:
        warmup_tensor = torch.randn(64, 64, device="cuda")
        _ = torch.matmul(warmup_tensor, warmup_tensor)
        torch.cuda.synchronize()

    operations = [
        ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
        ("Сложение",            lambda a, b: a + b),
        ("Умножение",           lambda a, b: a * b),
        ("Транспонирование",    lambda a, b: a.transpose(1, 2)),
        ("Сумма элементов",     lambda a, b: torch.sum(a)),
    ]

    for shape_name, tensor_a_cpu in tensors_cpu.items():
        tensor_b_cpu = torch.randn_like(tensor_a_cpu)
        validate_tensors(tensor_a_cpu, tensor_b_cpu)

        print(f"\nМатрица: {shape_name}")
        print(f"{'Операция':<19}| {'CPU (мс)':<8} | {'GPU (мс)':<8} | {'Ускорение'}")
        print("-" * 55)

        if has_cuda:
            tensor_a_gpu = tensor_a_cpu.to("cuda")
            tensor_b_gpu = tensor_b_cpu.to("cuda")
            validate_tensors(tensor_a_gpu, tensor_b_gpu)

        for op_name, op_func in operations:
            # 1. Измеряем время на CPU
            cpu_ms, _ = measure_time(op_func, tensor_a_cpu, tensor_b_cpu, device="cpu")

            # 2. Измеряем время на GPU (если доступен) и 3. Вычисляем ускорение
            if has_cuda:
                gpu_ms, _ = measure_time(op_func, tensor_a_gpu, tensor_b_gpu, device="cuda")
                speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
                gpu_str = f"{gpu_ms:>8.1f}"
                speedup_str = f"{speedup:>7.1f}x"
            else:
                gpu_str = f"{'N/A':>8}"
                speedup_str = f"{'N/A':>8}"

            # 4. Выводим строку таблицы
            print(f"{op_name:<19}| {cpu_ms:>8.1f} | {gpu_str} | {speedup_str}")

        # Освобождаем память GPU после каждого размера тензора
        if has_cuda:
            del tensor_a_gpu, tensor_b_gpu
            torch.cuda.empty_cache()



# 3.4 Анализ результатов (5 баллов)
def print_analysis() -> None:
    """Выводит текстовый анализ результатов сравнения производительности CPU и GPU."""
    analysis = """
=== 3.4 АНАЛИЗ РЕЗУЛЬТАТОВ ===
1. Наибольшее ускорение на GPU:
   Матричное умножение (torch.matmul) получает максимальный прирост скорости (особенно на 64x1024x1024),
   так как имеет высокую вычислительную сложность O(N^3) и эффективно распараллеливается на ядрах GPU.

2. Почему некоторые операции могут быть медленнее на GPU:
   Транспонирование (transpose) меняет только метаданные (strides) без перемещения данных в памяти и на CPU
   выполняется почти мгновенно. На GPU простые операции страдают от накладных расходов на запуск CUDA-ядер.

3. Влияние размера матриц на ускорение: С ростом размерности матриц ускорение GPU увеличивается: на крупных матрицах (1024x1024) видеокарта
   полностью загружает свои вычислительные блоки, а накладные расходы становятся незаметными.

4. Передача данных между CPU и GPU:
   Копирование происходит по шине PCIe, которая работает намного медленнее внутренней памяти GPU.
   Частая пересылка данных между CPU и GPU сводит на нет весь выигрыш от ускорения вычислений.
"""


