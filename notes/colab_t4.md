# Проверка llama-cpp-python на Google Colab T4

Эта процедура подтверждает CUDA-сборку и реальное выполнение всех трёх профилей на Tesla T4. Сохраните полный вывод ячеек и сведения о среде рядом с результатами. Успешная установка без проверки GPU-выполнения не считается прохождением.

## 1. Выбрать оборудование и установить CUDA-сборку

В Colab выберите Runtime → Change runtime type → T4 GPU. Затем выполните:

```bash
!nvidia-smi
!python -V
!pip uninstall -y llama-cpp-python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir llama-cpp-python
```

Перезапустите runtime, затем зафиксируйте версии и наличие CUDA:

```bash
!nvidia-smi
!pip show llama-cpp-python
!python -c "import llama_cpp; print('llama_cpp:', getattr(llama_cpp, '__version__', 'unknown'))"
!python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 2. Подготовить код и GGUF-модель

Скопируйте репозиторий и модель GGUF в runtime либо смонтируйте Google Drive. Запишите точный путь и SHA-256 файла модели в логе. Начните с модели, которая гарантированно помещается в 16 GiB VRAM с контекстом 4096.

## 3. Запустить профильный прогон

```bash
!python scripts/test_optimization.py \
  --model-path models/qwen2.5-3b-q4_k_m.gguf \
  --profiles baseline aggressive aggressive_q4 \
  --cases-jsonl data/quality_cases.jsonl \
  --max-tokens 128 --temperature 0 --seed 42 --repeats 3
```

Чтобы отделить влияние KV-квантизации от контекста и FlashAttention, выполните также абляцию:

```bash
!python scripts/test_optimization.py \
  --model-path models/qwen2.5-3b-q4_k_m.gguf \
  --kv-ablation-context 1024 \
  --cases-jsonl data/quality_cases.jsonl \
  --max-tokens 128 --temperature 0 --seed 42 --repeats 3
```

Скрипт создаёт отдельный JSON для каждого профиля в `results/raw_logs/`. Убедитесь, что файлы содержат три повтора каждого кейса и ненулевые метрики. В каждом файле проверьте версию backend, модель/GPU, фактические параметры профиля, TTFT, TPS, пик VRAM, сырой ответ и JSON-проверки.

## 4. Критерии прохождения

- `nvidia-smi` показывает Tesla T4 и занятость GPU растёт во время инференса.
- Лог llama.cpp подтверждает CUDA backend и размещение слоёв на GPU; одного значения `n_gpu_layers=-1` в конфигурации недостаточно.
- Все профили завершают генерацию; заданные KV-типы и FlashAttention принимаются backend без fallback или ошибки.
- JSON-результаты записаны как реальные запуски, а не mock.
- Если CUDA, FlashAttention или Q4 KV не поддерживаются установленной сборкой, сохраните ошибку и точные версии. Не помечайте такую комбинацию как проверенную.

## 5. Зафиксировать окружение

Сохраните вывод `nvidia-smi`, `pip show llama-cpp-python`, Python и CUDA, дату прогона и JSON-логи. После успешной проверки внесите проверенные версии в `requirements.txt` или отдельный файл окружения для Colab.
