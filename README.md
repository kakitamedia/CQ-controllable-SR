## Key Requirements

- Python 3.12
- CUDA 12.4
- PyTorch 2.2.1

## Training

Train with "Single" Fourier component
```
python train.py --config configs/train-div2k/recurrent_lte.yaml --name your_run_name --gpu gpu_id_to_use
```

Train with "Multiple" Fourier components
```
python train.py --config configs/train-div2k/recurrent_lte_block.yaml --name your_run_name --gpu gpu_id_to_use
```


After tarining, run the test with following command.
```
python test.py --config configs/test/div2k_4.yaml --model path_to_trained_model --save_dir your_save_directory --gpu gpu_id_to_use --num_pred number_of_fourier_component_at_test_time
```