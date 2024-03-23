# llama2-on-cpu

# How to run?

### Step 1

Clone the repository

```bash
git clone https://github.com/venkat5290/llama2-on-cpu.git
```

### Step 2
``` bash
conda create -n llamacpu python=3.8 -y
```

```bash
conda activate llamacpu
```

```bash
pip install -r requirements.txt
```

### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```bash
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

