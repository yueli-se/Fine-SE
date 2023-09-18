# FineSE-replication-package

This repository contains the source code and data that we used to perform the experiment in the paper titled "Fine-SE: Integrating Semantic Features and Expert Features for Software Effort Estimations".

## Dataset

- We use business data and open source data to train and test the performance of the baselines and our method named FineSE, the sample data is attached in the `data` directory.

Please follow the steps below to reproduce the result.

## Environment Setup

### Python Environment Setup

Run the following command in terminal (or command line) to prepare the virtual environment.

```shell
conda create -n finese python=3.8
conda activate finese
pip install -r requirements.txt
```

## Experiment Result Replication Guide

### **Baseline Implementation**

There are 3 baselines (i.e., `Expert Estimation`,  `Deep-SE`, and `GPT2SP`). To reproduce the results of baselines, run the following commands:

- Expert Estimation

  ```shell
  python baselines/ExpertEstimation.py
  ```

### **FineSE Implementation**

To  reproduce the results of FineSE on open source data, run the following command:

- EF

```shell
python FineSE/OpenSource/EF/EF.py
```

- SF

```bash
python FineSE/OpenSource/SF/SF.py
```

- FineSE

```bash
python FineSE/OpenSource/FineSE/FineSE.py
```

### **FineSE Implementation**(Cross repo)

To  reproduce the results of FineSE(Cross repo) on business data, first put the data on the corresponding path, and run the following command:

- EF

```shell
python FineSE/Business/Cross_EF.py
```

- SF

```bash
python FineSE/Business/Cross_SF.py
```

- FineSE

```bash
python FineSE/Business/Cross_FineSE.py
```

To  reproduce the results of FineSE(Cross repo) on open source data, first put the data on the corresponding path, and run the following command:

- EF

```shell
python FineSE/OpenSource/Cross_EF/Cross_EF.py
```

- SF

```bash
python FineSE/OpenSource/Cross_SF/Cross_SF.py
```

- FineSE

```bash
python FineSE/OpenSource/Cross_FineSE/Cross_FineSE.py
```

