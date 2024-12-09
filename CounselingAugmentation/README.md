# HkCloud Psychiatric Counseling Dialogue Generation

## Requirements

- python==3.8.15
- openai==1.43.0
- pandas==1.5.2
- tqdm==4.64.1

```bash
conda create -n proj-hkcloud python=3.8
conda activate proj-hkcloud
pip install -r requirements.txt
```

## Parameters

- example_path: Path for the example data
- input_path: Path for the input summary data
- output_path: Path for saving the result data
- prompt_path: Path for the prompt as instruction
- use_permutation: Use permutation for generating the dialogue, if True, the example data will be shuffled and generate 20 dialogues based on permutation of example data. If False, the framework will only use example_1 and example_2, and generate 1 dialogues per input summary.
- generation_model: OpenAI model to use for generating the dialogue. The default is 'gpt-4o-mini'.
- generation_temperature: The temperature for generating the dialogue. The default is 1.0.
- generation_top_p: The top_p for generating the dialogue. The default is 1.0.
- generation_max_tokens: The maximum tokens for generating the dialogue. The default is 2048.

## Usage

Use the predefined bash script, and change the parameters as needed.

```bash
bash generate.sh
```
