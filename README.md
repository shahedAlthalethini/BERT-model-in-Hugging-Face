# Conversational Question Answering with Hugging Face Transformers

This project demonstrates how to build and evaluate a Question Answering (QA) system using the Hugging Face `transformers` library. It specifically explores the difference in performance between a standard QA setup and one that incorporates conversational history to answer follow-up questions.

The core idea is to show how providing context from the ongoing dialogue significantly improves the model's ability to handle ambiguous or dependent questions (e.g., "and?", "how many?").

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Code Breakdown](#code-breakdown)
  - [1. Setup and Data Preparation](#1-setup-and-data-preparation)
  - [2. Loading the QA Model](#2-loading-the-qa-model)
  - [3. Experiment 1: QA without Dialog History](#3-experiment-1-qa-without-dialog-history)
  - [4. Experiment 2: QA with Dialog History](#4-experiment-2-qa-with-dialog-history)
- [Analysis of Results](#analysis-of-results)
- [Conclusion](#conclusion)

## Project Overview

This project uses a pre-trained BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) to perform extractive question answering on a text about the Vatican Library.

We conduct two experiments:
1.  **Stateless QA**: The model answers each question based *only* on the original text. This simulates a system with no memory of the conversation.
2.  **Stateful QA**: The model answers each question based on the original text *plus* the history of all previous questions and answers. This simple technique provides conversational context, allowing the model to resolve ambiguities.

The results clearly show that the stateful approach is far more effective for handling a natural, flowing conversation.

## Requirements

- Python 3.x
- Hugging Face `transformers` library
- PyTorch or TensorFlow (required by `transformers`)

You can install the necessary library using pip:
```bash
pip install transformers torch
# Or for TensorFlow:
# pip install transformers tensorflow
```

## How to Run

1.  Save the code from the prompt into a Python file (e.g., `qa_demo.py`).
2.  Make sure you have the required libraries installed.
3.  Run the script from your terminal:
    ```bash
    python qa_demo.py
    ```
The script will print the output from both experiments, showing the question, the model's answer, and the expected answer for comparison.

## Code Breakdown

### 1. Setup and Data Preparation

First, we define the source text (`context`) and a list of questions and their corresponding ground-truth answers. These are then combined into a list of dictionaries for easy iteration.

```python
context = (
    "The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, "
    "is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, "
    # ... (full text)
)

questions = [
    "When was the Vat formally opened?",
    "what is the library for?",
    # ... (full list of questions)
]
answers = [
    "It was formally established in 1475",
    "research",
    # ... (full list of answers)
]

data = [
    {"context": context, "question": q, "answer": a}
    for q, a in zip(questions, answers)
]
```

### 2. Loading the QA Model

We use the `pipeline` function from Hugging Face, which simplifies using pre-trained models for specific tasks. We load a `bert-large-uncased` model that has been fine-tuned on the SQuAD (Stanford Question Answering Dataset).

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad"
)
```

### 3. Experiment 1: QA without Dialog History

In this first run, we loop through our questions and pass only the original `context` to the model each time. The model has no memory of previous interactions.

```python
# System answers demonstrate without dialog history
for entry in data:
    result = qa_pipeline(question=entry["question"], context=entry["context"])
    print(f"Q: {entry['question']}")
    print(f"A: {result['answer']}")
    print(f"Expected: {entry['answer']}")
    print("-" * 50)
```
**Observation**: This approach works well for direct questions but fails on follow-ups. For example, when asked `Q: and?`, the model has no idea what the question refers to and gives an unrelated answer. Similarly, for `Q: how many?`, it incorrectly extracts "75,000" instead of "five" (referring to the five periods).

### 4. Experiment 2: QA with Dialog History

In the second experiment, we build a `dialog_history` string. After each question is answered, we append both the question and its *expected* answer to this string. For each new question, we provide the model with the original `context` plus the entire `dialog_history`.

```python
# System answers demonstrate with dialog history
dialog_history = ""
for entry in data:
    dialog_history += f"Question: {entry['question']} Answer: {entry['answer']}\n"
    context_with_history = context + "\n" + dialog_history
    result = qa_pipeline(question=entry["question"], context=context_with_history)
    # ... (print results)
```
**Observation**: This method shows a dramatic improvement.
- **`Q: and?`**: The preceding history is `...Answer: history, and law`. The model correctly understands "and?" as a request to continue the list and answers `philosophy, science and theology`.
- **`Q: how many?`**: The preceding history is `...Answer: into periods`. The model now correctly associates "how many?" with "periods" and answers `five`.
- **`Q: what will this allow?`**: The preceding history is `...Answer: digitising manuscripts`. The model correctly infers the outcome and answers `them to be viewed online`.

## Analysis of Results

| Question Type | Without History (Stateless) | With History (Stateful) | Analysis |
| :--- | :--- | :--- | :--- |
| **Direct Question** | ✅ **Correct** | ✅ **Correct** | Both methods handle standalone questions well. |
| `Q: and?` | ❌ **Incorrect** | ✅ **Correct** | The history provides the necessary context for the model to understand the continuation. |
| `Q: how many?` | ❌ **Incorrect** | ✅ **Correct** | The history provides the antecedent (`periods`) for the pronoun-like question. |
| `Q: what will this allow?` | ❌ **Incorrect** | ✅ **Correct** | The history clarifies the subject of the question (`the project started in 2014`). |

This comparison highlights a fundamental challenge in conversational AI: **context management**. While a simple extractive QA model is powerful, its utility is limited without a mechanism to track the dialogue's state. Appending the conversation history is a basic but highly effective strategy to overcome this limitation.

## Conclusion

This project serves as a practical introduction to building more context-aware QA systems. By simply augmenting the input context with the conversation's history, we can significantly enhance the model's ability to handle natural, multi-turn dialogues. This demonstrates that for conversational tasks, the "context" is not just a static document but a dynamic and growing history of the interaction.
