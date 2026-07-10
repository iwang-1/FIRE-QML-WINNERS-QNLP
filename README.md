# Quantum Natural Language Processing (QNLP): Sentence Classification with Quantum Circuits

A UMD FIRE quantum machine learning research project exploring **quantum natural language processing** for sentence classification. Sentences are parsed into pregroup-grammar diagrams, compiled to parameterized quantum circuits, and trained to classify a 130-sentence food-vs-IT dataset — with two novel ansätze and three enhanced optimizers that beat standard SPSA by **~20% training accuracy** and **~30% test accuracy**.

**Tech:** Python · Jupyter · [DisCoPy](https://discopy.org/) · [Qiskit](https://www.ibm.com/quantum/qiskit) · [pytket](https://docs.quantinuum.com/tket/)

## Repository Structure

```
├── code/
│   ├── mc_task.ipynb              # Main experiment notebook: model, ansätze, optimizers, evaluation
│   └── mc_task_simulation.ipynb   # Simulation runs on the Qiskit AerBackend
└── datasets/
    ├── mc_train_data.txt          # Training split
    ├── mc_dev_data.txt            # Development split
    └── mc_test_data.txt           # Test split
```

## Motivation

Traditional NLP struggles with ambiguous grammatical structures and long-range dependencies. QNLP offers a potential advantage by leveraging quantum-mechanical principles — superposition and entanglement — to represent and process linguistic information:

- **Encoding** — quantum states can represent high-dimensional feature spaces efficiently, potentially better capturing complex relationships in language.
- **Parallelism** — quantum algorithms can explore multiple possibilities simultaneously, potentially speeding up training and inference.
- **Expressibility** — quantum circuits can represent more complex functions than comparable classical networks, enabling the model to learn intricate linguistic patterns.

**Objective:** build and evaluate a quantum ML model for binary sentence classification, investigating whether quantum circuits can usefully encode and process pregroup-grammar representations of sentences.

## Method

### Model pipeline

1. **Pregroup grammar parsing** — input sentences are parsed into diagrams representing their grammatical structure.
2. **Diagram transformation** — diagrams are simplified ("bending nouns around") to prepare for quantum encoding.
3. **Quantum encoding** — DisCoPy's `CircuitFunctor` maps the transformed diagrams to quantum circuits; each word is encoded as a parameterized circuit (IQP ansatz baseline) with learnable parameters.
4. **Measurement & classification** — measuring the final quantum state yields a probability distribution used for binary classification, trained against cross-entropy loss.

### Novel ansätze

In addition to the standard IQP ansatz, we propose two new ansätze that perform on par with it:

| Ansatz | Design |
|---|---|
| **Sim14.1** | Each layer has two sublayers of *n* RZ gates followed by *n* CNOT gates in a ring topology, with the CNOT ring reversed in the second sublayer. |
| **Sim15.1** | Same layout as Sim14.1, but CNOT gates are replaced with CRZ gates plus an added layer of Hadamard gates, inspired by the IQP ansatz. |

### Enhanced optimizers

To improve on standard SPSA we introduce three variants, each converging faster:

- **Enhanced SPSA** — learning rates that adapt over iterations
- **ADAM** — an ADAM optimizer variant
- **Genetic algorithm** — population-based parameter search

## Dataset

An English, 130-sentence dataset for binary classification (food vs. IT), split into train / dev / test. Sentences follow the grammatical structures `N_TV_N`, `N_TV_ADJ_N`, `ADJ_N_TV_N`, and `ADJ_N_TV_ADJ_N`.

Preprocessing: tokenization → POS tagging → pregroup-grammar conversion → diagram transformation.

## Results

Experiments ran on Qiskit's **AerBackend** simulator, averaged over **20 runs of 2,000 iterations** each.

- Sim14.1 and Sim15.1 performed **on par with the standard IQP ansatz**.
- Enhanced SPSA, ADAM, and the genetic algorithm all **converged faster than standard SPSA**.
- Across all three ansätze, the enhanced optimizers outperformed standard SPSA on both training and test accuracy:

| Metric | Improvement over standard SPSA |
|---|---|
| Training accuracy | **~20% higher** |
| Test accuracy | **~30% higher** |

Performance on real QPUs may vary due to noise and other quantum effects.

## Getting Started

```bash
pip install discopy qiskit pytket jupyter
jupyter notebook code/mc_task.ipynb
```

The notebooks are self-contained: they load the datasets from `datasets/`, build the pregroup-grammar circuits, and run training/evaluation for each ansatz–optimizer combination.

## Future Work

- **Hardware implementation** — evaluate on real quantum computers to assess the impact of noise.
- **Model scaling** — more complex sentences and larger datasets.
- **Novel architectures** — alternative circuit designs and ansätze for encoding linguistic information.
- **Error analysis** — detailed analysis of model errors to find limitations and improvements.
- **Hybrid models** — combining classical and quantum models for more robust, practical QNLP.

## References

1. [QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer](https://jair.org/index.php/jair/article/view/14329/26923) (JAIR, 2024)
2. Khatri, N. — [Experimental Comparison of Ansätze for Quantum Natural Language Processing](https://www.cs.ox.ac.uk/people/aleks.kissinger/theses/khatri-thesis.pdf)
3. [SPSA — Qiskit Algorithms documentation](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html)
4. Spall, J. C. (2000). [Adaptive stochastic approximation by the simultaneous perturbation method](https://doi.org/10.1109/cdc.1998.761833). *Proceedings of the 37th IEEE Conference on Decision and Control.*
5. [qml.AdamOptimizer — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html)
