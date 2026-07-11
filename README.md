# Quantum Natural Language Processing (QNLP): Sentence Classification with Quantum Circuits

A UMD FIRE quantum machine learning research project exploring **quantum natural language processing** for sentence classification. Sentences are parsed into pregroup-grammar diagrams, compiled to parameterized quantum circuits, and trained to classify the 130-sentence food-vs-IT MC dataset. Building on Quantinuum's companion code for *QNLP in Practice* [1], the project evaluates two additional ansätze adapted from Sim et al.'s circuit catalogue [2] and three enhanced optimizers that beat standard SPSA by **~20% training accuracy** and **~30% test accuracy**.

**Tech:** Python · Jupyter · [DisCoPy](https://discopy.org/) · [Qiskit](https://www.ibm.com/quantum/qiskit) · [pytket](https://docs.quantinuum.com/tket/)

## Repository Structure

```
├── code/
│   ├── mc_task.ipynb              # Full pipeline: shot-based simulation (Qiskit AerBackend) and IonQ QPU integration
│   └── mc_task_simulation.ipynb   # Exact classical simulation (DisCoPy + JAX): the ansatz/optimizer comparison behind the headline results
└── datasets/
    ├── mc_train_data.txt          # Training split (70 sentences)
    ├── mc_dev_data.txt            # Development split (30 sentences)
    └── mc_test_data.txt           # Test split (30 sentences)
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

### Additional ansätze

In addition to the standard IQP ansatz, we evaluated two ansätze adapted from the circuit catalogue of Sim, Johnson & Aspuru-Guzik (2019) [2] — circuits 14 and 15, following lambeq's [`Sim14Ansatz`/`Sim15Ansatz`](https://github.com/CQCL/lambeq) implementations. Both perform on par with IQP (see Results):

| Ansatz | Design |
|---|---|
| **Sim14.1** | Each layer has two sublayers of *n* Ry rotations followed by a ring of *n* parameterized controlled-Rx (CRx) gates, with the ring orientation reversed in the second sublayer. |
| **Sim15.1** | Same layout as Sim14.1, with the parameterized CRx rings replaced by CNOT rings. |

The ansatz comparison lives in `mc_task_simulation.ipynb`; `mc_task.ipynb` uses the IQP ansatz.

### Enhanced optimizers

To improve on standard SPSA we introduce three variants, each converging faster:

- **Enhanced SPSA** — learning rates that adapt over iterations
- **ADAM** — an ADAM optimizer variant
- **Genetic algorithm** — population-based parameter search

## Dataset

The MC ("meaning classification") dataset: 130 English sentences for binary classification (food vs. IT), split into 70 train / 30 dev / 30 test. The dataset was released with Lorenz et al., *QNLP in Practice* [1] ([Quantinuum/qnlp_lorenz_etal_2021_resources](https://github.com/Quantinuum/qnlp_lorenz_etal_2021_resources), GPL-3.0); the files in `datasets/` are unmodified copies from that release. Sentences follow three grammatical structures: `N_TV_N` (47 sentences), `N_TV_ADJ_N` (44), and `ADJ_N_TV_N` (39).

Preprocessing: tokenization → POS tagging → pregroup-grammar conversion → diagram transformation.

## Results

The headline results come from `mc_task_simulation.ipynb` — exact (noiseless) classical simulation of the circuits via DisCoPy's `Circuit.eval()`, JIT-compiled with **JAX** — averaged over **20 runs of 2,000 iterations** each. (`mc_task.ipynb` runs the same pipeline with shot-based simulation on Qiskit's **AerBackend** and includes IonQ QPU integration; it is committed with small demo run counts.)

- Sim14.1 and Sim15.1 performed **on par with the standard IQP ansatz**.
- Enhanced SPSA, ADAM, and the genetic algorithm all **converged faster than standard SPSA**.
- Across all three ansätze, the enhanced optimizers outperformed standard SPSA on both training and test accuracy:

| Metric | Improvement over standard SPSA |
|---|---|
| Training accuracy | **~20% higher** |
| Test accuracy | **~30% higher** |

These numbers come from exact noiseless simulation — even shot noise is absent — so performance on real QPUs will differ due to noise and other hardware effects.

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

## Attribution & License

This project builds on the companion resources released with Lorenz et al., *QNLP in Practice* [1]: [Quantinuum/qnlp_lorenz_etal_2021_resources](https://github.com/Quantinuum/qnlp_lorenz_etal_2021_resources) (GPL-3.0). The dataset files are unmodified copies from that release, and the notebook pipeline (parsing, circuit encoding, and training loop) is derived from its reference implementation. The team's additions are the Sim14.1/Sim15.1 ansatz variants, the three alternative optimizers (Enhanced SPSA, Adam, and a genetic algorithm), and the comparative evaluation across ansatz–optimizer combinations.

Because this repository redistributes and derives from GPL-3.0 material, the derived content is subject to the terms of the [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

## References

1. Lorenz, R., Pearson, A., Meichanetzidis, K., Kartsaklis, D., & Coecke, B. (2023). [QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer](https://jair.org/index.php/jair/article/view/14329/26923). *Journal of Artificial Intelligence Research*. Companion code and data: [Quantinuum/qnlp_lorenz_etal_2021_resources](https://github.com/Quantinuum/qnlp_lorenz_etal_2021_resources).
2. Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://arxiv.org/abs/1905.10876). *Advanced Quantum Technologies*, 2(12).
3. Khatri, N. — [Experimental Comparison of Ansätze for Quantum Natural Language Processing](https://www.cs.ox.ac.uk/people/aleks.kissinger/theses/khatri-thesis.pdf)
4. [SPSA — Qiskit Algorithms documentation](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html)
5. Spall, J. C. (2000). [Adaptive stochastic approximation by the simultaneous perturbation method](https://doi.org/10.1109/cdc.1998.761833). *Proceedings of the 37th IEEE Conference on Decision and Control.*
6. [qml.AdamOptimizer — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html)
