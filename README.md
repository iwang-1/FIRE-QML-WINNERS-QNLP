  **<h1>Quantum Natural Language Processing (QNLP): Solving Complex Language Problems with Quantum Computing</h1>**
  **<h2>Introduction</h2>**
  <ol>
    <li><b>Motivation:</b> This project explores the application of quantum natural language processing (QNLP) to the task
  of sentence classification. Traditional natural language processing (NLP) often faces challenges in handling the
  complexities of language, particularly with ambiguous grammatical structures and long-range dependencies. QNLP offers a 
  potential advantage by leveraging the principles of quantum mechanics, such as superposition and entanglement, to
  represent and process linguistic information in new ways.</li>
    <li><b>Objectives:</b> The primary objective of this project is to build and evaluate a QML model that can classify
  sentences into predefined categories. Specifically, we focus on a binary classification task, where sentences need to be
  classified as either belonging to class 1 or class 0. We aim to investigate the feasibility of using quantum circuits to
  encode and process pregroup grammar representations of sentences, and to use these representations for
  classification.</li>
    <li><b>Quantum Advantage:</b> Quantum machine learning is explored for this task due to its potential advantages in:
      <ol>
        <li><b>Encoding:</b> Quantum states can represent high-dimensional feature spaces efficiently, potentially better
  capturing the complex relationships in language.</li>
        <li><b>Parallelism:</b> Quantum algorithms can explore multiple possibilities simultaneously, potentially leading
  to faster training and inference.</li>
        <li><b>Expressability:</b> Quantum circuits have the potential to represent more complex functions than classical
  neural networks, enabling the model to learn intricate linguistic patterns.</li>
      </ol>
    </li>
  </ol>

  **<h2>Methods</h2>**
  <ol>
    <li><b>Framework:</b> This project utilizes DisCoPy, a Python library for diagrammatic reasoning, alongside Qiskit, a
  framework for quantum computing. Pytket is employed for quantum circuit optimization and backend interaction.</li>
    <li><b>Model Architecture:</b> Here is a high-level overview of the model:
      <ol>
        <li><b>Pregroup Grammar:</b> Input sentences are parsed into diagrams representing their grammatical structure
  using a pregroup grammar.</li>
        <li><b>Diagram Transformation:</b> The diagrams are transformed to prepare for the quantum encoding by "bending
  nouns around" to simplify the circuit structure.</li>
        <li><b>Quantum Encoding:</b> DisCoPy's CircuitFunctor is used to map the transformed diagrams to quantum circuits.
  Each word in the sentence is encoded using parameterized quantum circuits, primarily based on IQP ansatz with learnable
  parameters.</li>
        <li><b>Measurement and Classification:</b> Measurement of the final quantum state provides a probability
  distribution, used for binary classification.</li>
      </ol>
    </li>
    <li><b>Ansatz Improvements:</b> In addition to the standard IQP ansatz, we propose two new ansatzes that performed
  comparably:
      <ol>
        <li><b>Sim14.1:</b> For an n-qubit circuit, each layer consists of two sublayers of n RZ gates followed by n CNOT
  gates arranged in a ring topology, with the ring of CNOT gates reversed in the second sublayer.</li>
        <li><b>Sim15.1:</b> The same layout as Sim14.1, but all CNOT gates are replaced with CRZ gates, and an additional
  layer of Hadamard gates is added, inspired by the IQP ansatz.</li>
      </ol>
    </li>
    <li><b>Training Improvements:</b> To improve on standard SPSA, we introduce learning rates that adapt over iterations
  (Enhanced SPSA), an ADAM optimizer variant, and a genetic algorithm, each aimed at faster convergence. The model is
  trained against a standard cross-entropy loss function.</li>
    <li><b>Quantum Circuit:</b> The core of the model is a quantum circuit built from DisCoPy's pregroup grammar diagrams.
  Each word is represented by a parametrized quantum gate or circuit. The specific structure depends on the sentence's
  grammatical structure. The circuits are then measured to obtain a probability distribution used for sentence
  classification.</li>
  </ol>
  
  **<h2>Dataset and Preprocessing</h2>**
  <ol>
    <li><b>Data Description:</b> The project utilizes an English, 130-sentence dataset categorizing sentences related to
  food or IT for binary classification. The data is split into training, development, and test sets. The dataset comprises
  sentences with specific grammatical structures, including 'N_TV_N', 'N_TV_ADJ_N', 'ADJ_N_TV_N', and 'ADJ_N_TV_ADJ_N'.
    </li>
    <li>The preprocessing steps include:
      <ol>
        <li><b>Tokenization:</b> Sentences are split into individual words.</li>
        <li><b>Part-of-Speech (POS) Tagging:</b> Words are tagged with their grammatical roles (noun, verb,
  adjective).</li>
        <li><b>Pregroup Grammar Conversion:</b> Sentences are represented as pregroup grammar diagrams.</li>
        <li><b>Diagram Transformation:</b> Pregroup grammar diagrams are transformed ("bending nouns around") for better
  quantum circuit construction.</li>
      </ol>
    </li>
  </ol>

  **<h2>Results</h2>**
  <ol>
    <li><b>Simulation:</b> The experiments were conducted using the AerBackend, a local quantum simulator provided by
  Qiskit. This choice allowed for rapid experimentation and model development before potential deployment on actual quantum
  hardware. While the use of a simulator facilitates faster evaluation, it's important to note that the performance on
  actual QPUs might vary due to noise and other quantum effects.</li>
    <li><b>Key Findings:</b>
      <ol>
        <li>Our two proposed ansatzes, Sim14.1 and Sim15.1, performed on par with the standard IQP ansatz for the sentence
  classification task.</li>
        <li>Adding adaptive learning rates to SPSA (Enhanced SPSA), an ADAM optimizer, and a genetic algorithm each
  converged faster than standard SPSA.</li>
        <li>Across all three ansatzes (IQP, Sim14.1, Sim15.1), the enhanced optimizers outperformed standard SPSA on both
  training and test accuracy.</li>
        <li>Results were averaged over 20 runs of 2000 iterations each on the Qiskit AerBackend simulator.</li>
      </ol>
    </li>
    <li><b>Performance Metrics:</b> Compared to standard SPSA across all algorithms, our optimizations achieved an
  approximate <b>20% increase in training accuracy</b> and a <b>30% increase in testing accuracy</b> on the 130-sentence
  food-vs-IT binary classification task.</li>
  </ol>

  **<h2>Conclusion</h2>**
  <ol>
    <li><b>Summary</b>: This project demonstrated the feasibility of using QNLP for sentence classification. The QML model,
  based on pregroup grammar and quantum circuits, achieved roughly a 20% improvement in training accuracy and a 30%
  improvement in testing accuracy over standard SPSA. The experiments conducted on a quantum simulator indicate the model's
  potential for learning linguistic structures.</li>
    <li><b>Impact</b>: This work contributes to the evolving field of QNLP by providing an approach for encoding and
  processing sentence structures. It adds to the body of evidence exploring the potential of quantum computing for natural
  language understanding.</li>
    <li><b>Future Work</b>: Several areas for future research and improvement exist:
      <ol>
        <li><b>Hardware Implementation:</b> Evaluate the model's performance on actual quantum computers to assess the
  impact of noise and potential quantum advantages.</li>
        <li><b>Model Scaling:</b> Experiment with more complex sentences and larger datasets to assess the scalability of 
  the QNLP model.</li>
        <li><b>Novel Architectures:</b> Explore alternative quantum circuit designs and ansatz for encoding and processing
  linguistic information.</li>
        <li><b>Error Analysis:</b> Conduct a detailed analysis of the model's errors to gain insights into its limitations
  and opportunities for improvement.</li>
        <li><b>Hybrid Models:</b> Investigate hybrid approaches combining classical and quantum models for more robust and
  practical QNLP applications.</li>
      </ol>
    </li>
  </ol>
  
  **<h2>References</h2>**
  <ol>
    <li>View of QNLP in Practice: Running Compositional Models of Meaning on a Quantum Computer. (2024). Jair.org.
  https://jair.org/index.php/jair/article/view/14329/26923</li>
    <li>Khatri, N., & Hall, L. (n.d.). Experimental Comparison of Ansätze for Quantum Natural Language Processing.
  https://www.cs.ox.ac.uk/people/aleks.kissinger/theses/khatri-thesis.pdf</li>
    <li>SPSA (v0.39) | IBM Quantum Documentation. (2017). IBM Quantum Documentation.
  https://docs.quantum.ibm.com/api/qiskit/0.39/qiskit.algorithms.optimizers.SPSA</li>
    <li>Spall, J. C. (2000). Adaptive stochastic approximation by the simultaneous perturbation method. Proceedings of the
  37th IEEE Conference on Decision and Control (Cat. No.98CH36171), 4, 3872–3879.
  https://doi.org/10.1109/cdc.1998.761833</li>
    <li>PennyLane. (n.d.). Qml.AdamOptimizer. qml.AdamOptimizer - PennyLane 0.39.0 documentation.
  https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html</li>
  </ol>

