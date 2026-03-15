# Knowledge Base Consolidator, Taxonomy Builder, & Comparator
A Python-based suite designed to process, clean, and structure Knowledge Bases (KBs). 
This tool specializes in transforming raw triplet extractions into refined, deduplicated, and hierarchically organized knowledge structures, with full support for logical operators.
It also allows for the evaluation of LLM-generated KBs against "Ground Truth" references using standard Information Retrieval metrics and semantic matching.

## 🚀 Key Features
- **Semantic Deduplication**: Uses a hybrid model (Sentence Transformers + Token-Set Ratio) to identify and remove redundant information without losing technical nuances (like logical operators).
- **Operator-Aware Processing**: Custom logic to protect logical symbols (`==`, `!=`, `>=`, etc.) during normalization and clustering, preventing the loss of critical semantic differences.
- **Multi-Stage Consolidation Pipeline**: 
    1. **Normalization**: Standardization of text via lemmatization and case-folding.
    2. **Deduplication**: Elimination of semantic overlaps based on a configurable threshold.
    3. **Clustering**: Grouping of similar predicates to unify the relational vocabulary.
- **Consistency Reporting**: Built-in metrics (Macro/Micro) to evaluate the reliability and stability of the knowledge base.
- **Invertible Taxonomy Construction**: Build hierarchical trees from Object-to-Subject (Normal) or Subject-to-Object (Inverted), maintaining original predicates and handling potential cycles.

## 📋 Prerequisites
- **Python 3.9+**
- Tested on **Linux** environments.

## 🛠️ Installation
1. **Clone the repository**: `git clone <your-repository-url>`
2. **Enter in main folder**: `cd llms4uvl`
3. **Create a virtual environment**: `python -m venv env`
4. **Activate the virtual environment**: `. env/bin/activate`
5. **Install dependencies**: `pip install -r requirements.txt`
6. **Download the NLP language model**: `python -m spacy download en_core_web_sm`


## 📖 How to Use
The suite consists of three main scripts designed to be used in sequence (Consolidate -> Visualize) or for evaluation purposes (Compare).

### 1. Knowledge Base Consolidator
Use this script to clean, normalize, deduplicate, and clustering a raw KB extracted from text or LLMs. It ensures the data is consistent before further processing.

**Command**:
`python main_consolidator.py path/to/raw_kb.csv --threshold 0.92`

- **Input**: A CSV file with columns Iteration, Seed, Run, Subject, Predicate, and Object.
- **Process**: Normalizes text, protects logical operators, removes semantic redundancies, and cluster triplets.
- **Output**: Generates a {file}_consolidated.csv and a terminal report with consistency metrics (Macro/Micro).

### 2. Knowledge Base Comparator
Use this script to evaluate an LLM-generated KB against a "Ground Truth" reference to measure fidelity and performance.

**Command**:
`python main_comparator.py path/to/ground_truth.csv path/to/llm_kb.csv --threshold 0.85`

- **Metrics**: Calculates Precision, Recall, F1 Score, and Hallucination Rate.
- **Key Logic**: Uses Operator-Aware Matching, ensuring that triplets with different logical symbols (e.g., == vs !=) are not marked as matches even if the text is similar.

### 3. Taxonomy Generator
Use this script to visualize the hierarchical structure of your knowledge base. It supports both bottom-up and top-down construction.

**Commands**:

- **Normal construction (Object -> Subject / Child -> Parent)**:
`python main_taxonomy.py path/to/kb.csv --beta 0.5`

- **Inverted construction (Subject -> Object / Parent -> Child)**:
`python main_taxonomy.py path/to/kb.csv --beta 0.5 --inverted`
    
**Arguments:**

- **--beta**: Weight for semantic similarity vs. frequency (default 0.5).

- **--inverted**: If present, starts the hierarchy from the subjects.

**Visual Output**: An iterative tree view with predicates:
```
└── UVL
    ├── [is a]──> variability modeling language
    └── [supports]──> Boolean features
```