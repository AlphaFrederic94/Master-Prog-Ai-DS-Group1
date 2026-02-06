🔹 Custom Data Structures & Algorithmic Design (Section 4)

This project includes a dedicated implementation of custom data structures as required in Section 4 of the project specification.
These structures are designed to efficiently manage, organize, and analyze patient data, while also demonstrating algorithmic rigor and software engineering best practices.

📌 Implemented Data Structures
1. Hash Table – HashEncoder

A hash-based encoder is implemented to handle categorical variables efficiently.

Purpose: rapid categorical encoding (e.g. age_group)

Average time complexity: O(1) per lookup

Benefit: avoids repeated linear searches during preprocessing

File:

src/structures/hash_encoder.py

2. Tree Structure – PatientBST

A binary search tree (BST) is implemented to organize patient records hierarchically using a numeric key (age).

Purpose: efficient insertion and retrieval of patient records

Demonstrates hierarchical data organization

Supports search by key and ordered traversal

File:

src/structures/patient_tree.py

3. Graph Structure – FeatureGraph

A feature correlation graph is implemented to model relationships between numerical clinical features.

Nodes represent features

Edges represent correlations above a configurable threshold

Used to analyze feature dependencies and potential multicollinearity

Derived features are excluded from the graph to avoid artificial correlations.

File:

src/structures/feature_graph.py

▶️ Demonstration Script

A standalone script is provided to demonstrate the usage of all custom data structures independently from the machine learning pipeline.

Run the demo:

python -m src.structures.demo_structures


Demonstrated outputs:

Hash table encoding of categorical variables

Binary search tree insertion and search on patient records

Feature correlation graph with nodes and significant edges

Demo file:

src/structures/demo_structures.py

🧪 Testing

Unit tests are provided to validate the correctness of the implemented data structures.

Run tests:

python -m pytest -q


All tests pass successfully.

Test files:

tests/test_hash_encoder.py
tests/test_patient_tree.py
tests/test_feature_graph.py

🎯 Benefits of Custom Data Structures

Performance:
Hash tables and trees reduce computational overhead compared to naive approaches.

Scalability:
The same structures remain effective as the number of patients or features increases.

Algorithmic Transparency:
Graph-based representations explicitly expose relationships between features, improving interpretability.

📎 Notes

This section corresponds to Section 4.2 and 4.3 of the project report and is fully aligned with the academic requirements of the course Programming in Python for Data Science and Artificial Intelligence.
