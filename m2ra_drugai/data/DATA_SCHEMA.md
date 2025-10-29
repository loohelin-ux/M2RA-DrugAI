M2RA-DrugAI Data Schema: Reaction-Activity Graph (RAG)
This document specifies the data structure for the Reaction-Activity Graph (RAG), which is the central data object used by the M²RA-GNN model. It is based on the torch_geometric.HeteroData format.

Node Types and Features
1. precursor_molecule
Description: Represents the 3-alkenyl-2,3'-bisindole precursor molecule.

Node Features (x): A tensor of size [num_atoms, 32]. Each row is an atom, with features including:

One-hot encoding of atom type (C, N, O, S, Halogen).

Atom degree, formal charge, hybridization.

Aromaticity flag.

3D Coordinates (pos): A tensor of size [num_atoms, 3] for the atom's (x, y, z) coordinates, obtained from RDKit embedding.

2. product_molecule
Description: Represents the final Racemosin B analogue product molecule.

Features: Same structure as precursor_molecule.

3. reaction_condition
Description: A single node representing the combined conditions for the photochemical cyclization.

Node Features (x): A tensor of size [1, 2]. The vector includes:

Normalized reaction temperature.

Normalized reaction time.

One-hot encoding of the solvent (e.g., ACN, DCM, Chloroform).

One-hot encoding of the photosensitizer (e.g., I₂).

Normalized sensitizer concentration.

Edge Types (Relations)
1. ('precursor_molecule', 'reacts_to', 'product_molecule')
Description: Represents the chemical transformation itself.

Edge Attributes (edge_attr): A tensor of size [1, 1] containing the ground-truth reaction yield (a value between 0 and 1).

2. ('reaction_condition', 'controls', 'precursor_molecule')
Description: Links the reaction conditions to the precursor it acts upon.

Edge Attributes: None. The relationship is captured by the existence of the edge.

3. ('product_molecule', 'exhibits', 'activity_target')
Description: Links the final product to its measured biological activity.

Edge Attributes (edge_attr): A tensor of size [1, 3] containing:

Ground-truth IC₅₀ value (log-transformed and normalized).

Ground-truth autophagy inhibition rate (normalized).
