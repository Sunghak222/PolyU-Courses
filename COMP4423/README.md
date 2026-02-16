# LEGO-Style Image Generator

A computer vision project that converts images and live camera input into LEGO-style renderings using grid-based brick placement and adaptive color merging.

This project was developed as part of a Computer Vision assignment, with a focus on building a complete, modular, and robust pipeline rather than optimizing purely for visual perfection.

---

## 🧱 Features

- Convert static images into LEGO-style outputs
- Support both grayscale (3-color) and full RGB modes
- Multiple brick sizes (1x1, 1x2, 2x2, 2x4, etc.)
- Greedy tiling-based brick placement
- Adaptive color merging using Manhattan distance
- Stud and shadow rendering for 3D effect
- Brick summary (total count and per-type count)
- Real-time camera deployment

---

## 📸 Example Outputs

<img width="900" height="504" alt="output" src="https://github.com/user-attachments/assets/7c717783-3a81-4100-b360-163e5e709819" />

- Original image
- Task 2 (3-color 1x1 LEGO)
- Task 3 (multi-size brick rendering)
- Real-time camera output

---

## 🧠 Core Algorithmic Components

### 1. Grid Discretization

The input image is resized to a bounded grid (≤ 100×100).  
All subsequent computations operate on this normalized grid to ensure predictable time complexity:

$$
\[
O(H \times W)
\]
$$

This prevents computation from scaling with raw image resolution.

---

### 2. Adaptive Color Processing

Instead of applying global color quantization, Task 3 performs region-based color merging during brick placement.

Color similarity is measured using Manhattan distance:

\[
D = |R_1 - R_2| + |G_1 - G_2| + |B_1 - B_2|
\]

Pixels within a threshold \( t \) are grouped into the same brick region.

This allows spatially aware color smoothing rather than uniform global reduction.

---

### 3. Greedy Tiling Strategy

Brick placement is modeled as a constrained covering problem.

At each unoccupied grid position:

- Attempt largest brick sizes first
- Validate boundary constraints
- Validate color homogeneity constraint
- Mark region as occupied

This deterministic greedy strategy:

- Avoids exponential global search
- Maintains stable execution time
- Produces near-optimal brick distributions in practice

Time complexity remains approximately:

\[
O(H \times W \times S)
\]

where \( S \) is the number of candidate brick shapes.

---

### 4. Rendering with Structural Awareness

Rendering operates at brick-level abstraction:

- Brick-level boundary separation
- Per-unit stud generation
- Shadow approximation using layered ellipses

Large bricks visually appear unified, rather than segmented pixel blocks.

---

## 📊 Brick Summary Analytics

During tiling, the system records:

- Total brick count
- Distribution per brick type

This provides structural metrics for evaluating compression vs detail preservation.

---

## 🎥 Real-Time Deployment

The pipeline supports live camera input.

Key design for stability:

- Bounded grid normalization
- Deterministic traversal (no backtracking)
- Heuristic tiling instead of global optimization

This ensures consistent per-frame processing time.

---

## 📈 Design Trade-Offs

- Region-based merging vs global quantization
- Greedy heuristic vs exhaustive search
- Visual realism vs computational efficiency
- Parameter sensitivity (threshold tuning)

---

## 🛠 Tech Stack

- Python
- NumPy
- Pillow
- OpenCV

---
