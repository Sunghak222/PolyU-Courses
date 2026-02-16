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

## 🏗 Pipeline Overview

The system follows a structured pipeline:

1. **Image Preprocessing**
   - Resize input to bounded grid size (≤ 100x100)
   - Optional grayscale conversion (Task 2)

2. **Color Processing**
   - Task 2: Explicit 3-level quantization (black, gray, white)
   - Task 3: Region-based color merging during brick placement

3. **Brick Placement**
   - Greedy tiling strategy
   - Largest brick-first placement
   - Color similarity check using Manhattan distance:

     ```
     D = |R1 - R2| + |G1 - G2| + |B1 - B2|
     ```

4. **Rendering**
   - Brick-level boundary rendering
   - Stud drawing with brightness adjustment
   - Shadow effect using layered ellipses

5. **Brick Summary**
   - Total brick count
   - Each Brick count

---

## ⚙️ Algorithm Design

Instead of performing global optimization, the project adopts a deterministic greedy strategy to ensure stable execution time and suitability for real-time deployment.

Color simplification is handled adaptively:
- Large bricks form in visually consistent regions
- Detailed areas naturally produce smaller bricks

This spatially aware merging improves structural realism compared to global color quantization.

---

## 🎥 Real-Time Mode

The project supports live camera input using OpenCV.

- Frames are resized before processing
- Brick placement operates on a bounded grid
- Execution time remains stable per frame

---

## 🧠 Design Decisions

- Region-based color merging instead of global quantization
- Greedy tiling instead of BFS/backtracking
- Brick-level gap rendering for realistic boundaries
- Simplified shadow rendering for real-time performance

---

## 📦 Requirements

- Python 3.9+
- OpenCV
- NumPy
- Pillow

Install dependencies:

```bash
pip install opencv-python numpy pillow
