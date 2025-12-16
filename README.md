# Overview

A comprehensive pipeline for solving **square jigsaw puzzles** using **computer vision** and **machine learning** techniques.
The system implements **two complementary approaches**:

* **Descriptor-Based Algorithm** (custom, feature-rich)
* **Paper-Based Algorithm** inspired by *Pomeranz et al., 2011*

Both approaches are enhanced with advanced preprocessing, robust matching strategies, and detailed visualization tools.



##  Pipeline Architecture

The system follows a **modular, multi-stage pipeline**, designed for robustness and extensibility.

## Input & Preprocessing

### Input

* ZIP file containing puzzle images
* Supported grid sizes: **2×2, 4×4, 8×8**

### Extraction

* Automatic ZIP extraction
* Images organized based on directory and filename structure

### Noise Estimation

* Dynamic noise level estimation per image
* Noise variance used to control denoising intensity

### Selective Denoising

* **Adaptive median filtering**
* Applied **only to high-noise regions**
* Preserves critical edge and boundary information

### Image Enhancement

* **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for local contrast
* **Luminance-channel sharpening** for detail preservation
* **Gamma correction** for global lightness control
* **Canny edge detection** for boundary analysis

##  Grid Detection & Piece Extraction

### Grid Size Detection

* Automatic detection based on filename and directory patterns

### Generic Grid Cropping

* Uniform division of the image into an **N×N grid**

### Piece Normalization

* Ensures consistent dimensions across all pieces
* Required for fair descriptor comparison



##  Edge Representation & Feature Extraction

Two **parallel edge compatibility strategies** are implemented.



### 🔹 Descriptor-Based Approach

#### Enhanced Edge Descriptors

* Extract **15-pixel-wide border regions** from each piece edge
* Vertical edges use **1.8× wider borders**

#### Feature Types

* **Color analysis**: RGB, grayscale, Lab-like channels
* **Color opponency channels** (human vision inspired)
* **Multi-scale gradients** (1, 2, 3 pixel scales)
* **Texture features**:

  * Local contrast
  * Entropy
  * LBP approximations

#### Descriptor Properties

* Descriptor length: **150 normalized values**
* Rich, multi-modal edge representation

#### Score Normalization

* Raw similarity scores compressed to avoid saturation
* Strict range: **[0.01, 0.99]**
* Centered around ~0.6 for better ranking stability



### 🔹 Paper-Based Approach (Pomeranz et al., 2011)

#### Compatibility Metrics

* **Lᵖ norm** with optimal parameters:

  * (p = 0.3)
  * (q = 1/16)

#### Prediction-Based Compatibility

* SAFE variant with **vertical context enhancement**

#### Assembly Strategy

* Greedy assembly
* Multi-start initialization
* Full rotation handling

#### Best Buddy Identification

* Reciprocal best matches
* Used for robust puzzle initialization



##  Matching & Assembly

### Pairwise Edge Comparison

* All piece edges compared
* All rotations evaluated

### Mutual Best Buddies

* Matches prioritized only if **reciprocal**
* Significantly increases match confidence

### Multi-Start Assembly

* Multiple seed pieces
* Multiple initial rotations

### Neighbor Consistency Checks

* Validate placements using surrounding pieces
* Early rejection of weak assemblies

### Grid Quality Evaluation

* Final score = **average edge compatibility** across the grid



##  Visualization & Output

### Comprehensive Visualization

* Original image
* Denoised image
* Enhanced image
* Edge-detection overlays

### Grid Cutting Visualization

* Visual confirmation of piece extraction

### Match Heatmaps

* Color-coded compatibility matrices

### Top Match Display

* Best edge matches visualized with connecting lines

### Final Assembly Output

* Reconstructed puzzle
* Overall quality score



##  Design Decisions & Justifications



## Preprocessing Decisions

### 1. Selective Denoising

**Why:**

* Full-image denoising degrades edges

**How:**

* Median filtering applied only when noise variance > **150**

**Impact:**

* Noise reduced without harming edge descriptors



### 2. Multi-Stage Enhancement

* **CLAHE**: Improves local contrast without over-enhancement
* **Luminance sharpening**: Enhances detail without color artifacts
* **Gamma correction**: Improves edge discrimination



### 3. Dynamic Border Width

* Standard edge width: **15 pixels**
* Vertical edges: **1.8× wider**

**Justification:**

* Vertical edges often contain less variation
* Wider borders improve descriptor richness



## Descriptor Design Decisions

### 1. Feature Diversity

* Multiple color representations
* Gradient-based shape cues
* Texture descriptors independent of shape

### 2. Multi-Scale Analysis

* Captures both fine and coarse texture patterns



### 3. Score Limitation Strategy

**Problem:**

* Perfect similarity scores prevent ranking

**Solution:**

* Structured noise addition
* Sigmoid-based compression

**Result:**

* Stable ranking of good vs. excellent matches



## Matching Algorithm Decisions

### 1. Dual Algorithm Strategy

* **Descriptor method**: Detailed texture & color matching
* **Paper method**: Proven mathematical compatibility

**Synergy:**

* Both methods inform confidence estimation



### 2. Multi-Start Assembly

**Problem:**

* Greedy assembly depends on initialization

**Solution:**

* Multiple seeds and rotations
* Light backtracking
* Early termination using consistency checks



### 3. Mutual Best Buddies Advantage

* High-confidence matches
* Strong puzzle initialization
* Reduced error propagation



## Key Innovations

* **Enhanced descriptor system** with wide, multi-feature borders
* **Score normalization strategy** preventing perfect matches
* **Robust assembly** with multi-start and neighbor validation
* **Comprehensive visualization tools** for analysis and debugging


##  Reference

* Pomeranz, D., Shemesh, M., & Ben-Shahar, O. (2011). *A Fully Automated Greedy Square Jigsaw Puzzle Solver*. IEEE CVPR.
