# Overview

A comprehensive pipeline for solving **square jigsaw puzzles** using **computer vision** and **machine learning** techniques. The system now implements **three complementary approaches** optimized for different grid sizes:

* **Descriptor-Based Algorithm** (custom, feature-rich) - for 2×2 puzzles
* **Advanced Solver Algorithm** (MSE-based with multi-feature analysis) - for 4×4 and 8×8 puzzles
* **Paper-Based Algorithm** inspired by *Pomeranz et al., 2011*

All approaches are enhanced with advanced preprocessing, robust matching strategies, and detailed visualization tools.

##  Used Techniques and Technologies

### 1. **Intelligent Solver Selection**
- **2×2 puzzles**: Uses `DescriptorBasedAssembler` with MSE-based edge comparison
- **4×4 and 8×8 puzzles**: Uses `AdvancedPuzzleSolver` with sophisticated feature analysis
- **Automatic grid detection** based on filenames (2x2, 4x4, 8x8)

### 2. **Enhanced Feature Analysis** (Advanced Solver)
- **LAB color space** for perceptually uniform color comparison
- **Multi-feature border analysis**: Color + Gradient Magnitude + Gradient Direction + Laplacian edges
- **Weighted distance metrics** with configurable importance weights
- **Border standardization** and reorientation for consistent comparison

### 3. **Improved Architecture**
- Modular design with separate modules for different functionalities
- Better error handling and validation
- Organized output structure with dedicated folders for each puzzle type

## Pipeline Architecture

The system follows a **modular, multi-stage pipeline**, designed for robustness and extensibility.

### Input & Preprocessing

#### Input
* ZIP file containing puzzle images
* Supported grid sizes: **2×2, 4×4, 8×8** (auto-detected)

#### Extraction
* Automatic ZIP extraction with directory preservation
* Images organized based on directory and filename structure

#### Noise Estimation
* Dynamic noise level estimation per image using Laplacian variance
* Noise variance used to control denoising intensity

#### Selective Denoising
* **Adaptive median filtering** only when noise > 150 variance
* Applied **only to high-noise regions** (threshold-based masking)
* Preserves critical edge and boundary information

#### Image Enhancement Pipeline
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - Local contrast enhancement
2. **Luminance-channel sharpening** - Detail preservation
3. **Gamma correction** (γ=0.9) - Global lightness control
4. **Canny edge detection** - Boundary analysis

### Grid Detection & Piece Extraction

#### Grid Size Detection
* Automatic detection from filename and directory patterns
* Pattern matching: "2x2", "4x4", "8x8" in filenames

#### Generic Grid Cropping
* Uniform division into **N×N grid**
* Consistent piece dimensions across all puzzles

### Edge Representation & Feature Extraction

####  Advanced Solver Feature Extraction (4×4, 8×8)
- **Border extraction**: 3-pixel wide borders (configurable)
- **Multi-feature representation**:
  - **Color features**: LAB color space (3 channels)
  - **Gradient magnitude**: Sobel-based edge strength
  - **Gradient direction**: Edge orientation (0-360°)
  - **Edge response**: Laplacian edge detection
- **Feature standardization**: Mean=0, STD=1 normalization

####  Descriptor-Based Approach (2×2)
- **MSE-based edge comparison**
- **LAB + gradient + laplacian features**
- **Brute-force search** for 2×2 permutations
- **Compatibility scoring** with confidence margins

### Matching & Assembly

#### Advanced Solver Matching Process
1. **Border feature extraction** for all pieces and sides
2. **Compatibility matrix generation**:
   - Top-Bottom compatibility
   - Right-Left compatibility
   - Bottom-Top compatibility
   - Left-Right compatibility
3. **Reciprocal best match verification**
4. **Greedy assembly with prioritization**:
   - Prioritize reciprocal matches
   - Weighted compatibility scores
   - Position-based optimization

#### Descriptor-Based Matching (2×2)
- **All permutations evaluation** (4! = 24 possibilities)
- **Total cost calculation** for each arrangement
- **Confidence margin** computation (gap between best and second-best)

### Visualization & Output

#### Comprehensive Visualization Suite
* **Before/After examples** with metrics
* **Grid cutting visualization** with piece numbering
* **Heatmaps** for horizontal and vertical match scores
* **Top matches display** with connecting lines
* **Final assembly** with quality score

#### Organized Output Structure
