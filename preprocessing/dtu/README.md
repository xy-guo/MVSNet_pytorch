# DTU Dataset

The dataset provides the ground truth point cloud with normal information.

1. screened Poisson surface reconstruction (SPSR) to generate the mesh
    * depth-of-tree is set to 11 in SPSR to acquire the high quality mesh
    * the mesh trimming-factor to 9.5 to alleviate mesh artifacts
2. render the mesh to each view point to generate the depth maps for training
3. 80 scenes. Each scan contains 49 images with 7 different lighting conditions, 
totally 27097 training samples

# split
Training (79): '2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,121, 122, 123, 124, 125, 126, 127, 128'
Validation (18): '3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117'
Evaluation (22): '1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118'