# Structure-from-Motion
***Structure from motion (SfM)*** is a photogrammetric range imaging technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals.

## Pipeline
### 1. Correspondence search (Feature Extraction & Descriptor Matching)
The first step is to find matching features between the images. 

	1) Convert RGB images A and B to grayscale images A’ and B’
	2) Detect SIFT(Scale Invariant Feature Transform) features A’ and B’
	3) Find corresponding feature from A’ to B’ using its descriptor

---
	
### 2. Estimate motion (Essential matrix estimation: 5-point algorithm)
Find essential matrix by findEssentialMat(). It internally uses 5-point algorithm to calculate the essential matrix from the matched features.

	1) Randomly select sets of 5 matches (feature correspondence)
	2) Generate E(Hypothesis) and evaluate using other points with pre-defined threshold – epipolar distance
	3) Iterate 1)~2) for hypothesis candidates
	4) Choose the most supportive hypothesis E having the most inliers
	5) By applying overall procedure which is called RANSAC(RANdom SAmple Consensus), we can effectively reject outliers

---

### 3. Pose recovery (Essential matrix decomposition)
The pose from the essential matrix is recovered by recoverPose(). It returns a 3×3 rotation (R) and translation vector (t).

---

### 4. Triangulation
We would now like to reconstruct the geometry. Given two poses the 2D image points can then be triangulated using triangulatePoints().

## Dependencies
```
$ pip install -r requirements.txt
```

## Run
```
$ python main.py --factor $FACTOR
```
**$FACTOR**: feature matching factor; the higher, the more outliers

## Results
### Correspondence search (Feature Extraction & Descriptor Matching)
![1](save/matches(0.7).jpg)

### 3D reconstruction results
While conducting visualization of 3D reconstruction several times, I adjusted feature matching factor. By sweeping the factor from 0.6 to 0.9 by 0.1, result shows that factor 0.7 satisfies both conditions best - 1) rejecting outliers well 2) shows recognizable reconstruction result  

| ![2](save/matches(0.6).jpg) |
|:---:|
| ![3](save/reconstructed_color(0.6).jpg) |
| FACTOR = 0.6 |

| ![4](save/matches(0.7).jpg) |
|:---:|
| ![5](save/reconstructed_color(0.7).jpg) |
| FACTOR = 0.7 |

| ![6](save/matches(0.8).jpg) |
|:---:|
| ![7](save/reconstructed_color(0.8).jpg) |
| FACTOR = 0.8 |

| ![8](save/matches(0.9).jpg) |
|:---:|
| ![9](save/reconstructed_color(0.9).jpg) |
| FACTOR = 0.9 |

### Multiple views of result
| ![10](save/reconstructed_color_front(0.7).jpg) | front view |
|:---:|:---:|
| ![11](save/reconstructed_color_left(0.7).jpg) | side(left) view |
| ![12](save/reconstructed_color_top(0.7).jpg) | top view |

### Using own data
Image(left) was taken on Iphone X. 3D Reconstruction(right) shows 3 separated objects quite well.

| ![13](save/matches_listerine.jpg) | feature matching |
|:---:|:---:|
| ![14](save/reconstructed_listerine.jpg) | reconstructed |

## References
| No. | References |
|:---:|:---:|
|[1] | https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html |
|[2] | https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f |
|[3] | https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv |
|[4] | https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_py_pose.html |
|[5] | https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html |