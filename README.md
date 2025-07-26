#  Unsupervised Learning for AR/VR Systems

This repository contains six independent yet connected tasks showcasing the use of **unsupervised machine learning** techniques in the context of **Augmented and Virtual Reality (AR/VR)**. Each task is implemented from scratch using Python and popular libraries like `scikit-learn`, `OpenCV`, and `matplotlib`.


---

##  Repository Structure

| File                          | Description                                                       |
|------------------------------|-------------------------------------------------------------------|
| `task_1.py`       | Synthetic data clustering using KMeans and DBSCAN                 |
| `task_2.py` | Image segmentation via pixel clustering                          |
| `task_3.py`    | Dimensionality reduction using PCA and t-SNE                      |
| `task_4.py`| Human activity sensor clustering from UCI HAR dataset             |
| `task_5.py` | Visual overlays (bounding boxes) from clustered image regions     |
| `task_6_image_segmentation`| Run and explore real-world repo on KMeans segmentation            |
| `requirements.txt`           | Dependencies for all tasks                                        |
| `README.md`                  | Project documentation                                             |

---

##  Installation

Install all dependencies from `requirements.txt`:

```bash
git clone https://github.com/jeeva-m-21/SpectroV-Tasks
cd SpectroV-Tasks
pip install -r requirements.txt
```

##  Task Implementations

### Task 1: 2D Clustering Comparison (K-Means vs DBSCAN)
**Objective:** Evaluate clustering performance on synthetic datasets  
**Key Insights:**  
- K-Means excels with globular clusters but fails on moon-shaped data
- DBSCAN handles irregular shapes and noise robustly
- Parameter tuning (`eps`/`min_samples`) is critical for density-based methods

 <img width="633" height="478" alt="image" src="https://github.com/user-attachments/assets/cf7debab-c1cb-40c7-b175-f316f200cfb3" />

<img width="634" height="477" alt="image" src="https://github.com/user-attachments/assets/d871b13f-0806-4c98-b611-ed99a8e11b2c" />


### Task 2: Image Segmentation via Color Clustering
**Objective:** Partition images using pixel color grouping  
**Discoveries:**  
- Optimal clusters (k=4-6) balance detail and simplicity
- LAB color space outperforms RGB for perceptual segmentation
- Segmentation creates effective "region masks" for object isolation

  <img width="1389" height="794" alt="image" src="https://github.com/user-attachments/assets/97befa98-fd15-4292-8fb7-bd0a121d87df" />


### Task 3: High-Dimensional Data Visualization
**Objective:** Visualize feature spaces using PCA/t-SNE  
**Findings:**  
- PCA preserves global structure but loses local relationships
- t-SNE reveals nonlinear patterns (ideal for cluster separation)
- Perplexity tuning dramatically impacts t-SNE layout quality

  <img width="1195" height="498" alt="image" src="https://github.com/user-attachments/assets/6603a810-c16e-4ee4-9644-929c3d29093a" />


### Task 4: Sensor Activity Clustering
**Objective:** Detect motion patterns from accelerometer/gyroscope data  
**Learnings:**  
- Feature engineering (magnitude, variance) improves cluster quality
- PCA preprocessing (95% variance) enhances both K-Means/DBSCAN
- Inertial sensors provide rich clustering signals for AR navigation

  <img width="1194" height="496" alt="image" src="https://github.com/user-attachments/assets/87e4acfd-3f6d-4831-92d7-3b77b5c71ac4" />


### Task 5: AR Object Overlay Simulation
**Objective:** Simulate AR bounding box detection  
**Breakthroughs:**  
- K-Means segmentation enables contour-based object detection
- Morphological operations clean segmentation artifacts
- OpenCV pipeline achieves 15 FPS - viable for real-time AR

  <img width="1495" height="474" alt="image" src="https://github.com/user-attachments/assets/e9460c3b-f0f9-4982-84c0-74ab1e1abcac" />


### Task 6: Production Segmentation Workflow
**Objective:** Implement industry-grade segmentation pipeline  
**Adaptations:**  
- Integrated [python_for_microscopists](https://github.com/bnsreenu/python_for_microscopists) repo
- Handled large datasets via on-demand resizing
- Established modular preprocessing pipeline  
**Dataset Note:** Large files excluded due to GitHub's 50MB limit - resize inputs locally

  <img width="637" height="472" alt="image" src="https://github.com/user-attachments/assets/c6b2c51e-709c-46c7-a6f0-24a1d80871ba" />

  <img width="633" height="479" alt="image" src="https://github.com/user-attachments/assets/2968a204-0e5d-42ea-882b-cf9c3f687826" />

### Key Learnings

1. **Task 1**: Understood how KMeans clusters circular data well, while DBSCAN excels with arbitrary shapes and noise resilience.

2. **Task 2**: Learned to segment images by grouping similar pixel colors using KMeans, offering insight into unsupervised visual understanding.

3. **Task 3**: Observed how dimensionality reduction reveals structure in high-dimensional data; PCA preserves variance, t-SNE reveals local groupings.

4. **Task 4**: Gained experience in preprocessing sensor data and applying clustering to detect patterns without labels—critical for AR motion analysis.

5. **Task 5**: Integrated clustering with visualization by overlaying detected regions on images, simulating AR-like object grouping.

6. **Task 6**: Explored a real-world GitHub repository, ran KMeans-based segmentation locally, and understood typical project structure and limitations (e.g., file size constraints).


---


