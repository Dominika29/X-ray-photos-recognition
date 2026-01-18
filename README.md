


  <h3 align="center">x-ray photo recognition</h3>

  <p align="center">
	  Dominika Czerwińska 159557
	</p>
	  <p align="center">
   Convolutional Neural Network
   </p>
    <p align="center">
   <b>course:Poznań University of Technology</b>
	</p>


## Table of contents

- [About](#about)
- [State of art](#state-of-art)
- [Description of the solution](#description-of-the-solution)
- [Proof of concept](#proof-of-concept)


## About
The objective of this project is the automated classification of bone fractures in X-ray images of upper and lower extremities. This is achieved through the execution of a CNN (Convolutional Neural Network) - a deep learning architecture specifically engineered to extract spatial hierarchies from medical imaging data.

- ### Input 
		Radiographic images (X-rays) of limbs, including arms, and legs.
  All data is standardized to greyscale with a uniform resolution (e.g., 128x128).
  The model categorizes images into two distinct labels:
  fractured, normal
  
- ### Intended results
		The system aims to provide an automated diagnostic suggestion based on user-provided X-rays.
  E.g., when a scan of a fractured radius is processed,
  the system should accurately assign the 'fractured' label.
  
- ### Motivation 
		Leveraging neural networks for bone fracture detection represents a significant 
  advancement in digital healthcare. This technology streamlines clinical 
  workflows, assists radiologists in high-pressure environments, and 
  minimizes the diagnostic "oversight" rate in emergency medicine.
  
## State of art
| **Machine Learning Methodology** | **Overview** | **Advantages** | **Disadvantages** |
|:---:|:---:|:---:|:---:|
| **CNN** (Convolutional Neural Networks) | Specialized layers automatically identify spatial hierarchies and patterns (like bone cracks) directly from raw pixels. | - Superior accuracy in medical imaging <br>- Automated feature discovery <br>- High efficiency via weight sharing | - High demand for GPU resources <br>- Performance is dependent on image quality |
| **DNN** (Deep Neural Networks) | Comprises multiple dense, fully connected layers that map complex non-linear relationships within data. | - Versatile for diverse data types <br>- Capable of building highly complex models | - Ignores spatial pixel relationships <br>- Extremely data-intensive <br>- Longer training duration |
| **Traditional Feature Extraction** | Manually defined descriptors (e.g., bone density, edge sharpness) are fed into a shallow classifier. | - Low computational overhead <br>- High transparency and interpretability <br>- Works well with limited datasets | - Accuracy is capped by human feature selection <br>- Fails to capture subtle visual nuances |

## Description of the solution
As highlighted in the comparison, convolutional neural networks are the industry standard for computer vision. The core mechanism involves applying various kernels (filters) to the input data to identify diagnostic markers, such as disruptions in bone cortex or alignment.

The network's deployment consists of the following stages:

Convolutional Layers: These act as feature detectors, scanning the X-ray for specific edges, shadows, and fracture patterns.

Max-Pooling Layers: These perform down-sampling to reduce data dimensionality while preserving the most critical diagnostic features.

Dense Layers: These fully connected layers aggregate the identified features to understand the overall context of the image.

Softmax Output: This final stage calculates the probability distribution across the two classes (Fractured vs. Normal).

System Output: The algorithm produces a probability vector. For instance, a result of [0.915, 0.085] translates to:

91.5% probability that the image is fractured.

8.5% probability that the image is normal.

## Proof of concept
