# SkyPulse: Fly Safe, Land Secure..!!

## AIRBUS AEROTHON 6.0
### Team SkyPulse

## Table of Contents
- [Problem Statement](#problem-statement)
- [Hypothesis](#hypothesis)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
  - [Image Analysis and Damage Assessment](#image-analysis-and-damage-assessment)
  - [Faulty Wire Identification](#faulty-wire-location-identification)
  - [Repair Recommendations](#repair-recommendations)
  - [Web Application (Full-Stack)](#web-application-full-stack)
- [Contributors](#contributors)

## Problem Statement

Aircraft maintenance and repair are integral components of the aviation industry, serving as the backbone of safety, reliability, and operational continuity. The meticulous assessment and repair of dents, damage, detection of faulty wires, and wear on aircraft fuselage, wings, and other components are paramount to ensuring flight safety, regulatory compliance, and public confidence.

First and foremost, the safety of passengers, crew, and cargo is the primary concern in aviation. Any compromise to the structural integrity of an aircraft, no matter how minor, poses a potential threat to safety. Damage, such as dents or structural deformities, can disrupt airflow, compromise aerodynamics, and weaken critical structural elements, increasing the risk of catastrophic failures during flight. Similarly, faulty wiring poses a significant safety risk, as it can lead to electrical malfunctions within critical aircraft systems. These malfunctions may result in system failures, in-flight emergencies, or even fires, jeopardizing the safety of passengers, crew, and the aircraft itself.

Thus, thorough assessments and repairs are essential to maintaining the airworthiness of aircraft and safeguarding against potential accidents.

## Hypothesis

By implementing an automated system that utilizes image recognition, machine learning algorithms, and a database of known aircraft damage, we can accurately detect and classify cracks, dents, and faulty wiring on aircraft components. This system can then predict the most appropriate repair methods and materials needed to address the identified damage, leading to faster turnaround times, reduced maintenance costs, and improved aircraft safety.

## Dataset

**Roboflow Universe:**
This dataset contains a collection of images of aircraft with labels for common damage types like cracks, dents, and faulty wiring. It's a good starting point for training a basic damage detection system but may require additional data for comprehensive performance.

## Tech Stack

### Image Analysis and Damage Assessment
- **OpenCV:** This open-source computer vision library provides foundational functionalities for image processing, manipulation, and feature extraction. It's used for tasks like image pre-processing, noise reduction, and basic image analysis.
- **YOLOv5 (You Only Look Once v5):** This modern object detection algorithm is used for identifying and localizing damage (cracks, dents, etc.) and faulty wires within aircraft images. YOLOv5's pre-trained model is leveraged for damage detection and fine-tuned on our specific dataset.

### Faulty Wire Identification
- **Random Forest Classifier:** This machine learning algorithm is used to classify individual wires within aircraft harnesses as faulty or non-faulty based on extracted image features. Random Forest is a robust choice for classification tasks due to its interpretability and ability to handle imbalanced datasets.

### Repair Recommendations
- **Machine Learning Algorithms:** The specific algorithms used here depend on the complexity of generating repair recommendations. It involves supervised learning techniques trained on historical data linking specific damage types and locations to recommended repair methods and materials.

### Web Application (Full-Stack)
- **Streamlit:** This Python library is the foundation of our web application backend. With Streamlit, we can define functions for:
  - **Image Upload:** Components for users to upload images of aircraft components for analysis.
  - **Machine Learning Model Integration:** Integration of our trained machine learning models (YOLOv5, Random Forest) to process the uploaded images and generate predictions.
  - **Data Processing:** Manipulation and analysis of prediction results (damage type, location, severity) for repair recommendations.
  - **User Interface Creation:** Various UI components like text boxes, buttons, and charts to create a user-friendly interface where users can interact with the system and see the results.

## Contributors

| SL.NO | NAME               | GITHUB PROFILE                               | LINKEDIN PROFILE                                      | EMAIL                          |
|-------|--------------------|-----------------------------------------------|-------------------------------------------------------|--------------------------------|
| 1.    | Soumik Saha        | [soumik-saha](https://github.com/soumik-saha) | [LinkedIn](https://www.linkedin.com/in/soumikisonline)    | sahasoumik1573@gmail.com             |
| 2.    | Bhagyasri Uddandam | [BhagyasriUddandam](https://github.com/BhagyasriUddandam) | [LinkedIn](https://www.linkedin.com/in/bhagyasri-u) | bhagyasrirama@gmail.com |
| 3.    | Ayushi Gupta       | [1001ayushi](https://github.com/1001ayushi)     | [LinkedIn](https://www.linkedin.com/in/ayushi-gupta-72a38121b/) | ayushigupta.9901@gmail.com             |
| 4.    | Pankaj Goel        | [whitewolf-debugger](https://github.com/whitewolf-debugger) | [LinkedIn](https://www.linkedin.com/in/pankaj-goel-30195720a/) | goelpankaj875@gmail.com             |
| 5.    | Souvik Dey         | [souvik987](https://github.com/souvik987) | [LinkedIn](https://www.linkedin.com/in/souvik-dey-80b033241/) | souvik001dey@gmail.com             |
