# OverBody Damage Detection

## Problem Statement

Vehicle damage assessment is traditionally a manual and time-consuming process, often requiring trained personnel to visually inspect and record damages on a vehicle's surface. This approach is prone to human error, inconsistency, and delays, especially in large-scale scenarios such as insurance claims, rental returns, or used car evaluations. The lack of automation limits scalability and introduces subjectivity in severity judgments.

The proposed project aims to automate this process using deep learning and computer vision. By leveraging YOLO-based object detection and OpenCV image processing, the system detects multiple types of visible damages (e.g., dents, cracks, scratches) on a vehicle from a single image and estimates their physical size and severity. This enables faster, more consistent, and objective vehicle condition reporting, reducing reliance on manual inspection while improving efficiency and accuracy in damage evaluation workflows.

---

## Scope

Overbody Damage Detection is an AI-powered car damage detection and severity assessment system designed to automate the visual inspection process of vehicles. By leveraging deep learning models and computer vision techniques, Overbody Damage Detection enables precise, real-time detection and evaluation of multiple types of exterior car damage from a single image. This solution is particularly valuable for industries like automotive insurance, rental services, used vehicle inspections, and fleet maintenance. Overbody Damage Detection not only identifies damage types but also quantifies their severity using a tailored strategy for each damage category, reducing the need for manual inspection and enabling objective, fast, and scalable assessments.

### Proposed Solution

- **Multi-Class Damage Detection**: Utilizes a YOLO-based object detection model to identify and localize multiple types of surface-level car damage including dents, cracks, scratches, rust, glass shatter, and broken lamps.

- **Damage-Specific Severity Estimation**:
  - For cracks and scratches, severity is computed by estimating physical length and area using custom models and pixel-to-cm calibration.
  - For dents, depth is estimated using ML-Depth Pro, a depth prediction model trained on LiDAR-enhanced datasets to estimate the 3D structure of the dent region.
  - For rust, a custom-trained ResNet-based image classifier assesses severity levels (mild, moderate, severe) based on texture and spread.

- **Real-World Length Calibration**: Supports pixel-to-centimeter conversion using trained models for reference objects (like a ruler), enabling realistic measurement of damages when physical scaling is available.

- **Visual Inspection Dashboard**: Provides a side-by-side visualization of original and annotated images with bounding boxes, damage labels, and confidence scores, aiding quick review and validation.

- **AI-Powered Repair Guidance & Part Replacement Suggestions**: Integrated AI agents analyze detected damages to generate a brief repair guide and provide replacement part links (via AI-driven recommendations and web scraping). This helps users directly access repair instructions and purchase required parts online, reducing turnaround time.

- **Modular Architecture**: Designed with modular functions in Python, allowing easy integration with new detection models, damage types, or real-time video inputs.

- **Scalable & Integration-Ready**: Can be integrated into insurance claim apps, vehicle inspection systems, or car rental platforms with minimal infrastructure changes.

---

## Project Architecture

<!-- Insert Project Architecture Diagram here -->

![Project Architecture Diagram](overbody-architecture.png)

---

## Business Impact

- **Faster Damage Assessment**: Automates the detection and severity evaluation process, reducing manual inspection time from several minutes per vehicle to a few seconds, thereby accelerating claim processing, return evaluations, and inspection workflows.

- **Operational Cost Reduction**: Minimizes the need for human inspectors and manual measurements, lowering labor costs and allowing skilled personnel to focus on higher-value activities such as customer service or claim resolution.

- **Objective & Consistent Evaluation**: Standardizes damage assessment across vehicles and users, eliminating subjectivity and inconsistency often present in manual evaluations, and ensuring fairer insurance claims and rental returns.

- **Enhanced Customer Experience**: Enables quick, transparent, and data-backed assessments for customers during vehicle returns or accident reporting, improving trust and satisfaction.

- **Improved Insurance and Rental Efficiency**: Helps insurers and rental agencies reduce fraudulent claims, streamline documentation, and optimize damage-related payouts by providing image-backed severity scores.

- **Seamless Integration with Inspection Systems**: Easily integrates with existing inspection pipelines, insurance platforms, or mobile claim apps, adding AI-powered automation without major infrastructure changes.

- **Data-Driven Insights**: Aggregated damage data and severity trends can be used to identify common failure patterns, improve vehicle design feedback loops, or optimize fleet maintenance strategies.

- **Audit-Ready Documentation**: Class-wise cropped images, annotated visuals, and severity reports create a detailed digital trail of inspection, aiding compliance, dispute resolution, and long-term record-keeping.
