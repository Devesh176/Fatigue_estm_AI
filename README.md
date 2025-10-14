## Fatigue Estimation of CFRP
`Note: Make sure to have "/data" repository, on the same level as /scripts, /models repositories.`

---
### Experiment Overview
- This dataset was generated from a collaborative research project between the Stanford Structures and Composites Laboratory (SACL) and the Prognostics Center of Excellence (PCoE) at NASA Ames Research Center. The core of the experiment involved subjecting Carbon Fiber Reinforced Polymer (CFRP) composite coupons to tension-tension fatigue tests.
- Specimen Details: The coupons(standardized test specimen made from composite material — specifically Carbon Fiber Reinforced Polymer (CFRP)) were made from Torayca T700G unidirectional carbon-prepreg material, shaped into a "dogbone" geometry to induce stress concentration. To analyze the impact of ply orientation, `three different symmetric layup` configurations were tested.
- Testing Protocol: The fatigue tests were performed on an MTS machine, adhering to ASTM Standards D3039 and D3479. These tests involved cyclical loadings at a frequency of 5.0 Hz with a stress ratio of R≈0.14

---
## About the Dataset:

### Experiment Purpose and Design
The dataset documents a series of fatigue aging tests on Carbon Fiber Reinforced Polymer (CFRP) composite materials. The research was a collaboration between the Stanford Structures and Composites Laboratory (SACL) and NASA's Prognostics Center of Excellence (PCoE). The primary goal was to monitor the progression of fatigue damage in the composites under controlled cyclical loading.

---

### Test Specimen (Coupon) Details
- **Material:** The test specimens, referred to as "coupons," were made from Torayca T700G unidirectional carbon-prepreg material.
- **Geometry:** Each coupon has a "dogbone" shape with dimensions of 15.24 cm x 25.4 cm and includes a notch to induce stress concentration.
- **Layup Configurations:** To study the effect of ply orientation, three different symmetric layup configurations were used:
  
|***Layup***|   ***Type***    |                                            ***Configuration Description***                                           |
|:---------:|:---------------:|:--------------------------------------------------------------------------------------------------------------------:|
|   Layup 1 |    [0₂ 90₄]     |2 layers at 0° and 4 layers at 90°. Prioritizes strength along the fiber direction (0°) and transverse stiffness (90°)|
|   Layup 2 |[0 90₂ 45 45 90] |            Mixed orientation: 0°, 90°, and ±45°. Offers balanced in-plane properties and shear resistance.           |
|   Layup 3 |  [90₂ 45 45]₂   |          Repeated sequence of 90°, 45°, 45°. Focuses on shear and transverse properties, less axial strength.        |
  
  - 0° fibers align with the loading direction → high tensile strength.
  - 90° fibers resist transverse loads → improve dimensional stability.
  - ±45° fibers enhance shear resistance → crucial for fatigue and torsion.
  - Testing Protocol: The coupons underwent tension-tension fatigue tests at a frequency of 5.0 Hz with a stress ratio of R ≈ 0.14. The tests followed ASTM Standards D3039 and D3479.
Note: L1_S17 stands for Layup 1 specimen 17th of Layup 1.
---

### Data Acquisition
Damage was monitored using Lamb wave propagation, with data collected at regular intervals.
- **Hardware:** Two SMART Layer® sets, each with six PZT (piezoelectric) sensors, were attached to each coupon, creating a setup of six actuators and six sensors.
- **Measurement Process:** The fatigue tests were paused every 50,000 cycles to collect sensor data. For each data collection instance:
  - All 36 actuator-sensor trajectories were actuated.
  -Each trajectory was interrogated at 7 different frequencies, ranging from 150-450 KHz.
  - This results in a total of 252 unique signal paths (36×7) for which both actuator and sensor signals are recorded.
---

### Boundary Conditions: 
Data for each coupon was collected under three different conditions:
- **Type 1:** Specimen loaded with the mean load.
- **Type 2:** Specimen unloaded but clamped.
- **Type 3:** Coupon removed from the testing machine (zero load).
- **Baseline Data:** Before testing, data was collected from each undamaged specimen to serve as a baseline reference for detecting damage propagation.

---

### Data Structure and Organization
- **File System:** All data for a specific coupon is stored in a dedicated folder named after the coupon.
- **LogBook:** Each folder contains a LogBook Excel Sheet that describes the data collection events, linking cycle counts, load, boundary conditions, and data file names.
- **Data Files:** The raw data for each measurement is stored in a Matlab struct array. This struct contains the Lamb wave signals for all 252 paths. A new, more intuitive data structure has been proposed to make the information more intelligible for users. This new structure is illustrated in Figure 3 of the report.

---

### Provided Tools (Matlab Scripts)
Several Matlab scripts are included to help users process and analyze the data. These scripts must be run from within the specific coupon's folder.
- `NEWFILEDEF.m:` Converts the data files from their original structure to the new, proposed format.
- `CHANGEPATH.m:` Corrects the internal path definition for specific coupons where the data acquisition convention differed.
- `DATA_MANAGEMENT.m:` A user interface to navigate the dataset, allowing users to import and plot data for a specific cycle or actuator-sensor pair.
- `PIECE1.m & PIECE2.m:` Example scripts that demonstrate how to import Lamb wave signals based on user-defined criteria like frequency, boundary condition, or for every cycle in the experiment


