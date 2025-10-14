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

---

## Decoding the data files

### PZT-data 
- The file nomenclature looks like:
```bash
L[Layup]S[Specimen]_[Cycle]_[Condition]_[Repeat].mat
```
where, for eg. `L1S11_0_1_2.mat` would break down as-

|    Segment    |                               Meaning                              |
|:-------------:|:------------------------------------------------------------------:|
| L1(layup)     | Layup type — here, Layup 1: [0₂ 90₄]                               |
| S11(Specimen) | Specimen number — the 11th coupon tested with Layup 1              |
| 0(Cycle)      | Cycle number — fatigue cycle at which data was collected (e.g., 0) |
| 1(Condition)  | Boundary condition:                                                |
|               | - 0: Baseline (undamaged, no load)                                 |
|               | - 1: Loaded                                                        |
|               | - 2: Clamped (unloaded but fixed)                                  |
|               | - 3: Traction-free (removed from machine)                          |
| 2(Repeat)     | Repeat index — distinguishes multiple recordings under same setup  |

- **What is inside the .mat file?**: Each file contains a MATLAB struct with fields like-
  - `coupon.cycles`: Fatigue cycle number
  - `coupon.load`: Load applied (in Kips)
  - `coupon.comment`: Any comment (eg. 'Baseline' or '14 cracks observed (H), 1 crack observed (V), delamination on 6-7')
  - `coupon.condition`: Boundary condition label
  - `coupon.path_data`: Array of 252 paths (36 actuator-sensor pairs × 7 frequencies)
    - Each path_data(i) includes:
      - `actuator`, `sensor`, `amplitude`, `frequency`, `gain`
      - `signal_actuator`: waveform from actuator (2000x1 in shape for a single path out of 252)
      - `signal_sensor`: waveform from sensor (2000x1 in shape for a single path out of 252)
      - `sampling_rate`: 1200000
  - `coupon.straingage_data`: Optional strain data and stiffness degradation
  - `coupon.XRay_data`: Path to X-ray image file (if available)

---
### StrainData (Still need to understand more clearly, also about the regions A, M and S)
- The file nomenclature looks like
  ```bash
  L[Layup]_S[Specimen]_F[CycleIndex]_STRAIN_[Region]_DAT.mat
  ```
where, for eg. `L1_S11_F01_STRAIN_M_DAT.mat` - 

|  Segment |                             Meaning                            |
|:--------:|:--------------------------------------------------------------:|
| L1       | Layup type — e.g., Layup 1: [0₂ 90₄]                           |
| S11      | Specimen number — Coupon 11                                    |
| F01      | Fatigue cycle index — not the actual cycle count, but an index |
| STRAIN_A | Strain gauge location:                                         |
|          | - A: Actuator region                                           |
|          | - M: Middle region                                             |
|          | - S: Sensor region                                             |
| DAT.mat  | MATLAB data file containing strain measurements                |

and for eg. `L1_S11_S11_STRAIN_A_DAT.mat` - This format appears when the cycle index is replaced by the specimen ID again. It typically indicates:
    - Baseline strain data for the specimen Or a non-cycle-specific strain snapshot (e.g., initial calibration or reference)

|  Segment |                         Meaning                        |
|:--------:|:------------------------------------------------------:|
| L1       | Layup type                                             |
| S11_S11  | Specimen ID repeated — implies static or baseline data |
| STRAIN_A | Region: Actuator                                       |
| DAT.mat  | MATLAB strain data                                     |

- **What is inside the .mat file?**: Each strain file — whether it's STRAIN_A, STRAIN_M, STRAIN_S, or the general DAT file — contains four separate arrays:
  - *Contents*: Variable Name	Description
    - `strain1`	Strain signal from gauge 1 in the region
    - `strain2`	Strain signal from gauge 2
    - `strain3`	Strain signal from gauge 3
    - `strain4`	Strain signal from gauge 4
  - All are of type double
  - Each array may have different lengths depending on sampling duration or gauge configuration
  - No struct wrapping — these are flat variables in the .mat workspace
  - The region (A, M, S) is encoded in the filename

  - The cycle index or baseline is also encoded in the filename

  - The strain1–strain4 arrays represent parallel strain readings from that region, possibly from different physical gauges or channels

  - Example: L1_S11_F01_STRAIN_M_DAT.mat → Middle region strain data for Layup 1, Specimen 11, Cycle Index 01 → Contains strain1, strain2, strain3, strain4 — all from the middle region
- **How these files help us?**
  - Plot each strainX to inspect signal quality, drift, or anomalies
  - Compare across cycles to track stiffness degradation
  - Align with PZT signals and X-ray images for holistic damage assessmen

---

### XRay
- The file nomenclature goes as-
  ```bash
  L[Layup]S[Specimen]_[Cycle].jpg
  ```
where, for eg  `L3S20_1_2.jpg`-

| Segment |                                Meaning                               |
|:-------:|:--------------------------------------------------------------------:|
| L3      | Layup type — e.g., Layup 3: [90₂ 45 45]₂                             |
| S20     | Specimen number — Coupon 20                                          |
| 1       | Cycle index or fatigue stage — often corresponds to a specific cycle |
| .jpg    | Image format — X-ray scan of the coupon                              |

or for eg. `L3S20_baseline.jpg`

|  Segment |                                Meaning                               |
|:--------:|:--------------------------------------------------------------------:|
| baseline | Undamaged state — no load, no fatigue                                |
| .jpg     | Initial X-ray image                                                  |

- **How these files can help?**
  - Each coupon folder contains these X-ray .jpg files.
  - These files are referenced in the MATLAB struct under coupon.XRay_data.file_location (PZT-data)
  - Used to visually confirm damage progression alongside:
    - Lamb wave signal changes
    - Strain gauge degradation
    - Load and cycle history  

