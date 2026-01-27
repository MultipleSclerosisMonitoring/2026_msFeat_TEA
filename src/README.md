## Directory Layout
* **`gait_analysis/`**: The primary package containing the logic for:
    * **Data Extraction**: Managed by the `GaitExtractor` class (InfluxDB to HDF5).
    * **Signal Processing**: Managed by the `GaitDataProcessor` class.
    * **Validation**: Strict schema enforcement using Pydantic v2.

## Core Modules & Logic

### 1. `processor.py` (Biomechanical Engine)
This is the heart of the analytical pipeline. It implements:
- **Auto-Calibration**: An algorithm that deduces the vertical axis of the sensor by analyzing gravitational components (detecting if the vertical is Ax, Ay, or Az).
- **DSP Pipeline**: 4th-order Zero-phase Butterworth filtering for noise reduction in pressure and inertial signals.
- **Heuristics**: Turn detection logic based on gyroscope thresholds and automated Heel Strike (step) identification.

### 2. `extractor.py` (Data Management)
Handles the high-performance communication with InfluxDB and the structural management of HDF5 storage.

## Modular Architecture
The code is designed to be imported as a library by the executable tools in the `scripts/` folder.

**Key Design Principles:**
- **OOP Compliance**: Logic is encapsulated within classes (`GaitDataProcessor`, `GaitExtractor`) to ensure state independence.
- **Hardware Agnostic**: The system is designed to handle sensors in any physical orientation thanks to the auto-calibration module.
- **Strict Typing**: Uses Python 3.12 type hinting and Pydantic for robust data integrity