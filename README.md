# BOM Mismatch Detection System

A deep learning-based system that automatically extracts Bill of Materials (BOM) data from engineering blueprints and compares it with a reference BOM to identify mismatches.

## Overview

Engineering blueprints often contain BOM tables with part numbers, descriptions, quantities, units of measurement, and material information. Manually comparing these tables with reference data is time-consuming and prone to errors.

This project automates BOM extraction, validation, correction of OCR errors, and mismatch detection using document analysis and image processing techniques.

## Features

* **Automated BOM Extraction:** Extracts text and tables from engineering blueprint images.
* **Table Structure Recognition:** Identifies table rows, columns, and headers.
* **Header Mapping:** Maps different header names to standard BOM fields.
* **Data Validation:** Validates extracted part numbers, descriptions, quantities, and units.
* **OCR Error Correction:** Helps normalize extracted text using similarity matching and reference data.
* **Mismatch Detection:** Identifies matching, missing, extra, and mismatched BOM items.
* **Duplicate Handling:** Preserves valid repeated part occurrences during comparison.
* **Result Visualization:** Displays comparison results through a React frontend.

## Technology Stack

| Component                 | Technologies              |
| ------------------------- | ------------------------- |
| Frontend                  | React, JavaScript, Axios  |
| Backend                   | Python, FastAPI           |
| OCR and Document Analysis | PaddleOCR, PP-StructureV3 |
| HTML Table Parsing        | BeautifulSoup             |
| Data Processing           | Pandas, NumPy             |

## System Workflow

1. Upload the engineering blueprint and reference BOM.
2. The React frontend sends the files to the FastAPI backend.
3. PaddleOCR and PP-StructureV3 extract text and identify table structures.
4. BeautifulSoup parses the extracted HTML tables.
5. The system maps headers and validates the extracted BOM records.
6. OCR correction and normalization help reduce extraction errors.
7. Extracted BOM records are compared with the reference BOM.
8. The frontend displays the detected mismatches and comparison results.

## Standard BOM Fields

* `PART_NO` – Part identification number
* `DESCRIPTION` – Description of the component
* `QTY` – Required quantity
* `UOM` – Unit of measurement
* `MATERIAL` – Material or grade of the component

**Project Structure**
BOM-Mismatch-Detection/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── data/
│   └── ground_truth/
│       └── blueprint_ground_truth.csv
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── README.md
├── model/
│   └── ground_truth/
│       └── blueprint_ground_truth.csv
├── output/
├── .gitignore
├── README.md
├── my_requirements.txt
├── package.json
└── package-lock.json

## Installation and Setup

### Prerequisites

* Python 3.11
* Node.js and npm
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/manjirikulkarni-2006/BOM-Mismatch-Detection.git
cd BOM-Mismatch-Detection
```

### 2. Set Up the Backend

Open a terminal in the project root:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the backend server:

```bash
uvicorn main:app --reload
```

Backend API: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 3. Set Up the Frontend

Open a second terminal in the project root:

```bash
cd frontend
npm install
npm start
```

Open the local frontend URL shown in the terminal. If your project is configured for port 3001, visit http://localhost:3001.

## Usage

1. Start the backend and frontend servers.
2. Open the application in your browser.
3. Upload the engineering blueprint and reference BOM.
4. Run the comparison.
5. Review the extracted BOM data and detected mismatches.

## Expected Output

The system helps identify:

* Matching BOM records
* Field-level differences
* Missing items
* Extra items
* Potential OCR extraction errors

Actual results depend on blueprint quality, table layout, OCR accuracy, and reference data.

## Dataset

The project uses engineering blueprint images and corresponding ground-truth BOM records for development and evaluation. Ensure that any dataset files are available locally and that their paths are configured correctly.

## Future Improvements

* Improve OCR accuracy on low-quality blueprint images.
* Enhance table extraction for complex engineering layouts.
* Improve automatic correction of ambiguous part numbers and quantities.
* Expand evaluation across different blueprint formats.

