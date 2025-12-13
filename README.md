# DNA Variant Detection App

A comprehensive web application that uses machine learning to detect pathogenic genetic variants and predict disease risks from DNA sequencing data.

## Overview

This project analyzes genomic data files (VCF, FASTQ, FASTA, BAM) to identify disease-causing genetic mutations across 7 core genetic conditions using advanced machine learning models.

## Features

- **File Upload**: Drag-and-drop support for genomic data files (up to 10GB)
- **ML Analysis**: Dual machine learning pipeline using Random Forest and Gradient Boosting classifiers
- **Real-time Processing**: Progress tracking during analysis
- **Comprehensive Results**: Detailed variant reports with confidence scores, pathogenicity ratings, and clinical recommendations
- **PDF Export**: Generate clinical-grade reports for medical records
- **7 Core Diseases Detected**:
  - Hereditary Breast and Ovarian Cancer (BRCA1/BRCA2)
  - Li-Fraumeni Syndrome (TP53)
  - Cystic Fibrosis (CFTR)
  - Huntington's Disease (HTT)
  - Marfan Syndrome (FBN1)
  - Alzheimer's Disease (APOE)
  - Hypertrophic Cardiomyopathy (MYBPC3, MYH7)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GreatBid/GenomeAI
cd GenomeAI
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run Development Server

```bash
npm run dev
```

### 4. Open in Browser

```bash
http://localhost:3000
```

## Usage

1. **Upload File**: Drag and drop or click to select a genomic data file (VCF, FASTQ, FASTA, BAM format)
2. **Wait for Analysis**: The ML model processes your file (typically 10-30 seconds)
3. **View Results**: Browse through three result tabs:
   - **Variants**: Individual variant details with genes, positions, and risk levels
   - **Summary**: High-level overview of risk distribution
   - **Recommendations**: Clinical guidance and next steps
4. **Export PDF**: Download a comprehensive report for medical records

