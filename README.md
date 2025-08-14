# Medical Research Data Mining and Pattern Discovery CLI

A comprehensive healthcare analytics command-line interface that processes anonymized patient datasets and clinical studies to discover patterns, perform statistical analysis, and generate evidence-based recommendations.

## Project Overview

This application implements advanced medical research data mining capabilities including:

- **Disease Pattern Recognition**: Statistical algorithms for identifying disease patterns and correlations
- **Survival Analysis**: Kaplan-Meier estimations and treatment effectiveness comparisons
- **Drug Interaction Analysis**: Adverse event correlation and risk assessment
- **Genetic Variant Analysis**: Population frequency calculations and clinical significance assessment
- **Clinical Research Reports**: Evidence-based treatment recommendations and comprehensive analytics

## Features

### Core Functionality

- **Patient Data Management**: Load and validate anonymized patient datasets
- **Statistical Analysis**: Comprehensive survival and pattern analysis
- **Drug Safety Assessment**: Interaction risk evaluation and adverse event analysis
- **Genetic Research**: Variant frequency analysis and clinical significance assessment
- **Report Generation**: Automated clinical research reports with recommendations

### Technical Capabilities

- **Data Validation**: Robust input validation and error handling
- **Modular Design**: Object-oriented architecture with inheritance and polymorphism
- **File Operations**: CSV and JSON data processing with error recovery
- **Statistical Computing**: Advanced statistical algorithms using standard library
- **Comprehensive Logging**: Detailed logging for debugging and audit trails

## Requirements Specification

### System Requirements

- Python 3.8 or higher
- Standard library modules only (no external dependencies)
- Minimum 4GB RAM for large datasets
- 1GB disk space for data and reports

### Functional Requirements

1. **Data Processing**

   - Load patient data from CSV files
   - Load drug interactions from JSON files
   - Validate data integrity and format
   - Handle missing or corrupted data gracefully

2. **Statistical Analysis**

   - Calculate survival statistics (mean, median, standard deviation)
   - Perform Kaplan-Meier survival analysis
   - Analyze disease patterns and correlations
   - Calculate biomarker statistics and correlations

3. **Drug Safety Analysis**

   - Identify high-risk drug interactions
   - Analyze adverse event patterns
   - Calculate interaction risk scores
   - Generate safety recommendations

4. **Genetic Analysis**

   - Identify rare genetic variants
   - Calculate population frequencies
   - Assess clinical significance
   - Group variants by gene

5. **Report Generation**
   - Generate survival analysis reports
   - Create disease pattern reports
   - Produce treatment recommendations
   - Export results in multiple formats

### Non-Functional Requirements

- **Performance**: Process datasets with 10,000+ patients in under 30 seconds
- **Reliability**: 99.9% uptime with comprehensive error handling
- **Security**: Anonymized data processing with no PII exposure
- **Usability**: Intuitive CLI interface with clear documentation

## Installation and Setup

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python3 --version

# Create virtual environment (recommended)
python3 -m venv medical_research_env
source medical_research_env/bin/activate  # On Windows: medical_research_env\Scripts\activate
```

### Installation

```bash
# Clone or download the project files
# No additional dependencies required - uses only Python standard library

# Make the main script executable
chmod +x medical_research_cli.py

# Run tests to verify installation
python3 test_medical_research.py
```

## User Guide

### Quick Start

1. **Run the Application**

   ```bash
   # Interactive mode (recommended for beginners)
   python3 medical_research_cli.py --interactive

   # Demo mode (runs all analyses with sample data)
   python3 medical_research_cli.py --demo
   ```

2. **Interactive Menu Options**
   - **Option 1**: Load Sample Data
   - **Option 2**: Run Survival Analysis
   - **Option 3**: Run Disease Pattern Analysis
   - **Option 4**: Run Drug Interaction Analysis
   - **Option 5**: Run Genetic Variant Analysis
   - **Option 6**: Generate Comprehensive Report
   - **Option 7**: Run All Analyses
   - **Option 8**: Exit

### Data Format Requirements

#### Patient Data (CSV Format)

```csv
patient_id,age,gender,diagnosis,treatment,survival_days,genetic_variants,adverse_events,biomarkers
P001,65,M,Lung Cancer,Chemotherapy,450,EGFR_T790M;ALK_fusion,Nausea;Fatigue,CEA:15.2;CA125:45.8
P002,58,F,Breast Cancer,Targeted Therapy,720,BRCA1_mutation;HER2_amplification,Rash;Diarrhea,HER2:3.2;ER:0.8
```

#### Drug Interactions (JSON Format)

```json
[
  {
    "drug1": "Warfarin",
    "drug2": "Aspirin",
    "interaction_type": "Anticoagulation",
    "severity": "High",
    "risk_score": 0.85,
    "evidence_level": "Strong",
    "adverse_events": ["Bleeding", "Bruising"]
  }
]
```

### Command Line Options

```bash
# Interactive mode
python3 medical_research_cli.py -i

# Demo mode with sample data
python3 medical_research_cli.py -d

# Help
python3 medical_research_cli.py --help
```

### Output Files

The application generates reports in the `reports/` directory:

- `survival_analysis.txt`: Detailed survival statistics and treatment comparisons
- `disease_patterns.txt`: Disease distribution and demographic analysis
- `treatment_recommendations.txt`: Evidence-based treatment recommendations
- `medical_research.log`: Application log file with detailed execution information

## Technical Documentation

### Architecture Overview

The application follows a modular, object-oriented design with clear separation of concerns:

```
MedicalResearchCLI (Main Controller)
├── DataLoader (Data Input/Validation)
├── StatisticalAnalyzer (Statistical Computations)
├── DrugInteractionAnalyzer (Drug Safety Analysis)
├── GeneticAnalyzer (Genetic Research)
└── ReportGenerator (Output Generation)
```

### Core Classes

#### Patient

Represents individual patient data with validation:

- Patient demographics and clinical information
- Survival data and treatment history
- Genetic variants and adverse events
- Biomarker measurements

#### DrugInteraction

Models drug-drug interactions with risk assessment:

- Interaction type and severity classification
- Risk scoring and evidence levels
- Associated adverse events

#### GeneticVariant

Represents genetic variants with population data:

- Variant identification and genomic coordinates
- Population frequency calculations
- Clinical significance assessment

### Statistical Algorithms

#### Survival Analysis

- **Kaplan-Meier Estimation**: Non-parametric survival curve estimation
- **Treatment Comparison**: Statistical comparison of treatment effectiveness
- **Risk Stratification**: Patient grouping by survival risk factors

#### Pattern Recognition

- **Disease Distribution**: Frequency analysis of diagnoses
- **Demographic Analysis**: Age and gender correlations
- **Biomarker Correlation**: Statistical relationships between biomarkers

#### Drug Safety Analysis

- **Risk Scoring**: Quantitative assessment of interaction risks
- **Adverse Event Analysis**: Pattern recognition in side effects
- **Evidence Assessment**: Quality evaluation of interaction data

### Error Handling

The application implements comprehensive error handling:

1. **Data Validation**: Input validation with detailed error messages
2. **File Operations**: Graceful handling of missing or corrupted files
3. **Statistical Computations**: Protection against division by zero and invalid data
4. **Memory Management**: Efficient processing of large datasets
5. **Logging**: Detailed logging for debugging and audit trails

### Performance Optimization

- **Efficient Data Structures**: Use of dictionaries and sets for fast lookups
- **Lazy Loading**: Data loaded only when needed
- **Memory Management**: Efficient handling of large datasets
- **Algorithm Optimization**: Optimized statistical computations

## Testing

### Running Tests

```bash
# Run all unit tests
python3 test_medical_research.py

# Run with verbose output
python3 -m unittest test_medical_research -v
```

### Test Coverage

The test suite covers:

- **Data Classes**: Patient, DrugInteraction, GeneticVariant validation
- **Data Validation**: Input validation and error handling
- **Statistical Analysis**: Survival analysis and pattern recognition
- **Drug Analysis**: Interaction analysis and risk assessment
- **Genetic Analysis**: Variant analysis and frequency calculations
- **Report Generation**: File output and content validation
- **CLI Functionality**: Main application workflow

### Test Data

Sample data is included for testing:

- 5 sample patients with various diagnoses and treatments
- 3 drug interactions with different risk levels
- 2 genetic variants with population frequency data

## Standard Library Modules Used

The application demonstrates proficiency with 5+ Python standard library modules:

1. **sys**: System-specific parameters and functions
2. **os**: Operating system interface
3. **json**: JSON encoder and decoder
4. **csv**: CSV file reading and writing
5. **statistics**: Mathematical statistics functions
6. **math**: Mathematical functions
7. **datetime**: Basic date and time types
8. **argparse**: Command line argument parsing
9. **logging**: Logging facility
10. **typing**: Support for type hints
11. **dataclasses**: Data classes
12. **abc**: Abstract base classes
13. **random**: Random number generation
14. **collections**: Specialized container datatypes
15. **tempfile**: Generate temporary files and directories
16. **unittest**: Unit testing framework
17. **shutil**: High-level file operations

## Troubleshooting

### Common Issues

1. **File Not Found Errors**

   - Ensure data files are in the correct directory
   - Check file permissions
   - Verify file format (CSV/JSON)

2. **Memory Errors**

   - Reduce dataset size for large files
   - Close other applications to free memory
   - Use data sampling for initial testing

3. **Statistical Calculation Errors**
   - Check for missing or invalid data
   - Ensure sufficient sample sizes
   - Verify data format and types

### Debug Mode

Enable detailed logging by modifying the logging level in the main script:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Guidelines

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for changes
5. Use type hints for function parameters

### Code Structure

- Main application: `medical_research_cli.py`
- Unit tests: `test_medical_research.py`
- Documentation: `README.md`
- Sample data: `data/` directory
- Generated reports: `reports/` directory

## License

This project is developed for educational purposes as part of a Python programming course. All code and documentation are provided as-is for learning and demonstration.

## Contact

For questions or issues related to this educational project, please refer to the course instructor or teaching assistant.

---

**Note**: This application is designed for educational purposes and should not be used for actual medical decision-making without proper clinical validation and regulatory approval.
