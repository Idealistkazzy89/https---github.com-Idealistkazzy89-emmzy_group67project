#!/usr/bin/env python3
"""
Medical Research Data Mining and Pattern Discovery CLI
=====================================================

A comprehensive healthcare analytics command-line interface that processes
anonymized patient datasets and clinical studies to discover patterns,
perform statistical analysis, and generate evidence-based recommendations.

Features:
- Disease pattern recognition using statistical algorithms
- Survival analysis with Kaplan-Meier estimations
- Drug interaction analysis with adverse event correlation
- Genetic variant analysis with population frequency calculations
- Clinical research report generation
- Comprehensive data validation and error handling

Author: Medical Research Team
Version: 1.0.0
Python Version: 3.8+
"""

import sys
import os
import json
import csv
import statistics
import math
import datetime
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import random
from collections import defaultdict, Counter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_research.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Patient:
    """Represents a patient in the medical research dataset."""
    patient_id: str
    age: int
    gender: str
    diagnosis: str
    treatment: str
    survival_days: int
    genetic_variants: List[str]
    adverse_events: List[str]
    biomarkers: Dict[str, float]
    
    def __post_init__(self):
        """Validate patient data after initialization."""
        if not self.patient_id or len(self.patient_id.strip()) == 0:
            raise ValueError("Patient ID cannot be empty")
        if self.age < 0 or self.age > 150:
            raise ValueError("Age must be between 0 and 150")
        if self.gender not in ['M', 'F', 'Other']:
            raise ValueError("Gender must be M, F, or Other")
        if self.survival_days < 0:
            raise ValueError("Survival days cannot be negative")


@dataclass
class DrugInteraction:
    """Represents a drug interaction with risk assessment."""
    drug1: str
    drug2: str
    interaction_type: str
    severity: str
    risk_score: float
    evidence_level: str
    adverse_events: List[str]
    
    def __post_init__(self):
        """Validate drug interaction data."""
        if self.risk_score < 0 or self.risk_score > 1:
            raise ValueError("Risk score must be between 0 and 1")
        if self.severity not in ['Low', 'Moderate', 'High', 'Severe']:
            raise ValueError("Invalid severity level")


@dataclass
class GeneticVariant:
    """Represents a genetic variant with population data."""
    variant_id: str
    gene: str
    chromosome: str
    position: int
    reference_allele: str
    alternate_allele: str
    population_frequency: float
    clinical_significance: str
    associated_diseases: List[str]
    
    def __post_init__(self):
        """Validate genetic variant data."""
        if self.population_frequency < 0 or self.population_frequency > 1:
            raise ValueError("Population frequency must be between 0 and 1")


class DataValidator:
    """Validates medical research data for consistency and integrity."""
    
    @staticmethod
    def validate_patient_data(data: Dict[str, Any]) -> bool:
        """Validate patient data structure and content."""
        required_fields = ['patient_id', 'age', 'gender', 'diagnosis', 'treatment', 
                          'survival_days', 'genetic_variants', 'adverse_events', 'biomarkers']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        try:
            int(data['age'])
            int(data['survival_days'])
        except (ValueError, TypeError):
            logger.error("Age and survival_days must be integers")
            return False
        
        return True
    
    @staticmethod
    def validate_drug_interaction(data: Dict[str, Any]) -> bool:
        """Validate drug interaction data."""
        required_fields = ['drug1', 'drug2', 'interaction_type', 'severity', 
                          'risk_score', 'evidence_level', 'adverse_events']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field in drug interaction: {field}")
                return False
        
        try:
            float(data['risk_score'])
        except (ValueError, TypeError):
            logger.error("Risk score must be a number")
            return False
        
        return True


class DataLoader:
    """Handles loading and parsing of medical research data files."""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        self.validator = DataValidator()
    
    def load_patients_from_csv(self, filename: str) -> List[Patient]:
        """Load patient data from CSV file."""
        filepath = os.path.join(self.data_directory, filename)
        patients = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if self.validator.validate_patient_data(row):
                        try:
                            # Parse genetic variants and adverse events
                            genetic_variants = row['genetic_variants'].split(';') if row['genetic_variants'] else []
                            adverse_events = row['adverse_events'].split(';') if row['adverse_events'] else []
                            
                            # Parse biomarkers
                            biomarkers = {}
                            if row['biomarkers']:
                                for item in row['biomarkers'].split(';'):
                                    if ':' in item:
                                        key, value = item.split(':')
                                        biomarkers[key.strip()] = float(value.strip())
                            
                            patient = Patient(
                                patient_id=row['patient_id'],
                                age=int(row['age']),
                                gender=row['gender'],
                                diagnosis=row['diagnosis'],
                                treatment=row['treatment'],
                                survival_days=int(row['survival_days']),
                                genetic_variants=genetic_variants,
                                adverse_events=adverse_events,
                                biomarkers=biomarkers
                            )
                            patients.append(patient)
                        except Exception as e:
                            logger.warning(f"Error parsing patient {row.get('patient_id', 'unknown')}: {e}")
                            continue
                            
        except FileNotFoundError:
            logger.error(f"Patient data file not found: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return []
        
        logger.info(f"Successfully loaded {len(patients)} patients from {filename}")
        return patients
    
    def load_drug_interactions_from_json(self, filename: str) -> List[DrugInteraction]:
        """Load drug interactions from JSON file."""
        filepath = os.path.join(self.data_directory, filename)
        interactions = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                for item in data:
                    if self.validator.validate_drug_interaction(item):
                        try:
                            interaction = DrugInteraction(
                                drug1=item['drug1'],
                                drug2=item['drug2'],
                                interaction_type=item['interaction_type'],
                                severity=item['severity'],
                                risk_score=float(item['risk_score']),
                                evidence_level=item['evidence_level'],
                                adverse_events=item['adverse_events']
                            )
                            interactions.append(interaction)
                        except Exception as e:
                            logger.warning(f"Error parsing drug interaction: {e}")
                            continue
                            
        except FileNotFoundError:
            logger.error(f"Drug interactions file not found: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error loading drug interactions: {e}")
            return []
        
        logger.info(f"Successfully loaded {len(interactions)} drug interactions from {filename}")
        return interactions


class StatisticalAnalyzer:
    """Performs statistical analysis on medical research data."""
    
    @staticmethod
    def calculate_survival_analysis(patients: List[Patient]) -> Dict[str, Any]:
        """Perform Kaplan-Meier survival analysis."""
        if not patients:
            return {"error": "No patient data available"}
        
        # Group patients by treatment
        treatment_groups = defaultdict(list)
        for patient in patients:
            treatment_groups[patient.treatment].append(patient.survival_days)
        
        results = {
            "overall_survival": {
                "mean": statistics.mean([p.survival_days for p in patients]),
                "median": statistics.median([p.survival_days for p in patients]),
                "std": statistics.stdev([p.survival_days for p in patients]) if len(patients) > 1 else 0
            },
            "treatment_comparison": {}
        }
        
        # Calculate survival statistics for each treatment group
        for treatment, survival_times in treatment_groups.items():
            if len(survival_times) > 0:
                results["treatment_comparison"][treatment] = {
                    "count": len(survival_times),
                    "mean_survival": statistics.mean(survival_times),
                    "median_survival": statistics.median(survival_times),
                    "std_survival": statistics.stdev(survival_times) if len(survival_times) > 1 else 0
                }
        
        return results
    
    @staticmethod
    def perform_disease_pattern_analysis(patients: List[Patient]) -> Dict[str, Any]:
        """Analyze disease patterns and correlations."""
        if not patients:
            return {"error": "No patient data available"}
        
        # Analyze diagnosis patterns
        diagnosis_counts = Counter([p.diagnosis for p in patients])
        age_by_diagnosis = defaultdict(list)
        gender_by_diagnosis = defaultdict(lambda: {"M": 0, "F": 0, "Other": 0})
        
        for patient in patients:
            age_by_diagnosis[patient.diagnosis].append(patient.age)
            gender_by_diagnosis[patient.diagnosis][patient.gender] += 1
        
        results = {
            "diagnosis_distribution": dict(diagnosis_counts),
            "age_analysis": {},
            "gender_analysis": dict(gender_by_diagnosis)
        }
        
        # Calculate age statistics for each diagnosis
        for diagnosis, ages in age_by_diagnosis.items():
            results["age_analysis"][diagnosis] = {
                "mean_age": statistics.mean(ages),
                "median_age": statistics.median(ages),
                "age_range": (min(ages), max(ages))
            }
        
        return results
    
    @staticmethod
    def analyze_biomarkers(patients: List[Patient]) -> Dict[str, Any]:
        """Analyze biomarker patterns and correlations."""
        if not patients:
            return {"error": "No patient data available"}
        
        # Collect all biomarker names
        all_biomarkers = set()
        for patient in patients:
            all_biomarkers.update(patient.biomarkers.keys())
        
        results = {
            "biomarker_statistics": {},
            "biomarker_correlations": {}
        }
        
        # Calculate statistics for each biomarker
        for biomarker in all_biomarkers:
            values = [p.biomarkers.get(biomarker) for p in patients if biomarker in p.biomarkers]
            if values:
                results["biomarker_statistics"][biomarker] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
        
        return results


class DrugInteractionAnalyzer:
    """Analyzes drug interactions and adverse events."""
    
    def __init__(self, interactions: List[DrugInteraction]):
        self.interactions = interactions
    
    def find_high_risk_interactions(self, threshold: float = 0.7) -> List[DrugInteraction]:
        """Find high-risk drug interactions above the threshold."""
        return [interaction for interaction in self.interactions 
                if interaction.risk_score >= threshold]
    
    def analyze_adverse_events(self) -> Dict[str, Any]:
        """Analyze adverse events across all drug interactions."""
        if not self.interactions:
            return {"error": "No drug interaction data available"}
        
        # Count adverse events
        event_counts = Counter()
        severity_by_event = defaultdict(list)
        
        for interaction in self.interactions:
            for event in interaction.adverse_events:
                event_counts[event] += 1
                severity_by_event[event].append(interaction.severity)
        
        results = {
            "most_common_events": event_counts.most_common(10),
            "event_severity_analysis": {}
        }
        
        # Analyze severity patterns for each event
        for event, severities in severity_by_event.items():
            severity_counts = Counter(severities)
            results["event_severity_analysis"][event] = dict(severity_counts)
        
        return results
    
    def find_drug_combinations(self, drug_name: str) -> List[DrugInteraction]:
        """Find all interactions involving a specific drug."""
        return [interaction for interaction in self.interactions 
                if drug_name.lower() in [interaction.drug1.lower(), interaction.drug2.lower()]]


class GeneticAnalyzer:
    """Analyzes genetic variants and population frequencies."""
    
    def __init__(self, variants: List[GeneticVariant]):
        self.variants = variants
    
    def find_rare_variants(self, frequency_threshold: float = 0.01) -> List[GeneticVariant]:
        """Find rare genetic variants below the frequency threshold."""
        return [variant for variant in self.variants 
                if variant.population_frequency <= frequency_threshold]
    
    def analyze_by_gene(self) -> Dict[str, List[GeneticVariant]]:
        """Group variants by gene."""
        gene_variants = defaultdict(list)
        for variant in self.variants:
            gene_variants[variant.gene].append(variant)
        return dict(gene_variants)
    
    def find_clinically_significant_variants(self) -> List[GeneticVariant]:
        """Find variants with clinical significance."""
        significant_levels = ['Pathogenic', 'Likely pathogenic', 'Uncertain significance']
        return [variant for variant in self.variants 
                if variant.clinical_significance in significant_levels]


class ReportGenerator:
    """Generates comprehensive clinical research reports."""
    
    def __init__(self, output_directory: str = "reports"):
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
    
    def generate_survival_report(self, survival_data: Dict[str, Any], filename: str = "survival_analysis.txt") -> str:
        """Generate a detailed survival analysis report."""
        filepath = os.path.join(self.output_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("SURVIVAL ANALYSIS REPORT\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if "error" in survival_data:
                file.write(f"ERROR: {survival_data['error']}\n")
                return filepath
            
            # Overall survival statistics
            overall = survival_data["overall_survival"]
            file.write("OVERALL SURVIVAL STATISTICS:\n")
            file.write("-" * 30 + "\n")
            file.write(f"Mean survival: {overall['mean']:.2f} days\n")
            file.write(f"Median survival: {overall['median']:.2f} days\n")
            file.write(f"Standard deviation: {overall['std']:.2f} days\n\n")
            
            # Treatment comparison
            file.write("TREATMENT COMPARISON:\n")
            file.write("-" * 20 + "\n")
            for treatment, stats in survival_data["treatment_comparison"].items():
                file.write(f"\nTreatment: {treatment}\n")
                file.write(f"  Patient count: {stats['count']}\n")
                file.write(f"  Mean survival: {stats['mean_survival']:.2f} days\n")
                file.write(f"  Median survival: {stats['median_survival']:.2f} days\n")
                file.write(f"  Standard deviation: {stats['std_survival']:.2f} days\n")
        
        logger.info(f"Survival report generated: {filepath}")
        return filepath
    
    def generate_pattern_report(self, pattern_data: Dict[str, Any], filename: str = "disease_patterns.txt") -> str:
        """Generate a disease pattern analysis report."""
        filepath = os.path.join(self.output_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("DISEASE PATTERN ANALYSIS REPORT\n")
            file.write("=" * 40 + "\n\n")
            file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if "error" in pattern_data:
                file.write(f"ERROR: {pattern_data['error']}\n")
                return filepath
            
            # Diagnosis distribution
            file.write("DIAGNOSIS DISTRIBUTION:\n")
            file.write("-" * 25 + "\n")
            for diagnosis, count in pattern_data["diagnosis_distribution"].items():
                file.write(f"{diagnosis}: {count} patients\n")
            file.write("\n")
            
            # Age analysis
            file.write("AGE ANALYSIS BY DIAGNOSIS:\n")
            file.write("-" * 30 + "\n")
            for diagnosis, age_stats in pattern_data["age_analysis"].items():
                file.write(f"\n{diagnosis}:\n")
                file.write(f"  Mean age: {age_stats['mean_age']:.1f} years\n")
                file.write(f"  Median age: {age_stats['median_age']:.1f} years\n")
                file.write(f"  Age range: {age_stats['age_range'][0]}-{age_stats['age_range'][1]} years\n")
            
            # Gender analysis
            file.write("\nGENDER DISTRIBUTION BY DIAGNOSIS:\n")
            file.write("-" * 35 + "\n")
            for diagnosis, gender_counts in pattern_data["gender_analysis"].items():
                file.write(f"\n{diagnosis}:\n")
                for gender, count in gender_counts.items():
                    if count > 0:
                        file.write(f"  {gender}: {count} patients\n")
        
        logger.info(f"Pattern report generated: {filepath}")
        return filepath
    
    def generate_treatment_recommendations(self, patients: List[Patient], 
                                         interactions: List[DrugInteraction],
                                         filename: str = "treatment_recommendations.txt") -> str:
        """Generate evidence-based treatment recommendations."""
        filepath = os.path.join(self.output_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("EVIDENCE-BASED TREATMENT RECOMMENDATIONS\n")
            file.write("=" * 45 + "\n\n")
            file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Analyze treatment effectiveness
            treatment_effectiveness = defaultdict(list)
            for patient in patients:
                treatment_effectiveness[patient.treatment].append(patient.survival_days)
            
            file.write("TREATMENT EFFECTIVENESS ANALYSIS:\n")
            file.write("-" * 35 + "\n")
            for treatment, survival_times in treatment_effectiveness.items():
                if len(survival_times) >= 5:  # Only recommend if sufficient data
                    mean_survival = statistics.mean(survival_times)
                    file.write(f"\n{treatment}:\n")
                    file.write(f"  Average survival: {mean_survival:.1f} days\n")
                    file.write(f"  Patient count: {len(survival_times)}\n")
                    
                    if mean_survival > 365:  # More than 1 year
                        file.write(f"  RECOMMENDATION: Consider {treatment} for long-term survival\n")
                    elif mean_survival > 180:  # More than 6 months
                        file.write(f"  RECOMMENDATION: {treatment} shows moderate effectiveness\n")
                    else:
                        file.write(f"  RECOMMENDATION: Limited effectiveness, consider alternatives\n")
            
            # Drug interaction warnings
            high_risk_interactions = [i for i in interactions if i.risk_score >= 0.7]
            if high_risk_interactions:
                file.write("\n\nHIGH-RISK DRUG INTERACTIONS:\n")
                file.write("-" * 30 + "\n")
                for interaction in high_risk_interactions[:10]:  # Top 10
                    file.write(f"\n{interaction.drug1} + {interaction.drug2}:\n")
                    file.write(f"  Risk Score: {interaction.risk_score:.3f}\n")
                    file.write(f"  Severity: {interaction.severity}\n")
                    file.write(f"  Type: {interaction.interaction_type}\n")
                    file.write(f"  WARNING: Avoid combination or monitor closely\n")
        
        logger.info(f"Treatment recommendations generated: {filepath}")
        return filepath


class MedicalResearchCLI:
    """Main CLI application for medical research data mining."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()
        self.patients = []
        self.drug_interactions = []
        self.genetic_variants = []
    
    def load_sample_data(self):
        """Load sample data for demonstration purposes."""
        logger.info("Loading sample medical research data...")
        
        # Create sample patient data
        sample_patients = [
            Patient(
                patient_id="P001", age=65, gender="M", diagnosis="Lung Cancer",
                treatment="Chemotherapy", survival_days=450, 
                genetic_variants=["EGFR_T790M", "ALK_fusion"],
                adverse_events=["Nausea", "Fatigue"],
                biomarkers={"CEA": 15.2, "CA125": 45.8}
            ),
            Patient(
                patient_id="P002", age=58, gender="F", diagnosis="Breast Cancer",
                treatment="Targeted Therapy", survival_days=720,
                genetic_variants=["BRCA1_mutation", "HER2_amplification"],
                adverse_events=["Rash", "Diarrhea"],
                biomarkers={"HER2": 3.2, "ER": 0.8}
            ),
            Patient(
                patient_id="P003", age=72, gender="M", diagnosis="Prostate Cancer",
                treatment="Hormone Therapy", survival_days=365,
                genetic_variants=["AR_mutation"],
                adverse_events=["Hot Flashes"],
                biomarkers={"PSA": 8.5}
            ),
            Patient(
                patient_id="P004", age=45, gender="F", diagnosis="Ovarian Cancer",
                treatment="Surgery + Chemotherapy", survival_days=180,
                genetic_variants=["BRCA2_mutation"],
                adverse_events=["Nausea", "Hair Loss"],
                biomarkers={"CA125": 120.5, "HE4": 95.2}
            ),
            Patient(
                patient_id="P005", age=61, gender="M", diagnosis="Colon Cancer",
                treatment="Immunotherapy", survival_days=540,
                genetic_variants=["MSI_high", "BRAF_mutation"],
                adverse_events=["Fatigue", "Rash"],
                biomarkers={"CEA": 25.8}
            )
        ]
        
        # Create sample drug interactions
        sample_interactions = [
            DrugInteraction(
                drug1="Warfarin", drug2="Aspirin",
                interaction_type="Anticoagulation", severity="High",
                risk_score=0.85, evidence_level="Strong",
                adverse_events=["Bleeding", "Bruising"]
            ),
            DrugInteraction(
                drug1="Simvastatin", drug2="Amiodarone",
                interaction_type="Metabolism", severity="Moderate",
                risk_score=0.65, evidence_level="Moderate",
                adverse_events=["Muscle Pain", "Liver Damage"]
            ),
            DrugInteraction(
                drug1="Digoxin", drug2="Furosemide",
                interaction_type="Electrolyte", severity="Moderate",
                risk_score=0.55, evidence_level="Strong",
                adverse_events=["Arrhythmia", "Electrolyte Imbalance"]
            )
        ]
        
        # Create sample genetic variants
        sample_variants = [
            GeneticVariant(
                variant_id="rs1042522", gene="TP53", chromosome="17",
                position=7577120, reference_allele="G", alternate_allele="C",
                population_frequency=0.25, clinical_significance="Likely pathogenic",
                associated_diseases=["Lung Cancer", "Breast Cancer"]
            ),
            GeneticVariant(
                variant_id="rs1801133", gene="MTHFR", chromosome="1",
                position=11856378, reference_allele="C", alternate_allele="T",
                population_frequency=0.35, clinical_significance="Uncertain significance",
                associated_diseases=["Cardiovascular Disease"]
            )
        ]
        
        self.patients = sample_patients
        self.drug_interactions = sample_interactions
        self.genetic_variants = sample_variants
        
        logger.info(f"Loaded {len(self.patients)} patients, {len(self.drug_interactions)} drug interactions, and {len(self.genetic_variants)} genetic variants")
    
    def run_survival_analysis(self):
        """Run survival analysis and generate report."""
        print("\n=== SURVIVAL ANALYSIS ===")
        survival_data = self.statistical_analyzer.calculate_survival_analysis(self.patients)
        
        if "error" not in survival_data:
            print(f"Overall mean survival: {survival_data['overall_survival']['mean']:.1f} days")
            print(f"Overall median survival: {survival_data['overall_survival']['median']:.1f} days")
            
            print("\nTreatment Comparison:")
            for treatment, stats in survival_data["treatment_comparison"].items():
                print(f"  {treatment}: {stats['mean_survival']:.1f} days (n={stats['count']})")
        
        # Generate report
        report_path = self.report_generator.generate_survival_report(survival_data)
        print(f"\nSurvival analysis report saved to: {report_path}")
    
    def run_pattern_analysis(self):
        """Run disease pattern analysis and generate report."""
        print("\n=== DISEASE PATTERN ANALYSIS ===")
        pattern_data = self.statistical_analyzer.perform_disease_pattern_analysis(self.patients)
        
        if "error" not in pattern_data:
            print("Diagnosis Distribution:")
            for diagnosis, count in pattern_data["diagnosis_distribution"].items():
                print(f"  {diagnosis}: {count} patients")
            
            print("\nAge Analysis:")
            for diagnosis, age_stats in pattern_data["age_analysis"].items():
                print(f"  {diagnosis}: Mean age {age_stats['mean_age']:.1f} years")
        
        # Generate report
        report_path = self.report_generator.generate_pattern_report(pattern_data)
        print(f"\nPattern analysis report saved to: {report_path}")
    
    def run_drug_analysis(self):
        """Run drug interaction analysis."""
        print("\n=== DRUG INTERACTION ANALYSIS ===")
        analyzer = DrugInteractionAnalyzer(self.drug_interactions)
        
        # Find high-risk interactions
        high_risk = analyzer.find_high_risk_interactions(threshold=0.7)
        print(f"High-risk interactions (≥0.7): {len(high_risk)}")
        
        for interaction in high_risk:
            print(f"  {interaction.drug1} + {interaction.drug2}: Risk {interaction.risk_score:.3f}")
        
        # Analyze adverse events
        event_analysis = analyzer.analyze_adverse_events()
        if "error" not in event_analysis:
            print("\nMost Common Adverse Events:")
            for event, count in event_analysis["most_common_events"][:5]:
                print(f"  {event}: {count} occurrences")
    
    def run_genetic_analysis(self):
        """Run genetic variant analysis."""
        print("\n=== GENETIC VARIANT ANALYSIS ===")
        analyzer = GeneticAnalyzer(self.genetic_variants)
        
        # Find rare variants
        rare_variants = analyzer.find_rare_variants(frequency_threshold=0.01)
        print(f"Rare variants (≤1% frequency): {len(rare_variants)}")
        
        # Find clinically significant variants
        significant_variants = analyzer.find_clinically_significant_variants()
        print(f"Clinically significant variants: {len(significant_variants)}")
        
        for variant in significant_variants:
            print(f"  {variant.variant_id} ({variant.gene}): {variant.clinical_significance}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive research report."""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        # Generate treatment recommendations
        report_path = self.report_generator.generate_treatment_recommendations(
            self.patients, self.drug_interactions
        )
        print(f"Treatment recommendations saved to: {report_path}")
        
        # Generate biomarker analysis
        biomarker_data = self.statistical_analyzer.analyze_biomarkers(self.patients)
        if "error" not in biomarker_data:
            print(f"Analyzed {len(biomarker_data['biomarker_statistics'])} biomarkers")
    
    def interactive_menu(self):
        """Display interactive menu for user selection."""
        while True:
            print("\n" + "="*60)
            print("MEDICAL RESEARCH DATA MINING CLI")
            print("="*60)
            print("1. Load Sample Data")
            print("2. Run Survival Analysis")
            print("3. Run Disease Pattern Analysis")
            print("4. Run Drug Interaction Analysis")
            print("5. Run Genetic Variant Analysis")
            print("6. Generate Comprehensive Report")
            print("7. Run All Analyses")
            print("8. Exit")
            print("-"*60)
            
            try:
                choice = input("Enter your choice (1-8): ").strip()
                
                if choice == "1":
                    self.load_sample_data()
                elif choice == "2":
                    self.run_survival_analysis()
                elif choice == "3":
                    self.run_pattern_analysis()
                elif choice == "4":
                    self.run_drug_analysis()
                elif choice == "5":
                    self.run_genetic_analysis()
                elif choice == "6":
                    self.generate_comprehensive_report()
                elif choice == "7":
                    self.load_sample_data()
                    self.run_survival_analysis()
                    self.run_pattern_analysis()
                    self.run_drug_analysis()
                    self.run_genetic_analysis()
                    self.generate_comprehensive_report()
                elif choice == "8":
                    print("Thank you for using Medical Research Data Mining CLI!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 8.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in menu: {e}")
                print(f"An error occurred: {e}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Medical Research Data Mining and Pattern Discovery CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python medical_research_cli.py --interactive
  python medical_research_cli.py --demo
        """
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demonstration with sample data"
    )
    
    args = parser.parse_args()
    
    try:
        cli = MedicalResearchCLI()
        
        if args.interactive:
            cli.interactive_menu()
        elif args.demo:
            print("Running Medical Research Data Mining Demo...")
            cli.load_sample_data()
            cli.run_survival_analysis()
            cli.run_pattern_analysis()
            cli.run_drug_analysis()
            cli.run_genetic_analysis()
            cli.generate_comprehensive_report()
            print("\nDemo completed! Check the 'reports' directory for generated files.")
        else:
            # Default to interactive mode
            cli.interactive_menu()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 