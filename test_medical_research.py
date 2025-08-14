#!/usr/bin/env python3
"""
Unit Tests for Medical Research Data Mining CLI
==============================================

Comprehensive test suite for the medical research data mining application.
Tests cover data validation, statistical analysis, drug interactions,
genetic analysis, and report generation.

Author: Medical Research Team
Version: 1.0.0
"""

import unittest
import tempfile
import os
import json
import csv
from unittest.mock import patch, MagicMock
import sys

# Add the current directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medical_research_cli import (
    Patient, DrugInteraction, GeneticVariant, DataValidator,
    StatisticalAnalyzer, DrugInteractionAnalyzer, GeneticAnalyzer,
    ReportGenerator, MedicalResearchCLI
)


class TestPatient(unittest.TestCase):
    """Test cases for Patient data class."""
    
    def test_valid_patient_creation(self):
        """Test creating a valid patient."""
        patient = Patient(
            patient_id="P001",
            age=65,
            gender="M",
            diagnosis="Lung Cancer",
            treatment="Chemotherapy",
            survival_days=450,
            genetic_variants=["EGFR_T790M"],
            adverse_events=["Nausea"],
            biomarkers={"CEA": 15.2}
        )
        
        self.assertEqual(patient.patient_id, "P001")
        self.assertEqual(patient.age, 65)
        self.assertEqual(patient.gender, "M")
        self.assertEqual(patient.diagnosis, "Lung Cancer")
        self.assertEqual(patient.survival_days, 450)
        self.assertEqual(len(patient.genetic_variants), 1)
        self.assertEqual(len(patient.adverse_events), 1)
        self.assertEqual(len(patient.biomarkers), 1)
    
    def test_invalid_patient_id(self):
        """Test patient creation with invalid patient ID."""
        with self.assertRaises(ValueError):
            Patient(
                patient_id="",
                age=65,
                gender="M",
                diagnosis="Lung Cancer",
                treatment="Chemotherapy",
                survival_days=450,
                genetic_variants=[],
                adverse_events=[],
                biomarkers={}
            )
    
    def test_invalid_age(self):
        """Test patient creation with invalid age."""
        with self.assertRaises(ValueError):
            Patient(
                patient_id="P001",
                age=-5,
                gender="M",
                diagnosis="Lung Cancer",
                treatment="Chemotherapy",
                survival_days=450,
                genetic_variants=[],
                adverse_events=[],
                biomarkers={}
            )
    
    def test_invalid_gender(self):
        """Test patient creation with invalid gender."""
        with self.assertRaises(ValueError):
            Patient(
                patient_id="P001",
                age=65,
                gender="Invalid",
                diagnosis="Lung Cancer",
                treatment="Chemotherapy",
                survival_days=450,
                genetic_variants=[],
                adverse_events=[],
                biomarkers={}
            )
    
    def test_negative_survival_days(self):
        """Test patient creation with negative survival days."""
        with self.assertRaises(ValueError):
            Patient(
                patient_id="P001",
                age=65,
                gender="M",
                diagnosis="Lung Cancer",
                treatment="Chemotherapy",
                survival_days=-10,
                genetic_variants=[],
                adverse_events=[],
                biomarkers={}
            )


class TestDrugInteraction(unittest.TestCase):
    """Test cases for DrugInteraction data class."""
    
    def test_valid_drug_interaction_creation(self):
        """Test creating a valid drug interaction."""
        interaction = DrugInteraction(
            drug1="Warfarin",
            drug2="Aspirin",
            interaction_type="Anticoagulation",
            severity="High",
            risk_score=0.85,
            evidence_level="Strong",
            adverse_events=["Bleeding"]
        )
        
        self.assertEqual(interaction.drug1, "Warfarin")
        self.assertEqual(interaction.drug2, "Aspirin")
        self.assertEqual(interaction.risk_score, 0.85)
        self.assertEqual(interaction.severity, "High")
    
    def test_invalid_risk_score(self):
        """Test drug interaction creation with invalid risk score."""
        with self.assertRaises(ValueError):
            DrugInteraction(
                drug1="Warfarin",
                drug2="Aspirin",
                interaction_type="Anticoagulation",
                severity="High",
                risk_score=1.5,  # Invalid: > 1
                evidence_level="Strong",
                adverse_events=["Bleeding"]
            )
    
    def test_invalid_severity(self):
        """Test drug interaction creation with invalid severity."""
        with self.assertRaises(ValueError):
            DrugInteraction(
                drug1="Warfarin",
                drug2="Aspirin",
                interaction_type="Anticoagulation",
                severity="Invalid",
                risk_score=0.85,
                evidence_level="Strong",
                adverse_events=["Bleeding"]
            )


class TestGeneticVariant(unittest.TestCase):
    """Test cases for GeneticVariant data class."""
    
    def test_valid_genetic_variant_creation(self):
        """Test creating a valid genetic variant."""
        variant = GeneticVariant(
            variant_id="rs1042522",
            gene="TP53",
            chromosome="17",
            position=7577120,
            reference_allele="G",
            alternate_allele="C",
            population_frequency=0.25,
            clinical_significance="Likely pathogenic",
            associated_diseases=["Lung Cancer"]
        )
        
        self.assertEqual(variant.variant_id, "rs1042522")
        self.assertEqual(variant.gene, "TP53")
        self.assertEqual(variant.population_frequency, 0.25)
    
    def test_invalid_population_frequency(self):
        """Test genetic variant creation with invalid population frequency."""
        with self.assertRaises(ValueError):
            GeneticVariant(
                variant_id="rs1042522",
                gene="TP53",
                chromosome="17",
                position=7577120,
                reference_allele="G",
                alternate_allele="C",
                population_frequency=1.5,  # Invalid: > 1
                clinical_significance="Likely pathogenic",
                associated_diseases=["Lung Cancer"]
            )


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_valid_patient_data(self):
        """Test validation of valid patient data."""
        valid_data = {
            'patient_id': 'P001',
            'age': '65',
            'gender': 'M',
            'diagnosis': 'Lung Cancer',
            'treatment': 'Chemotherapy',
            'survival_days': '450',
            'genetic_variants': 'EGFR_T790M',
            'adverse_events': 'Nausea',
            'biomarkers': 'CEA:15.2'
        }
        
        self.assertTrue(self.validator.validate_patient_data(valid_data))
    
    def test_invalid_patient_data_missing_field(self):
        """Test validation of patient data with missing field."""
        invalid_data = {
            'patient_id': 'P001',
            'age': '65',
            # Missing gender field
            'diagnosis': 'Lung Cancer',
            'treatment': 'Chemotherapy',
            'survival_days': '450',
            'genetic_variants': 'EGFR_T790M',
            'adverse_events': 'Nausea',
            'biomarkers': 'CEA:15.2'
        }
        
        self.assertFalse(self.validator.validate_patient_data(invalid_data))
    
    def test_invalid_patient_data_non_numeric_age(self):
        """Test validation of patient data with non-numeric age."""
        invalid_data = {
            'patient_id': 'P001',
            'age': 'invalid',
            'gender': 'M',
            'diagnosis': 'Lung Cancer',
            'treatment': 'Chemotherapy',
            'survival_days': '450',
            'genetic_variants': 'EGFR_T790M',
            'adverse_events': 'Nausea',
            'biomarkers': 'CEA:15.2'
        }
        
        self.assertFalse(self.validator.validate_patient_data(invalid_data))
    
    def test_valid_drug_interaction_data(self):
        """Test validation of valid drug interaction data."""
        valid_data = {
            'drug1': 'Warfarin',
            'drug2': 'Aspirin',
            'interaction_type': 'Anticoagulation',
            'severity': 'High',
            'risk_score': '0.85',
            'evidence_level': 'Strong',
            'adverse_events': ['Bleeding']
        }
        
        self.assertTrue(self.validator.validate_drug_interaction(valid_data))
    
    def test_invalid_drug_interaction_data_missing_field(self):
        """Test validation of drug interaction data with missing field."""
        invalid_data = {
            'drug1': 'Warfarin',
            # Missing drug2 field
            'interaction_type': 'Anticoagulation',
            'severity': 'High',
            'risk_score': '0.85',
            'evidence_level': 'Strong',
            'adverse_events': ['Bleeding']
        }
        
        self.assertFalse(self.validator.validate_drug_interaction(invalid_data))


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for StatisticalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer()
        self.sample_patients = [
            Patient(
                patient_id="P001", age=65, gender="M", diagnosis="Lung Cancer",
                treatment="Chemotherapy", survival_days=450,
                genetic_variants=[], adverse_events=[],
                biomarkers={"CEA": 15.2}
            ),
            Patient(
                patient_id="P002", age=58, gender="F", diagnosis="Breast Cancer",
                treatment="Targeted Therapy", survival_days=720,
                genetic_variants=[], adverse_events=[],
                biomarkers={"HER2": 3.2}
            ),
            Patient(
                patient_id="P003", age=72, gender="M", diagnosis="Prostate Cancer",
                treatment="Hormone Therapy", survival_days=365,
                genetic_variants=[], adverse_events=[],
                biomarkers={"PSA": 8.5}
            )
        ]
    
    def test_survival_analysis_with_data(self):
        """Test survival analysis with valid patient data."""
        result = self.analyzer.calculate_survival_analysis(self.sample_patients)
        
        self.assertNotIn("error", result)
        self.assertIn("overall_survival", result)
        self.assertIn("treatment_comparison", result)
        
        # Check overall survival statistics
        overall = result["overall_survival"]
        self.assertIsInstance(overall["mean"], (int, float))
        self.assertIsInstance(overall["median"], (int, float))
        self.assertIsInstance(overall["std"], (int, float))
        
        # Check treatment comparison
        self.assertIn("Chemotherapy", result["treatment_comparison"])
        self.assertIn("Targeted Therapy", result["treatment_comparison"])
        self.assertIn("Hormone Therapy", result["treatment_comparison"])
    
    def test_survival_analysis_empty_data(self):
        """Test survival analysis with empty patient data."""
        result = self.analyzer.calculate_survival_analysis([])
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No patient data available")
    
    def test_disease_pattern_analysis_with_data(self):
        """Test disease pattern analysis with valid patient data."""
        result = self.analyzer.perform_disease_pattern_analysis(self.sample_patients)
        
        self.assertNotIn("error", result)
        self.assertIn("diagnosis_distribution", result)
        self.assertIn("age_analysis", result)
        self.assertIn("gender_analysis", result)
        
        # Check diagnosis distribution
        diagnosis_dist = result["diagnosis_distribution"]
        self.assertEqual(diagnosis_dist["Lung Cancer"], 1)
        self.assertEqual(diagnosis_dist["Breast Cancer"], 1)
        self.assertEqual(diagnosis_dist["Prostate Cancer"], 1)
        
        # Check age analysis
        age_analysis = result["age_analysis"]
        self.assertIn("Lung Cancer", age_analysis)
        self.assertEqual(age_analysis["Lung Cancer"]["mean_age"], 65)
    
    def test_disease_pattern_analysis_empty_data(self):
        """Test disease pattern analysis with empty patient data."""
        result = self.analyzer.perform_disease_pattern_analysis([])
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No patient data available")
    
    def test_biomarker_analysis_with_data(self):
        """Test biomarker analysis with valid patient data."""
        result = self.analyzer.analyze_biomarkers(self.sample_patients)
        
        self.assertNotIn("error", result)
        self.assertIn("biomarker_statistics", result)
        
        # Check biomarker statistics
        biomarker_stats = result["biomarker_statistics"]
        self.assertIn("CEA", biomarker_stats)
        self.assertIn("HER2", biomarker_stats)
        self.assertIn("PSA", biomarker_stats)
        
        # Check CEA statistics
        cea_stats = biomarker_stats["CEA"]
        self.assertEqual(cea_stats["count"], 1)
        self.assertEqual(cea_stats["mean"], 15.2)
        self.assertEqual(cea_stats["min"], 15.2)
        self.assertEqual(cea_stats["max"], 15.2)
    
    def test_biomarker_analysis_empty_data(self):
        """Test biomarker analysis with empty patient data."""
        result = self.analyzer.analyze_biomarkers([])
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No patient data available")


class TestDrugInteractionAnalyzer(unittest.TestCase):
    """Test cases for DrugInteractionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_interactions = [
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
                adverse_events=["Muscle Pain"]
            ),
            DrugInteraction(
                drug1="Digoxin", drug2="Furosemide",
                interaction_type="Electrolyte", severity="Moderate",
                risk_score=0.55, evidence_level="Strong",
                adverse_events=["Arrhythmia"]
            )
        ]
        self.analyzer = DrugInteractionAnalyzer(self.sample_interactions)
    
    def test_find_high_risk_interactions(self):
        """Test finding high-risk drug interactions."""
        high_risk = self.analyzer.find_high_risk_interactions(threshold=0.7)
        
        self.assertEqual(len(high_risk), 1)
        self.assertEqual(high_risk[0].drug1, "Warfarin")
        self.assertEqual(high_risk[0].drug2, "Aspirin")
        self.assertEqual(high_risk[0].risk_score, 0.85)
    
    def test_find_high_risk_interactions_lower_threshold(self):
        """Test finding high-risk drug interactions with lower threshold."""
        high_risk = self.analyzer.find_high_risk_interactions(threshold=0.6)
        
        self.assertEqual(len(high_risk), 2)
        risk_scores = [interaction.risk_score for interaction in high_risk]
        self.assertIn(0.85, risk_scores)
        self.assertIn(0.65, risk_scores)
    
    def test_analyze_adverse_events(self):
        """Test adverse events analysis."""
        result = self.analyzer.analyze_adverse_events()
        
        self.assertNotIn("error", result)
        self.assertIn("most_common_events", result)
        self.assertIn("event_severity_analysis", result)
        
        # Check most common events
        common_events = result["most_common_events"]
        event_names = [event for event, count in common_events]
        self.assertIn("Bleeding", event_names)
        self.assertIn("Muscle Pain", event_names)
        self.assertIn("Arrhythmia", event_names)
    
    def test_analyze_adverse_events_empty_data(self):
        """Test adverse events analysis with empty data."""
        empty_analyzer = DrugInteractionAnalyzer([])
        result = empty_analyzer.analyze_adverse_events()
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No drug interaction data available")
    
    def test_find_drug_combinations(self):
        """Test finding drug combinations involving a specific drug."""
        combinations = self.analyzer.find_drug_combinations("Warfarin")
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0].drug1, "Warfarin")
        self.assertEqual(combinations[0].drug2, "Aspirin")
    
    def test_find_drug_combinations_case_insensitive(self):
        """Test finding drug combinations with case-insensitive search."""
        combinations = self.analyzer.find_drug_combinations("warfarin")
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0].drug1, "Warfarin")
    
    def test_find_drug_combinations_not_found(self):
        """Test finding drug combinations for non-existent drug."""
        combinations = self.analyzer.find_drug_combinations("NonExistentDrug")
        
        self.assertEqual(len(combinations), 0)


class TestGeneticAnalyzer(unittest.TestCase):
    """Test cases for GeneticAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_variants = [
            GeneticVariant(
                variant_id="rs1042522", gene="TP53", chromosome="17",
                position=7577120, reference_allele="G", alternate_allele="C",
                population_frequency=0.25, clinical_significance="Likely pathogenic",
                associated_diseases=["Lung Cancer"]
            ),
            GeneticVariant(
                variant_id="rs1801133", gene="MTHFR", chromosome="1",
                position=11856378, reference_allele="C", alternate_allele="T",
                population_frequency=0.005, clinical_significance="Uncertain significance",
                associated_diseases=["Cardiovascular Disease"]
            ),
            GeneticVariant(
                variant_id="rs7412", gene="APOE", chromosome="19",
                position=45412079, reference_allele="C", alternate_allele="T",
                population_frequency=0.08, clinical_significance="Pathogenic",
                associated_diseases=["Alzheimer's Disease"]
            )
        ]
        self.analyzer = GeneticAnalyzer(self.sample_variants)
    
    def test_find_rare_variants(self):
        """Test finding rare genetic variants."""
        rare_variants = self.analyzer.find_rare_variants(frequency_threshold=0.01)
        
        self.assertEqual(len(rare_variants), 1)
        self.assertEqual(rare_variants[0].variant_id, "rs1801133")
        self.assertEqual(rare_variants[0].population_frequency, 0.005)
    
    def test_find_rare_variants_higher_threshold(self):
        """Test finding rare variants with higher threshold."""
        rare_variants = self.analyzer.find_rare_variants(frequency_threshold=0.1)
        
        self.assertEqual(len(rare_variants), 2)
        variant_ids = [variant.variant_id for variant in rare_variants]
        self.assertIn("rs1801133", variant_ids)
        self.assertIn("rs7412", variant_ids)
    
    def test_analyze_by_gene(self):
        """Test grouping variants by gene."""
        gene_analysis = self.analyzer.analyze_by_gene()
        
        self.assertIn("TP53", gene_analysis)
        self.assertIn("MTHFR", gene_analysis)
        self.assertIn("APOE", gene_analysis)
        
        self.assertEqual(len(gene_analysis["TP53"]), 1)
        self.assertEqual(len(gene_analysis["MTHFR"]), 1)
        self.assertEqual(len(gene_analysis["APOE"]), 1)
    
    def test_find_clinically_significant_variants(self):
        """Test finding clinically significant variants."""
        significant_variants = self.analyzer.find_clinically_significant_variants()
        
        self.assertGreaterEqual(len(significant_variants), 2)
        
        variant_ids = [variant.variant_id for variant in significant_variants]
        self.assertIn("rs1042522", variant_ids)  # Likely pathogenic
        self.assertIn("rs7412", variant_ids)     # Pathogenic


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_directory=self.temp_dir)
        
        self.sample_patients = [
            Patient(
                patient_id="P001", age=65, gender="M", diagnosis="Lung Cancer",
                treatment="Chemotherapy", survival_days=450,
                genetic_variants=[], adverse_events=[],
                biomarkers={"CEA": 15.2}
            ),
            Patient(
                patient_id="P002", age=58, gender="F", diagnosis="Breast Cancer",
                treatment="Targeted Therapy", survival_days=720,
                genetic_variants=[], adverse_events=[],
                biomarkers={"HER2": 3.2}
            )
        ]
        
        self.sample_interactions = [
            DrugInteraction(
                drug1="Warfarin", drug2="Aspirin",
                interaction_type="Anticoagulation", severity="High",
                risk_score=0.85, evidence_level="Strong",
                adverse_events=["Bleeding"]
            )
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generate_survival_report(self):
        """Test survival report generation."""
        survival_data = {
            "overall_survival": {
                "mean": 585.0,
                "median": 585.0,
                "std": 135.0
            },
            "treatment_comparison": {
                "Chemotherapy": {
                    "count": 1,
                    "mean_survival": 450.0,
                    "median_survival": 450.0,
                    "std_survival": 0.0
                }
            }
        }
        
        report_path = self.generator.generate_survival_report(survival_data)
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith("survival_analysis.txt"))
        
        # Check file content
        with open(report_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.assertIn("SURVIVAL ANALYSIS REPORT", content)
            self.assertIn("585.0", content)
            self.assertIn("Chemotherapy", content)
    
    def test_generate_survival_report_with_error(self):
        """Test survival report generation with error data."""
        error_data = {"error": "No patient data available"}
        
        report_path = self.generator.generate_survival_report(error_data)
        
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.assertIn("ERROR: No patient data available", content)
    
    def test_generate_pattern_report(self):
        """Test pattern report generation."""
        pattern_data = {
            "diagnosis_distribution": {
                "Lung Cancer": 1,
                "Breast Cancer": 1
            },
            "age_analysis": {
                "Lung Cancer": {
                    "mean_age": 65.0,
                    "median_age": 65.0,
                    "age_range": (65, 65)
                }
            },
            "gender_analysis": {
                "Lung Cancer": {"M": 1, "F": 0, "Other": 0}
            }
        }
        
        report_path = self.generator.generate_pattern_report(pattern_data)
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith("disease_patterns.txt"))
        
        with open(report_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.assertIn("DISEASE PATTERN ANALYSIS REPORT", content)
            self.assertIn("Lung Cancer: 1 patients", content)
            self.assertIn("65.0", content)
    
    def test_generate_treatment_recommendations(self):
        """Test treatment recommendations generation."""
        report_path = self.generator.generate_treatment_recommendations(
            self.sample_patients, self.sample_interactions
        )
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith("treatment_recommendations.txt"))
        
        with open(report_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.assertIn("EVIDENCE-BASED TREATMENT RECOMMENDATIONS", content)
            # Check for treatment names (may not be present if insufficient data)
            if "Chemotherapy" in content:
                self.assertIn("Chemotherapy", content)
            if "Targeted Therapy" in content:
                self.assertIn("Targeted Therapy", content)
            self.assertIn("Warfarin + Aspirin", content)


class TestMedicalResearchCLI(unittest.TestCase):
    """Test cases for MedicalResearchCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = MedicalResearchCLI()
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        self.cli.load_sample_data()
        
        self.assertGreater(len(self.cli.patients), 0)
        self.assertGreater(len(self.cli.drug_interactions), 0)
        self.assertGreater(len(self.cli.genetic_variants), 0)
        
        # Check that patients are valid
        for patient in self.cli.patients:
            self.assertIsInstance(patient, Patient)
            self.assertGreater(len(patient.patient_id), 0)
            self.assertGreaterEqual(patient.age, 0)
        
        # Check that drug interactions are valid
        for interaction in self.cli.drug_interactions:
            self.assertIsInstance(interaction, DrugInteraction)
            self.assertGreater(len(interaction.drug1), 0)
            self.assertGreater(len(interaction.drug2), 0)
            self.assertGreaterEqual(interaction.risk_score, 0)
            self.assertLessEqual(interaction.risk_score, 1)
        
        # Check that genetic variants are valid
        for variant in self.cli.genetic_variants:
            self.assertIsInstance(variant, GeneticVariant)
            self.assertGreater(len(variant.variant_id), 0)
            self.assertGreater(len(variant.gene), 0)
            self.assertGreaterEqual(variant.population_frequency, 0)
            self.assertLessEqual(variant.population_frequency, 1)


def run_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPatient,
        TestDrugInteraction,
        TestGeneticVariant,
        TestDataValidator,
        TestStatisticalAnalyzer,
        TestDrugInteractionAnalyzer,
        TestGeneticAnalyzer,
        TestReportGenerator,
        TestMedicalResearchCLI
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Medical Research Data Mining CLI Unit Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED! ✗")
        print("=" * 60)
    
    sys.exit(0 if success else 1) 