"""
Improved evaluation using validated datasets instead of manual questions
"""

import json
import random
from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd

class ValidatedDatasetLoader:
    """Load and prepare validated medical and math datasets"""
    
    def __init__(self):
        self.datasets = {}
        
    def load_medical_dataset(self, dataset_name: str = "medqa", num_samples: int = 100) -> List[Dict[str, Any]]:
        """Load validated medical questions from MedQA or MedMCQA"""
        
        if dataset_name == "medqa":
            # Load MedQA USMLE dataset
            try:
                dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
                questions = []
                
                for i, item in enumerate(random.sample(list(dataset), min(num_samples, len(dataset)))):
                    # MedQA format
                    question_data = {
                        "question_id": f"medqa_{i+1:03d}",
                        "question": item["question"],
                        "options": item["options"],  # Dictionary of A, B, C, D options
                        "answer": item["answer"],
                        "answer_idx": item["answer_idx"],
                        "domain": "medicine",
                        "difficulty": "professional",  # USMLE level
                        "type": "medical",
                        "source": "USMLE",
                        "validated": True
                    }
                    questions.append(question_data)
                    
                print(f"✅ Loaded {len(questions)} validated MedQA questions from USMLE")
                return questions
                
            except Exception as e:
                print(f"❌ Error loading MedQA: {e}")
                print("💡 Install with: pip install datasets")
                return self._fallback_medical_samples()
        
        elif dataset_name == "medmcqa":
            try:
                dataset = load_dataset("medmcqa", split="test")
                questions = []
                
                for i, item in enumerate(random.sample(list(dataset), min(num_samples, len(dataset)))):
                    question_data = {
                        "question_id": f"medmcqa_{i+1:03d}",
                        "question": item["question"],
                        "options": {
                            "A": item["opa"],
                            "B": item["opb"], 
                            "C": item["opc"],
                            "D": item["opd"]
                        },
                        "answer": item[f"op{['a','b','c','d'][item['cop']]}"],
                        "answer_idx": ['A','B','C','D'][item['cop']],
                        "domain": item.get("subject_name", "medicine"),
                        "difficulty": "professional",
                        "type": "medical",
                        "source": "AIIMS/NEET",
                        "validated": True
                    }
                    questions.append(question_data)
                    
                print(f"✅ Loaded {len(questions)} validated MedMCQA questions")
                return questions
                
            except Exception as e:
                print(f"❌ Error loading MedMCQA: {e}")
                return self._fallback_medical_samples()
    
    def load_math_dataset(self, dataset_name: str = "gsm8k", num_samples: int = 100) -> List[Dict[str, Any]]:
        """Load validated math questions from GSM8K or MATH dataset"""
        
        if dataset_name == "gsm8k":
            try:
                dataset = load_dataset("openai/gsm8k", "main", split="test")
                questions = []
                
                for i, item in enumerate(random.sample(list(dataset), min(num_samples, len(dataset)))):
                    # Extract final answer from GSM8K format
                    answer_text = item["answer"]
                    final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else "Unknown"
                    
                    question_data = {
                        "question_id": f"gsm8k_{i+1:03d}",
                        "question": item["question"],
                        "answer": final_answer,
                        "solution_steps": answer_text.split("####")[0].strip() if "####" in answer_text else answer_text,
                        "domain": "mathematics",
                        "difficulty": "grade_school",
                        "type": "mathematical",
                        "source": "GSM8K",
                        "validated": True,
                        "requires_reasoning": True
                    }
                    questions.append(question_data)
                
                print(f"✅ Loaded {len(questions)} validated GSM8K questions")
                return questions
                
            except Exception as e:
                print(f"❌ Error loading GSM8K: {e}")
                return self._fallback_math_samples()
        
        elif dataset_name == "math":
            try:
                dataset = load_dataset("hendrycks/competition_math", split="test")
                questions = []
                
                for i, item in enumerate(random.sample(list(dataset), min(num_samples, len(dataset)))):
                    question_data = {
                        "question_id": f"math_{i+1:03d}",
                        "question": item["problem"],
                        "answer": item["solution"],
                        "domain": item["type"],  # algebra, geometry, etc.
                        "difficulty": f"level_{item['level']}",  # 1-5 difficulty
                        "type": "mathematical",
                        "source": "Competition Math",
                        "validated": True,
                        "requires_reasoning": True
                    }
                    questions.append(question_data)
                
                print(f"✅ Loaded {len(questions)} validated Competition Math questions")
                return questions
                
            except Exception as e:
                print(f"❌ Error loading MATH dataset: {e}")
                return self._fallback_math_samples()
    
    def _fallback_medical_samples(self) -> List[Dict[str, Any]]:
        """Fallback medical questions if datasets not available"""
        print("⚠️  Using fallback medical questions - NOT validated")
        return [
            {
                "question_id": "fallback_med_001",
                "question": "A 65-year-old patient presents with chest pain, diaphoresis, and ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
                "options": {"A": "Anterior STEMI", "B": "Inferior STEMI", "C": "Unstable angina", "D": "Pericarditis"},
                "answer": "Inferior STEMI",
                "answer_idx": "B",
                "domain": "cardiology",
                "difficulty": "medium",
                "type": "medical",
                "source": "Manual",
                "validated": False
            }
        ]
    
    def _fallback_math_samples(self) -> List[Dict[str, Any]]:
        """Fallback math questions if datasets not available"""
        print("⚠️  Using fallback math questions - NOT validated")
        return [
            {
                "question_id": "fallback_math_001",
                "question": "Sarah has 24 stickers. She gives 1/3 of them to her brother and 1/4 of the remainder to her sister. How many stickers does Sarah have left?",
                "answer": "12",
                "domain": "arithmetic",
                "difficulty": "easy",
                "type": "mathematical",
                "source": "Manual",
                "validated": False
            }
        ]
    
    def create_balanced_dataset(self, total_questions: int = 120) -> List[Dict[str, Any]]:
        """Create a balanced dataset using validated sources"""
        
        # Calculate splits
        medical_count = total_questions // 2
        math_count = total_questions - medical_count
        
        questions = []
        
        # Load medical questions (split between MedQA and MedMCQA)
        medqa_count = medical_count // 2
        medmcqa_count = medical_count - medqa_count
        
        questions.extend(self.load_medical_dataset("medqa", medqa_count))
        questions.extend(self.load_medical_dataset("medmcqa", medmcqa_count))
        
        # Load math questions (split between GSM8K and MATH)
        gsm8k_count = math_count // 2
        math_comp_count = math_count - gsm8k_count
        
        questions.extend(self.load_math_dataset("gsm8k", gsm8k_count))
        questions.extend(self.load_math_dataset("math", math_comp_count))
        
        # Shuffle final dataset
        random.shuffle(questions)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"Total questions: {len(questions)}")
        
        # Count by source
        source_counts = {}
        validation_counts = {"validated": 0, "manual": 0}
        
        for q in questions:
            source = q.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            
            if q.get("validated", False):
                validation_counts["validated"] += 1
            else:
                validation_counts["manual"] += 1
        
        print(f"Sources: {source_counts}")
        print(f"Validation: {validation_counts}")
        
        return questions
    
    def validate_dataset_quality(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of loaded datasets"""
        
        validation_report = {
            "total_questions": len(questions),
            "validated_percentage": 0,
            "sources": {},
            "domains": {},
            "difficulties": {},
            "quality_issues": []
        }
        
        validated_count = 0
        
        for q in questions:
            # Count validated vs manual
            if q.get("validated", False):
                validated_count += 1
            
            # Count sources
            source = q.get("source", "Unknown")
            validation_report["sources"][source] = validation_report["sources"].get(source, 0) + 1
            
            # Count domains
            domain = q.get("domain", "Unknown")
            validation_report["domains"][domain] = validation_report["domains"].get(domain, 0) + 1
            
            # Count difficulties
            difficulty = q.get("difficulty", "Unknown")
            validation_report["difficulties"][difficulty] = validation_report["difficulties"].get(difficulty, 0) + 1
            
            # Check for quality issues
            if not q.get("question", "").strip():
                validation_report["quality_issues"].append(f"Empty question: {q.get('question_id')}")
            
            if not q.get("answer", "").strip():
                validation_report["quality_issues"].append(f"Empty answer: {q.get('question_id')}")
        
        validation_report["validated_percentage"] = (validated_count / len(questions)) * 100
        
        return validation_report

# Example usage
def demonstrate_improved_evaluation():
    """Demonstrate the improved evaluation with validated datasets"""
    
    print("🔬 IMPROVED EVALUATION WITH VALIDATED DATASETS")
    print("=" * 60)
    
    # Initialize loader
    loader = ValidatedDatasetLoader()
    
    # Create balanced dataset
    questions = loader.create_balanced_dataset(total_questions=50)  # Smaller for demo
    
    # Validate quality
    report = loader.validate_dataset_quality(questions)
    
    print(f"\n📋 QUALITY REPORT:")
    print(f"Validated questions: {report['validated_percentage']:.1f}%")
    print(f"Sources: {report['sources']}")
    print(f"Domains: {report['domains']}")
    
    if report['quality_issues']:
        print(f"⚠️  Quality issues: {len(report['quality_issues'])}")
    else:
        print("✅ No quality issues detected")
    
    # Show examples
    print(f"\n📝 SAMPLE QUESTIONS:")
    for i, q in enumerate(questions[:3]):
        print(f"\n{i+1}. {q['question_id']} ({q['source']}):")
        print(f"   Q: {q['question'][:100]}...")
        print(f"   Validated: {'✅' if q.get('validated') else '❌'}")
    
    return questions

if __name__ == "__main__":
    questions = demonstrate_improved_evaluation()
