"""
Statistically Meaningful GPT-OSS Evaluation
Scales up to 100+ questions with proper statistical analysis
"""

import torch
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import random
from pathlib import Path
from test import ValidatedDatasetLoader

@dataclass
class ReasoningStep:
    step_text: str
    knowledge_claim: str
    is_knowledge_correct: Optional[bool] = None
    info_gain: Optional[float] = None

@dataclass
class EvaluationResult:
    question_id: str
    question: str
    model_answer: str
    reasoning_steps: List[ReasoningStep]
    knowledge_index: float
    info_gain: float
    is_correct: Optional[bool] = None
    reasoning_level: str = ""
    response_length: int = 0
    generation_time: float = 0.0
    domain: str = ""
    difficulty: str = ""

class StatisticalReasoningEvaluator:
    """
    Large-scale evaluation with proper statistical analysis
    """
    
    def __init__(self, 
                 gpt_oss_model_path: str = "openai/gpt-oss-20b",
                 batch_size: int = 1,
                 save_checkpoints: bool = True):
        
        self.model_path = gpt_oss_model_path
        self.reasoning_levels = ["low", "medium", "high"]
        self.batch_size = batch_size
        self.save_checkpoints = save_checkpoints
        
        print("🧠 Loading GPT-OSS model for large-scale evaluation...")
        torch.cuda.empty_cache()
        gc.collect()
        
        self.tokenizer = AutoTokenizer.from_pretrained(gpt_oss_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            gpt_oss_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_device = next(self.model.parameters()).device
        print(f"✅ Model loaded on {self.model_device}")
    
    def generate_comprehensive_dataset(self, total_questions: int = 120) -> List[Dict[str, str]]:
        """Generate a comprehensive dataset for statistical analysis"""
        
        # Medical questions by domain and difficulty
        medical_questions = {
            "cardiology": {
                "easy": [
                    {"q": "What are the classic symptoms of myocardial infarction?", "a": "chest pain, shortness of breath, diaphoresis"},
                    {"q": "What is the normal range for systolic blood pressure?", "a": "90-120 mmHg"},
                    {"q": "What does ECG show in atrial fibrillation?", "a": "irregular rhythm, absent P waves"},
                    {"q": "What is the first-line treatment for hypertension?", "a": "ACE inhibitors or diuretics"},
                    {"q": "What causes heart failure with reduced ejection fraction?", "a": "systolic dysfunction"},
                ],
                "medium": [
                    {"q": "A 55-year-old presents with chest pain and ST elevation in leads II, III, aVF. What is the diagnosis and management?", "a": "inferior STEMI, primary PCI"},
                    {"q": "What are the contraindications to thrombolytic therapy in STEMI?", "a": "recent surgery, bleeding disorders, stroke"},
                    {"q": "Differentiate between stable and unstable angina clinically.", "a": "unstable has rest pain, crescendo pattern"},
                    {"q": "What is the pathophysiology of heart failure with preserved ejection fraction?", "a": "diastolic dysfunction, impaired relaxation"},
                    {"q": "When should you suspect cardiac tamponade and what are Beck's triad?", "a": "elevated JVP, hypotension, muffled heart sounds"},
                ],
                "hard": [
                    {"q": "A 45-year-old with cocaine use presents with chest pain. ECG shows diffuse ST elevation. Troponins are elevated. Coronary angiography shows normal arteries. What is the diagnosis and pathophysiology?", "a": "cocaine-induced coronary spasm, myocarditis"},
                    {"q": "Explain the mechanism of action and contraindications of SGLT2 inhibitors in heart failure.", "a": "glucose excretion, volume reduction, contraindicated in DKA"},
                    {"q": "What is the difference between Type A and Type B aortic dissection in terms of management and prognosis?", "a": "Type A needs surgery, Type B medical management"},
                ]
            },
            "pulmonology": {
                "easy": [
                    {"q": "What are the symptoms of pneumonia?", "a": "fever, cough, chest pain"},
                    {"q": "What does spirometry show in COPD?", "a": "reduced FEV1/FVC ratio"},
                    {"q": "What is the most common cause of pneumonia in healthy adults?", "a": "Streptococcus pneumoniae"},
                    {"q": "What is the first-line treatment for asthma exacerbation?", "a": "bronchodilators, corticosteroids"},
                ],
                "medium": [
                    {"q": "A 28-year-old presents with sudden onset dyspnea and pleuritic chest pain. What is your differential and initial workup?", "a": "pneumothorax, pulmonary embolism, chest X-ray"},
                    {"q": "What are the indications for non-invasive ventilation in COPD exacerbation?", "a": "pH 7.25-7.35, hypercapnia, respiratory distress"},
                    {"q": "Differentiate between exudative and transudative pleural effusions.", "a": "protein ratio >0.5, LDH criteria"},
                ],
                "hard": [
                    {"q": "A 35-year-old immunocompromised patient presents with bilateral ground-glass opacities on CT. What organisms should you consider and what diagnostic tests would you order?", "a": "PCP, CMV, fungal, BAL with staining"},
                ]
            },
            "neurology": {
                "easy": [
                    {"q": "What are the signs of increased intracranial pressure?", "a": "headache, vomiting, papilledema"},
                    {"q": "What is the classic triad of Parkinson's disease?", "a": "tremor, rigidity, bradykinesia"},
                    {"q": "What imaging is first-line for suspected stroke?", "a": "non-contrast CT head"},
                ],
                "medium": [
                    {"q": "A 65-year-old presents with sudden onset right-sided weakness and aphasia. What is your immediate management?", "a": "stroke protocol, tPA if eligible"},
                    {"q": "What are the different types of seizures and how do you treat status epilepticus?", "a": "focal/generalized, benzodiazepines first-line"},
                ],
                "hard": [
                    {"q": "A 25-year-old woman presents with optic neuritis and MRI shows periventricular white matter lesions. What is the diagnosis and disease-modifying treatments?", "a": "multiple sclerosis, interferons, glatiramer"},
                ]
            },
            "endocrinology": {
                "easy": [
                    {"q": "What are the symptoms of hypothyroidism?", "a": "fatigue, weight gain, cold intolerance"},
                    {"q": "What is the target HbA1c for most diabetic patients?", "a": "less than 7%"},
                    {"q": "What are the signs of diabetic ketoacidosis?", "a": "hyperglycemia, ketones, acidosis"},
                ],
                "medium": [
                    {"q": "A 45-year-old diabetic presents with glucose 450 mg/dL, no ketones, osmolality 350. What is the diagnosis and treatment?", "a": "hyperosmolar hyperglycemic state, fluid resuscitation"},
                    {"q": "What is the workup for suspected Cushing's syndrome?", "a": "dexamethasone suppression test, 24-hour urine cortisol"},
                ],
                "hard": [
                    {"q": "A 30-year-old presents with hypertension, hypokalemia, and metabolic alkalosis. What is your differential and diagnostic approach?", "a": "primary aldosteronism, renovascular hypertension, plasma aldosterone"},
                ]
            },
            "gastroenterology": {
                "easy": [
                    {"q": "What are the alarm symptoms in dyspepsia that require urgent evaluation?", "a": "weight loss, dysphagia, bleeding"},
                    {"q": "What is the most common cause of upper GI bleeding?", "a": "peptic ulcer disease"},
                ],
                "medium": [
                    {"q": "A 45-year-old presents with right upper quadrant pain, fever, and jaundice. What is Charcot's triad and your management?", "a": "cholangitis, ERCP, antibiotics"},
                    {"q": "What are the complications of inflammatory bowel disease?", "a": "strictures, perforation, cancer risk"},
                ],
                "hard": [
                    {"q": "A 55-year-old with cirrhosis presents with altered mental status. What is the pathophysiology and treatment of hepatic encephalopathy?", "a": "ammonia toxicity, lactulose, rifaximin"},
                ]
            }
        }
        
        # Mathematical/reasoning questions for comparison
        math_questions = {
            "easy": [
                {"q": "Solve: 2x + 5 = 15", "a": "x = 5"},
                {"q": "Find the derivative of f(x) = x²", "a": "f'(x) = 2x"},
                {"q": "What is 15% of 200?", "a": "30"},
                {"q": "Solve: x² - 4 = 0", "a": "x = ±2"},
            ],
            "medium": [
                {"q": "Find the derivative of f(x) = x³ + 2x² - 5x + 1", "a": "f'(x) = 3x² + 4x - 5"},
                {"q": "Solve the system: 2x + y = 7, x - y = 2", "a": "x = 3, y = 1"},
                {"q": "What is the area under y = x² from x = 0 to x = 2?", "a": "8/3"},
            ],
            "hard": [
                {"q": "Find the Taylor series expansion of e^x around x = 0", "a": "1 + x + x²/2! + x³/3! + ..."},
                {"q": "Solve the differential equation dy/dx = y with initial condition y(0) = 1", "a": "y = e^x"},
            ]
        }
        
        # Generate balanced dataset
        questions = []
        question_id = 1
        
        # Add medical questions
        for domain, difficulties in medical_questions.items():
            for difficulty, q_list in difficulties.items():
                for item in q_list:
                    questions.append({
                        "question_id": f"med_{question_id:03d}",
                        "question": item["q"],
                        "answer": item["a"],
                        "domain": domain,
                        "difficulty": difficulty,
                        "type": "medical"
                    })
                    question_id += 1
        
        # Add math questions
        for difficulty, q_list in math_questions.items():
            for item in q_list:
                questions.append({
                    "question_id": f"math_{question_id:03d}",
                    "question": item["q"],
                    "answer": item["a"],
                    "domain": "mathematics",
                    "difficulty": difficulty,
                    "type": "mathematical"
                })
                question_id += 1
        
        # Shuffle and limit to requested size
        random.shuffle(questions)
        return questions[:total_questions]
    
    def format_prompt(self, question: str, reasoning_level: str) -> str:
        """Format prompt based on reasoning level"""
        reasoning_instructions = {
            "low": "Give a brief, direct answer in 1-2 sentences.",
            "medium": "Think through this step by step before answering. Provide 3-4 reasoning steps.",
            "high": "Provide detailed reasoning with multiple steps, considering all relevant factors, differential diagnosis when applicable, and supporting evidence. Show your complete thought process."
        }
        
        system_msg = f"""You are a medical expert. {reasoning_instructions[reasoning_level]}

Structure your response as:
<analysis>
[Your reasoning here]
</analysis>
<answer>
[Your final answer here]
</answer>"""

        prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_response(self, question: str, reasoning_level: str) -> Dict[str, Any]:
        """Generate response with error handling and optimization"""
        prompt = self.format_prompt(question, reasoning_level)
        torch.cuda.empty_cache()
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
            
            # Optimized parameters for each level
            generation_params = {
                "low": {"max_new_tokens": 96, "temperature": 0.3, "top_p": 0.8},
                "medium": {"max_new_tokens": 256, "temperature": 0.5, "top_p": 0.9},
                "high": {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.95}
            }
            
            params = generation_params[reasoning_level]
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = full_response[len(prompt):].strip()
            
            # Parse response
            reasoning = ""
            answer = response_text
            
            if "<analysis>" in response_text and "</analysis>" in response_text:
                analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
                if analysis_match:
                    reasoning = analysis_match.group(1).strip()
            
            if "<answer>" in response_text and "</answer>" in response_text:
                answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
            
            # Cleanup
            del outputs, inputs
            torch.cuda.empty_cache()
            
            return {
                "reasoning": reasoning if reasoning else response_text,
                "answer": answer,
                "full_response": response_text,
                "generation_time": generation_time,
                "response_length": len(response_text.split()),
                "success": True
            }
            
        except Exception as e:
            print(f"    Generation error: {e}")
            torch.cuda.empty_cache()
            return {
                "reasoning": "",
                "answer": "",
                "full_response": "",
                "generation_time": 0.0,
                "response_length": 0,
                "success": False,
                "error": str(e)
            }
    
    def decompose_reasoning(self, reasoning_text: str) -> List[ReasoningStep]:
        """Improved reasoning decomposition"""
        if not reasoning_text.strip():
            return []
        
        steps_raw = []
        
        # Look for numbered steps
        numbered_pattern = r'(\d+\.?\s+[^.!?]*[.!?])'
        numbered_steps = re.findall(numbered_pattern, reasoning_text)
        
        if len(numbered_steps) >= 2:
            steps_raw = numbered_steps
        else:
            # Split by sentences and group logically
            sentences = re.split(r'[.!?]+', reasoning_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Group consecutive short sentences
            grouped_steps = []
            current_step = ""
            
            for sentence in sentences:
                if len(current_step) + len(sentence) < 100:
                    current_step += (" " + sentence if current_step else sentence)
                else:
                    if current_step:
                        grouped_steps.append(current_step)
                    current_step = sentence
            
            if current_step:
                grouped_steps.append(current_step)
            
            steps_raw = grouped_steps
        
        # Convert to ReasoningStep objects
        reasoning_steps = []
        for step_text in steps_raw:
            step_text = step_text.strip()
            if len(step_text) > 15:
                reasoning_steps.append(ReasoningStep(
                    step_text=step_text,
                    knowledge_claim=step_text,
                    is_knowledge_correct=True  # Simplified for large-scale eval
                ))
        
        return reasoning_steps
    
    def calculate_info_gain(self, steps: List[ReasoningStep], question: str, answer: str) -> List[float]:
        """Calculate information gain for each step"""
        if not steps:
            return []
        
        info_gains = []
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        for i, step in enumerate(steps):
            step_words = set(step.step_text.lower().split())
            
            # Information metrics
            step_length = len(step_words)
            question_overlap = len(step_words & question_words)
            answer_relevance = len(step_words & answer_words)
            
            # Calculate base information gain
            base_gain = min(step_length / 20.0, 2.0)
            relevance_bonus = (question_overlap + answer_relevance * 2) / 10.0
            position_weight = (i + 1) / len(steps) * 0.5
            
            info_gain = base_gain + relevance_bonus + position_weight
            info_gains.append(info_gain)
            step.info_gain = info_gain
        
        return info_gains
    
    def check_answer_correctness(self, model_answer: str, ground_truth: str) -> bool:
        """Enhanced answer correctness checking"""
        model_lower = model_answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        # Direct substring match
        if truth_lower in model_lower:
            return True
        
        # Keyword matching with threshold
        truth_words = [w for w in truth_lower.split() if len(w) > 2]
        model_words = model_lower.split()
        
        if not truth_words:
            return False
        
        matching_words = sum(1 for word in truth_words if word in model_words)
        match_ratio = matching_words / len(truth_words)
        
        return match_ratio > 0.5
    
    def evaluate_question_fast(self, item: Dict[str, str], reasoning_level: str) -> EvaluationResult:
        """Fast evaluation for large-scale processing"""
        question_id = item["question_id"]
        question = item["question"]
        ground_truth = item["answer"]
        domain = item.get("domain", "unknown")
        difficulty = item.get("difficulty", "unknown")
        
        # Generate response
        response = self.generate_response(question, reasoning_level)
        
        if not response["success"]:
            return EvaluationResult(
                question_id=question_id,
                question=question,
                model_answer="",
                reasoning_steps=[],
                knowledge_index=0.0,
                info_gain=0.0,
                is_correct=False,
                reasoning_level=reasoning_level,
                response_length=0,
                generation_time=0.0,
                domain=domain,
                difficulty=difficulty
            )
        
        # Process reasoning
        steps = self.decompose_reasoning(response["reasoning"])
        info_gains = self.calculate_info_gain(steps, question, response["answer"])
        
        # Calculate metrics
        knowledge_index = 1.0  # Simplified for speed
        avg_info_gain = np.mean(info_gains) if info_gains else 0.0
        is_correct = self.check_answer_correctness(response["answer"], ground_truth)
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            model_answer=response["answer"],
            reasoning_steps=steps,
            knowledge_index=knowledge_index,
            info_gain=avg_info_gain,
            is_correct=is_correct,
            reasoning_level=reasoning_level,
            response_length=response["response_length"],
            generation_time=response["generation_time"],
            domain=domain,
            difficulty=difficulty
        )
    
    def run_statistical_evaluation(self, 
                                 questions: List[Dict[str, str]], 
                                 checkpoint_every: int = 20) -> Dict[str, List[EvaluationResult]]:
        """Run large-scale evaluation with checkpointing"""
        
        print(f"🚀 STATISTICAL EVALUATION: {len(questions)} questions × 3 levels = {len(questions) * 3} total evaluations")
        print("=" * 80)
        
        all_results = {level: [] for level in self.reasoning_levels}
        total_evaluations = len(questions) * len(self.reasoning_levels)
        completed = 0
        
        start_time = time.time()
        
        for i, item in enumerate(questions):
            print(f"\n📝 Question {i+1}/{len(questions)} ({item['question_id']}) - {item['domain']}/{item['difficulty']}")
            print(f"Q: {item['question'][:80]}...")
            
            level_results = {}
            
            # Evaluate at each reasoning level
            for level in self.reasoning_levels:
                print(f"  [{level.upper()}] Processing...", end=" ")
                
                result = self.evaluate_question_fast(item, level)
                all_results[level].append(result)
                level_results[level] = result
                completed += 1
                
                print(f"✓ Steps: {len(result.reasoning_steps)}, "
                      f"Correct: {result.is_correct}, "
                      f"InfoGain: {result.info_gain:.2f}, "
                      f"Time: {result.generation_time:.1f}s")
            
            # Progress tracking
            elapsed = time.time() - start_time
            avg_time_per_eval = elapsed / completed
            eta = avg_time_per_eval * (total_evaluations - completed)
            
            print(f"  📊 Progress: {completed}/{total_evaluations} ({completed/total_evaluations:.1%})")
            print(f"  ⏱️  ETA: {eta/60:.1f} minutes")
            
            # Checkpoint saving
            if self.save_checkpoints and (i + 1) % checkpoint_every == 0:
                checkpoint_file = f"checkpoint_{i+1}_{len(questions)}.json"
                self.save_checkpoint(all_results, checkpoint_file)
                print(f"  💾 Checkpoint saved: {checkpoint_file}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Evaluation complete! Total time: {total_time/60:.1f} minutes")
        
        return all_results
    
    def save_checkpoint(self, results: Dict[str, List[EvaluationResult]], filename: str):
        """Save evaluation checkpoint"""
        serializable_results = {}
        for level, level_results in results.items():
            serializable_results[level] = [asdict(result) for result in level_results]
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def statistical_analysis(self, all_results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """Comprehensive statistical analysis"""
        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 80)
        
        analysis = {
            "sample_sizes": {},
            "descriptive_stats": {},
            "statistical_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "domain_analysis": {},
            "difficulty_analysis": {}
        }
        
        # Sample sizes
        for level in self.reasoning_levels:
            analysis["sample_sizes"][level] = len(all_results[level])
        
        print(f"📏 Sample sizes: {analysis['sample_sizes']}")
        
        # Descriptive statistics
        for level in self.reasoning_levels:
            results = all_results[level]
            if not results:
                continue
            
            # Extract metrics
            accuracy_scores = [int(r.is_correct) for r in results]
            info_gain_scores = [r.info_gain for r in results]
            response_lengths = [r.response_length for r in results]
            generation_times = [r.generation_time for r in results]
            reasoning_steps = [len(r.reasoning_steps) for r in results]
            
            analysis["descriptive_stats"][level] = {
                "accuracy": {
                    "mean": np.mean(accuracy_scores),
                    "std": np.std(accuracy_scores),
                    "n": len(accuracy_scores)
                },
                "info_gain": {
                    "mean": np.mean(info_gain_scores),
                    "std": np.std(info_gain_scores),
                    "median": np.median(info_gain_scores),
                    "q25": np.percentile(info_gain_scores, 25),
                    "q75": np.percentile(info_gain_scores, 75)
                },
                "response_length": {
                    "mean": np.mean(response_lengths),
                    "std": np.std(response_lengths)
                },
                "generation_time": {
                    "mean": np.mean(generation_times),
                    "std": np.std(generation_times)
                },
                "reasoning_steps": {
                    "mean": np.mean(reasoning_steps),
                    "std": np.std(reasoning_steps)
                }
            }
        
        # Statistical significance tests
        if len(self.reasoning_levels) >= 2:
            print("\n🧮 Statistical Significance Tests:")
            
            # Pairwise comparisons for accuracy
            for i in range(len(self.reasoning_levels)):
                for j in range(i+1, len(self.reasoning_levels)):
                    level1, level2 = self.reasoning_levels[i], self.reasoning_levels[j]
                    
                    acc1 = [int(r.is_correct) for r in all_results[level1]]
                    acc2 = [int(r.is_correct) for r in all_results[level2]]
                    
                    # Chi-square test for accuracy
                    chi2, p_acc = stats.chi2_contingency([[sum(acc1), len(acc1)-sum(acc1)], 
                                                         [sum(acc2), len(acc2)-sum(acc2)]])[:2]
                    
                    # T-test for info gain
                    ig1 = [r.info_gain for r in all_results[level1]]
                    ig2 = [r.info_gain for r in all_results[level2]]
                    t_stat, p_ig = stats.ttest_ind(ig1, ig2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(ig1)-1)*np.var(ig1) + (len(ig2)-1)*np.var(ig2)) / (len(ig1)+len(ig2)-2))
                    cohens_d = (np.mean(ig1) - np.mean(ig2)) / pooled_std if pooled_std > 0 else 0
                    
                    analysis["statistical_tests"][f"{level1}_vs_{level2}"] = {
                        "accuracy_chi2_p": p_acc,
                        "info_gain_ttest_p": p_ig,
                        "cohens_d": cohens_d,
                        "significant_accuracy": p_acc < 0.05,
                        "significant_info_gain": p_ig < 0.05
                    }
                    
                    print(f"  {level1.upper()} vs {level2.upper()}:")
                    print(f"    Accuracy: χ² p = {p_acc:.4f} {'*' if p_acc < 0.05 else ''}")
                    print(f"    Info Gain: t-test p = {p_ig:.4f} {'*' if p_ig < 0.05 else ''}")
                    print(f"    Effect size (d): {cohens_d:.3f}")
        
        # Domain analysis
        domains = set()
        for level_results in all_results.values():
            domains.update(r.domain for r in level_results)
        
        for domain in domains:
            domain_stats = {}
            for level in self.reasoning_levels:
                domain_results = [r for r in all_results[level] if r.domain == domain]
                if domain_results:
                    domain_stats[level] = {
                        "n": len(domain_results),
                        "accuracy": np.mean([int(r.is_correct) for r in domain_results]),
                        "info_gain": np.mean([r.info_gain for r in domain_results])
                    }
            analysis["domain_analysis"][domain] = domain_stats
        
        # Print summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        for level in self.reasoning_levels:
            stats_data = analysis["descriptive_stats"][level]
            print(f"  {level.upper()}:")
            print(f"    Accuracy: {stats_data['accuracy']['mean']:.1%} ± {stats_data['accuracy']['std']:.3f}")
            print(f"    Info Gain: {stats_data['info_gain']['mean']:.3f} ± {stats_data['info_gain']['std']:.3f}")
            print(f"    Avg Steps: {stats_data['reasoning_steps']['mean']:.1f}")
            print(f"    Avg Time: {stats_data['generation_time']['mean']:.1f}s")
        
        return analysis
    
    def create_statistical_visualizations(self, all_results: Dict[str, List[EvaluationResult]], analysis: Dict[str, Any]):
        """Create comprehensive statistical visualizations"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Prepare DataFrame
        df_data = []
        for level, results in all_results.items():
            for result in results:
                df_data.append({
                    'Reasoning Level': level.capitalize(),
                    'Accuracy': int(result.is_correct),
                    'Information Gain': result.info_gain,
                    'Response Length': result.response_length,
                    'Generation Time': result.generation_time,
                    'Reasoning Steps': len(result.reasoning_steps),
                    'Domain': result.domain,
                    'Difficulty': result.difficulty
                })
        
        df = pd.DataFrame(df_data)
        
        # 1. Accuracy comparison with confidence intervals
        ax1 = plt.subplot(3, 4, 1)
        accuracy_by_level = df.groupby('Reasoning Level')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        accuracy_by_level['se'] = accuracy_by_level['std'] / np.sqrt(accuracy_by_level['count'])
        accuracy_by_level['ci'] = 1.96 * accuracy_by_level['se']
        
        bars = ax1.bar(accuracy_by_level['Reasoning Level'], accuracy_by_level['mean'], 
                      yerr=accuracy_by_level['ci'], capsize=5, alpha=0.8)
        ax1.set_title('Accuracy by Reasoning Level\n(95% Confidence Intervals)')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add sample sizes
        for i, (bar, n) in enumerate(zip(bars, accuracy_by_level['count'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'n={n}', ha='center', va='bottom')
        
        # 2. Information Gain distribution
        ax2 = plt.subplot(3, 4, 2)
        sns.boxplot(data=df, x='Reasoning Level', y='Information Gain', ax=ax2)
        ax2.set_title('Information Gain Distribution')
        
        # 3. Domain-specific performance
        ax3 = plt.subplot(3, 4, 3)
        domain_pivot = df.groupby(['Domain', 'Reasoning Level'])['Accuracy'].mean().unstack()
        domain_pivot.plot(kind='bar', ax=ax3, alpha=0.8)
        ax3.set_title('Accuracy by Domain')
        ax3.set_ylabel('Accuracy')
        ax3.legend(title='Reasoning Level')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. Difficulty analysis
        ax4 = plt.subplot(3, 4, 4)
        difficulty_pivot = df.groupby(['Difficulty', 'Reasoning Level'])['Accuracy'].mean().unstack()
        difficulty_pivot.plot(kind='bar', ax=ax4, alpha=0.8)
        ax4.set_title('Accuracy by Difficulty')
        ax4.set_ylabel('Accuracy')
        ax4.legend(title='Reasoning Level')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # 5. Response length vs accuracy
        ax5 = plt.subplot(3, 4, 5)
        for level in df['Reasoning Level'].unique():
            level_data = df[df['Reasoning Level'] == level]
            ax5.scatter(level_data['Response Length'], level_data['Accuracy'], 
                       alpha=0.6, label=level, s=30)
        ax5.set_xlabel('Response Length (words)')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Response Length vs Accuracy')
        ax5.legend()
        
        # 6. Generation time efficiency
        ax6 = plt.subplot(3, 4, 6)
        efficiency_data = df.groupby('Reasoning Level').agg({
            'Generation Time': 'mean',
            'Accuracy': 'mean'
        }).reset_index()
        
        for i, row in efficiency_data.iterrows():
            ax6.scatter(row['Generation Time'], row['Accuracy'], s=200, alpha=0.7)
            ax6.annotate(row['Reasoning Level'], 
                        (row['Generation Time'], row['Accuracy']),
                        xytext=(5, 5), textcoords='offset points')
        
        ax6.set_xlabel('Average Generation Time (s)')
        ax6.set_ylabel('Average Accuracy')
        ax6.set_title('Speed vs Accuracy Trade-off')
        ax6.grid(True, alpha=0.3)
        
        # 7. Reasoning steps analysis
        ax7 = plt.subplot(3, 4, 7)
        sns.violinplot(data=df, x='Reasoning Level', y='Reasoning Steps', ax=ax7)
        ax7.set_title('Reasoning Steps Distribution')
        
        # 8. Statistical significance heatmap
        ax8 = plt.subplot(3, 4, 8)
        if "statistical_tests" in analysis:
            sig_matrix = np.zeros((3, 3))
            levels = ['low', 'medium', 'high']
            
            for i, level1 in enumerate(levels):
                for j, level2 in enumerate(levels):
                    if i != j:
                        key = f"{level1}_vs_{level2}"
                        if key in analysis["statistical_tests"]:
                            p_val = analysis["statistical_tests"][key]["info_gain_ttest_p"]
                            sig_matrix[i, j] = -np.log10(p_val) if p_val > 0 else 10
            
            im = ax8.imshow(sig_matrix, cmap='RdYlBu_r')
            ax8.set_xticks(range(3))
            ax8.set_yticks(range(3))
            ax8.set_xticklabels([l.capitalize() for l in levels])
            ax8.set_yticklabels([l.capitalize() for l in levels])
            ax8.set_title('Statistical Significance\n(-log10 p-value)')
            plt.colorbar(im, ax=ax8)
        
        # 9-12. Domain-specific detailed analysis
        domains = df['Domain'].unique()[:4]  # Top 4 domains
        for idx, domain in enumerate(domains):
            ax = plt.subplot(3, 4, 9 + idx)
            domain_data = df[df['Domain'] == domain]
            
            sns.barplot(data=domain_data, x='Reasoning Level', y='Information Gain', 
                       ax=ax, alpha=0.8)
            ax.set_title(f'{domain.capitalize()}\nInformation Gain')
        
        plt.tight_layout()
        plt.savefig('statistical_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Statistical visualizations saved as 'statistical_evaluation_results.png'")
    
    def save_comprehensive_results(self, all_results: Dict[str, List[EvaluationResult]], 
                                 analysis: Dict[str, Any], questions: List[Dict[str, str]]):
        """Save comprehensive results with statistical analysis"""
        
        # Convert to serializable format
        serializable_results = {}
        for level, results in all_results.items():
            serializable_results[level] = [asdict(result) for result in results]
        
        output_data = {
            "metadata": {
                "model_path": self.model_path,
                "evaluation_timestamp": time.time(),
                "total_questions": len(questions),
                "reasoning_levels": self.reasoning_levels,
                "domains": list(set(q["domain"] for q in questions)),
                "difficulties": list(set(q["difficulty"] for q in questions))
            },
            "statistical_analysis": analysis,
            "detailed_results": serializable_results,
            "summary_stats": {
                level: {
                    "total_questions": len(results),
                    "accuracy": np.mean([int(r.is_correct) for r in results]),
                    "avg_info_gain": np.mean([r.info_gain for r in results]),
                    "avg_response_length": np.mean([r.response_length for r in results]),
                    "avg_generation_time": np.mean([r.generation_time for r in results]),
                    "avg_reasoning_steps": np.mean([len(r.reasoning_steps) for r in results])
                }
                for level, results in all_results.items()
            }
        }
        
        with open("statistical_gpt_oss_evaluation.json", "w") as f:
            # In the save_comprehensive_results function, add this conversion:


            def convert_for_json(obj):
                """Convert problematic types for JSON serialization"""
                if isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

            # Use this instead:
            json.dump(output_data, f, indent=2, default=convert_for_json)           
                    # Also save CSV for easy analysis
        df_data = []
        for level, results in all_results.items():
            for result in results:
                df_data.append({
                    'question_id': result.question_id,
                    'reasoning_level': result.reasoning_level,
                    'domain': result.domain,
                    'difficulty': result.difficulty,
                    'accuracy': int(result.is_correct),
                    'info_gain': result.info_gain,
                    'response_length': result.response_length,
                    'generation_time': result.generation_time,
                    'reasoning_steps': len(result.reasoning_steps),
                    'question': result.question[:100],
                    'model_answer': result.model_answer[:200]
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv("statistical_gpt_oss_results.csv", index=False)
        
        print("💾 Results saved:")
        print("   📊 statistical_gpt_oss_evaluation.json (comprehensive)")
        print("   📝 statistical_gpt_oss_results.csv (tabular)")

def generate_comprehensive_dataset(total_questions: int = 120):
    """Replace the manual dataset with validated questions"""
    loader = ValidatedDatasetLoader()
    return loader.create_balanced_dataset(total_questions)

def main():
    """Main evaluation with statistical rigor"""
    
    print("🎯 STATISTICALLY MEANINGFUL GPT-OSS EVALUATION")
    print("📊 Large-scale Knowledge vs Reasoning Analysis")
    print("=" * 80)
    
    # Configuration
    TOTAL_QUESTIONS = 120  # Adjust based on your needs and time constraints
    MODEL_PATH = "openai/gpt-oss-20b"  # Update to your model
    
    print(f"📝 Generating {TOTAL_QUESTIONS} diverse questions...")
    
    # Initialize evaluator
    evaluator = StatisticalReasoningEvaluator(
        gpt_oss_model_path=MODEL_PATH,
        save_checkpoints=True
    )
    
    # Generate comprehensive dataset
    questions = generate_comprehensive_dataset(120)  # Get 120 validated questions

    
    print(f"✅ Dataset generated: {len(questions)} questions")
    print(f"📊 Domains: {set(q['domain'] for q in questions)}")
    print(f"📊 Difficulties: {set(q['difficulty'] for q in questions)}")
    print(f"📊 Total evaluations: {len(questions) * 3}")
    
    # Estimate time
    est_time = len(questions) * 3 * 15 / 60  # ~15s per evaluation
    print(f"⏰ Estimated time: {est_time:.1f} minutes")
    
    input("\nPress Enter to start evaluation...")
    
    # Run evaluation
    all_results = evaluator.run_statistical_evaluation(questions)
    
    # Statistical analysis
    analysis = evaluator.statistical_analysis(all_results)
    
    # Create visualizations
    evaluator.create_statistical_visualizations(all_results, analysis)
    
    # Save results
    evaluator.save_comprehensive_results(all_results, analysis, questions)
    
    print("\n🎉 STATISTICAL EVALUATION COMPLETE!")
    print("\n📋 Final Summary:")
    
    for level in evaluator.reasoning_levels:
        results = all_results[level]
        accuracy = np.mean([int(r.is_correct) for r in results])
        info_gain = np.mean([r.info_gain for r in results])
        avg_time = np.mean([r.generation_time for r in results])
        
        print(f"  {level.upper()}: {len(results)} questions, "
              f"Accuracy: {accuracy:.1%}, "
              f"Info Gain: {info_gain:.3f}, "
              f"Avg Time: {avg_time:.1f}s")
    
    # Key statistical insights
    if "statistical_tests" in analysis:
        sig_comparisons = [k for k, v in analysis["statistical_tests"].items() 
                          if v["significant_info_gain"] or v["significant_accuracy"]]
        print(f"\n📊 Significant differences found in {len(sig_comparisons)} comparisons")
    
    print("\n📁 Check these files for detailed analysis:")
    print("   📊 statistical_gpt_oss_evaluation.json")
    print("   📝 statistical_gpt_oss_results.csv") 
    print("   🖼️  statistical_evaluation_results.png")

if __name__ == "__main__":
    main()
