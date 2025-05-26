import re
import json
import difflib
import os

import json
import networkx as nx
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import utils
from retrieval import MedRAG
from tqdm import tqdm
import re
import difflib
import os


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index

def match_choice(text,options):
    # For HuatuoGPT-o1
    if '## Final Response\n\n' in text:
        text = text.split('## Final Response\n\n')[-1]
    # for our model
    elif '## Final Answer\n\n' in text:
        text = text.split('## Final Answer\n\n')[-1]
    # for R1
    elif '</think>' in text:
        text = text.split('</think>')[-1]
    
    # for strict prompt 
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first,ans_last],1

    # non strict
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    matches = list(re.finditer(r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first,ans_last],1

    text = text.lower()
    opsindex = [(opt,text.rindex(options[opt].lower())) for opt in options if options[opt].lower() in text]
    if len(opsindex) > 0:
        ans_last = sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0]
        opsindex = [(opt,text.index(options[opt].lower())) for opt in options if options[opt].lower() in text]
        ans_first = sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0]
        return [ans_first,ans_last],2
    else:
        oplabels = [x for x in options]
        opans = [options[x].lower() for x in options]
        ansindex = find_most_similar_index(opans,text.lower())
        return [oplabels[ansindex],oplabels[ansindex]],3

def match(prediction, ground_truth):
    for gt in ground_truth:
        matchres = re.search(r"(\W|^)("+re.escape(gt)+r")(\W|$)",prediction.lower(),re.S)
        if matchres:
            return 1
    return 0



def score(data,ignore_miss= False):
    res = {}
    wrong_data = []
    cor_data = []
    for da in data:
        if 'source' not in da:
            da['source'] = 'unknown'
        if da['source'] not in res:
            res[da['source']] = [0,0,0,0]

        output = da['output']
        ans,ans_type = match_choice(output,da['options'])
        if ignore_miss and ans_type!= 1:
            continue

        da['ans'] = ans
        da['ans_type'] = ans_type

        if ans[0].lower() == da['answer_idx'].lower():
            res[da['source']][1] += 1
            cor_data.append(da)
        else:
            wrong_data.append(da)
        
        if ans[1].lower() == da['answer_idx'].lower():
            res[da['source']][3] += 1

        res[da['source']][2] += 1

    for k in res:
        head_match_score = res[k][1] / res[k][2]
        tail_match_score = res[k][3] / res[k][2]
        if head_match_score > tail_match_score:
            res[k][0] = head_match_score
        else:
            res[k][0] = tail_match_score

    return res,wrong_data,cor_data



def get_results(res_path, cal_acc = True):
    with open(res_path) as f:
        data = json.load(f) 
    if cal_acc:
        res,wrong_data,cor_data =  score(data)  
    else:
        res = {}
    res_file_name = os.path.basename(res_path)
    print(f"*Logging_file: {res_file_name}*")
    print(json.dumps(res,indent=4))
    
    # save results, replace logs with results
    result_path = res_path.replace('logs','result')
    result_file_name = os.path.basename(result_path)
    # add 'result_' prefix to the result file name
    result_file_name = 'result_' + result_file_name
    result_path = os.path.join(os.path.dirname(result_path), result_file_name)
    with open(result_path,'w') as fw:
        json.dump(res,fw,ensure_ascii=False,indent=2)


class RoscoeScorer:
    def __init__(self, emb_model_name: str = 'abhinand/MedEmbed-large-v0.1', device: str = 'cuda'):
        self.device = device
        self.emd_model = SentenceTransformer(emb_model_name).to(self.device)
        
    def embedding_alignment(self, ref_emb, hyp_emb):
        """
        Return embedding matching alignment for each item in hypo_emb
        ref_emb: list of reference embeddings
        hypo_emb: list oh hypothesises embeddings
        """
        scores = []
        for he in hyp_emb:
            # some embeddings can be empty. For example, for latex-style equations, or empty string
            if len(he) > 0:
                out = [self.emd_model.similarity(he, re) for re in ref_emb if len(re) > 0]
                if len(out) > 0:
                    scores.append(max(out))
        return scores
    
    # need to breakdown the context and reasoning into list of sentences first
    def get_scores(self, references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
        """
        references: list of reference texts
        hypotheses: list of hypothesis texts
        """
        ref_emb = self.emd_model.encode(references)
        hyp_emb = self.emd_model.encode(hypotheses)
        
        y_x_sent_emb = self.embedding_alignment(ref_emb, hyp_emb)
        x_y_sent_emb = self.embedding_alignment(hyp_emb, ref_emb)
        
        faithful_score = utils.al_mean(y_x_sent_emb)
        informative_score = (utils.al_mean(y_x_sent_emb) + utils.al_mean(x_y_sent_emb)) / 2.0
        faithful_scores_steps = [step_score.item() for step_score in y_x_sent_emb]
        
        return {
            'faithful_scores_mean': faithful_score.item(),
            'faithful_scores_steps': faithful_scores_steps,
            'informative_scores_full': informative_score.item()
        }
        
    def forward(self, question, reasoning_steps):
        hypotheses = [step['step_text'] for step in reasoning_steps['Steps']]
        # decompose the question into sentences, by splitting at full stops
        references = question.split('.')
        scores = self.get_scores(references, hypotheses)
        return scores
            
                
class RetrievalScorer:
    def __init__(self, llm_name="OpenAI/gpt-4o-0806-nofilter-global", retriever_name="MedCPT", corpus_name="Textbooks", return_gt = False):
        print("Loading RAG model...")
        self.return_gt = return_gt
        self.RAG_model = MedRAG(llm_name=llm_name, rag=True, retriever_name=retriever_name, corpus_name=corpus_name)
        print("RAG model loaded.")
        
        
    def retrieval_score_step(self, step_knowledge: str):
        """
        Args:
            step_knowledge (str): knowledge text for a single reasoning step
        Returns:
            retrieval_score (float): retrieval score for the reasoning step
        """
        answer = ""
        try:
            query = utils.llm_generate_query(step_knowledge)['query']
        except:
            print("No knowledge points found in the reasoning step, generate a perfect score.")
            if self.return_gt:
                return 1.0, answer
            else:
                return 1.0
        
        
        if query.lower() == "none":
            if self.return_gt:
                return 1.0, answer
            else:
                return 1.0

        answer, snippets, scores = self.RAG_model.answer(question=query, k=12)
        llm_judge = utils.llm_judge_knowledge(llm_output=step_knowledge, reference=answer)
        
        if llm_judge.lower() == "true":
            if self.return_gt:
                return 1.0, answer
            else:
                return 1.0
        else:
            if self.return_gt:
                return 0.0, answer
            else:
                return 0.0
    
    def forward(self, reasoning_steps):
        """
        Args:
            reasoning_steps (dict): dictionary containing the reasoning steps
        Returns:
            retrieval_scores (list): list of retrieval scores for each reasoning step
        """
        retrieval_scores = {}
        retrieval_scores['Steps'] = []
        retrieval_scores['Mean'] = 0
        if self.return_gt:
            retrieval_scores['GT'] = []
        for step in reasoning_steps['Steps']:
            if self.return_gt:
                retrieval_score, gt_answer = self.retrieval_score_step(step['knowledge'])
                retrieval_scores['GT'].append(gt_answer)
            else:
                retrieval_score = self.retrieval_score_step(step['knowledge'])
            retrieval_scores['Steps'].append(retrieval_score)
            retrieval_scores['Mean'] += retrieval_score
        retrieval_scores['Mean'] /= len(retrieval_scores['Steps'])
        return retrieval_scores
    

class InformationGainScorer:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def calculate_perplexity(self, logits, target, attention_mask):
        """
        Calculate perplexity from logits and target labels.

        Args:
        - logits (torch.Tensor): Logits output from the model (batch_size, seq_length, vocab_size).
        - target (torch.Tensor): Ground truth labels (batch_size, seq_length).

        Returns:
        - perplexity (float): The perplexity score.
        """

        shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
        shift_labels = target[:, 1:] # shift by 1 token position

        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather the log probabilities for the correct tokens
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask out positions corresponding to padding tokens
        target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

        # Compute the mean negative log-likelihood for each sequence
        negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)

        perplexities = torch.exp(negative_log_likelihood)

        return perplexities

    def calculate_true_answer_probability(self, steps_text, true_answer):
        """
        Given the current steps in text format, calculate the probability of the true answer.
        
        Args:
        - steps_text: A string representing the steps taken so far.
        - true_answer: The correct answer in text.

        Returns:
        - Probability of the true answer given the steps.
        """
        # Tokenize input (steps) and true answer
        

        inputs = self.tokenizer(steps_text + ' Answer: ' + true_answer, return_tensors="pt")
        input_with_answer_ids = inputs.input_ids
        attention_mask_with_answer_ids = inputs.attention_mask

        with torch.no_grad():
            logits = self.model(input_with_answer_ids.to(self.model.device), return_dict=True).logits.cpu()
        answer_perplexity = self.calculate_perplexity(logits, input_with_answer_ids, attention_mask_with_answer_ids)

        return answer_perplexity.item()

 
    def information_gain_score_step(self, cumulative_step_knowledge: str, answer: str):
        """
        Args:
            cumulative_step_knowledge (str): knowledge text for reasoning step so far
            answer (str): true answer
        Returns:
            information_gain_score (float): retrieval score for the reasoning step
        """
        true_answer_probability = self.calculate_true_answer_probability(cumulative_step_knowledge, answer)
        return true_answer_probability
        
    
    def forward(self, question, reasoning_steps, answer):
        """
        Args:
            reasoning_steps (dict): dictionary containing the reasoning steps
        Returns:
            retrieval_scores (list): list of retrieval scores for each reasoning step
        """
        information_gain_scores = {}
        information_gain_scores['Steps'] = []
        information_gain_scores['Mean'] = 0
        information_gain_scores['Info_gain'] = 0

        if reasoning_steps['Steps'] <= 2:
            return information_gain_scores
            
        cumulative_step_actions = question

        information_gain_score = self.information_gain_score_step(cumulative_step_actions, answer)
        information_gain_scores['Steps'].append(information_gain_score)
        
        for step in reasoning_steps['Steps']:
            cumulative_step_actions += '\n' + step['knowledge']
            information_gain_score = self.information_gain_score_step(cumulative_step_actions, answer)
            information_gain_scores['Steps'].append(information_gain_score)
            diff = information_gain_scores['Steps'][-2] - information_gain_scores['Steps'][-1]
            information_gain_scores['Info_gain'] += diff
            information_gain_scores['Mean'] += information_gain_score
        information_gain_scores['Mean'] /= len(information_gain_scores['Steps'])
        information_gain_scores['Info_gain'] /= len(information_gain_scores['Steps'])
        
        return information_gain_scores    