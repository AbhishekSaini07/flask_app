# -*- coding: utf-8 -*-
# app.py

import google.generativeai as genai
import os
import json
import re
import numpy as np
import time
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional # Optional for cleaner type hints

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SBERT_MODEL_NAME = 'all-mpnet-base-v2' # Using a stronger model for initial retrieval
ASSESSMENT_DATA_FILE = 'assessments.json'

# --- Load Environment Variables (for API Key) ---
load_dotenv() # Load variables from .env file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Global Variables (to hold loaded data/models) ---
assessments_data: List[Dict[str, Any]] = []
sbert_model: Optional[SentenceTransformer] = None
gemini_configured: bool = False
sbert_model_loaded: bool = False
data_loaded: bool = False

# --- Helper Function - Extract Minutes ---
def extract_minutes(duration_str: Optional[str]) -> Optional[int]:
    """Extracts the number of minutes from a string like '30 minutes'."""
    if not isinstance(duration_str, str): return None
    match = re.search(r'(\d+)\s*minutes?', duration_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\s*mins?', duration_str, re.IGNORECASE) # Added 'mins' variation
    if match:
        return int(match.group(1))
    return None

# --- Helper Function - Generate Gemini Prompt ---
def generate_llm_relevance_prompt(query_text: str, candidate_assessments: List[Dict[str, Any]]) -> str:
    """Creates the prompt for Gemini to re-rank assessments."""
    max_candidates_in_prompt = 25 # Limit to avoid exceeding context limits
    candidates_to_prompt = candidate_assessments[:max_candidates_in_prompt]

    prompt = f"You are an expert assistant helping hiring managers find relevant pre-employment assessments.\n"
    prompt += f"Based *only* on the provided assessment names and descriptions, please re-rank the following candidate assessments according to their relevance to the user's query.\n\n"
    prompt += f"User Query: \"{query_text}\"\n\n"
    prompt += "Candidate Assessments:\n"
    prompt += "----------------------\n"
    for i, assessment in enumerate(candidates_to_prompt):
        prompt += f"{i+1}. Name: {assessment.get('assignment', 'N/A')}\n"
        desc = assessment.get('description', 'N/A')
        prompt += f"   Description: {desc[:300]}{'...' if len(desc) > 300 else ''}\n" # Limit description length
    prompt += "----------------------\n\n"
    prompt += f"Instructions: Return ONLY the re-ranked list of the {len(candidates_to_prompt)} assessment names provided above, one name per line, starting with the most relevant assessment based on the query and the descriptions. Do not include numbers, introductory text, explanations, or assessments not listed above."
    return prompt

# --- Helper Function - Call Gemini API ---
def call_gemini_api(prompt: str) -> Optional[str]:
    """Calls the Gemini API and handles basic errors."""
    global gemini_configured
    if not gemini_configured:
        logging.error("Gemini API key not configured. Cannot call API.")
        return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Using Flash model for speed/cost
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024, # Adjust if needed based on initial_retrieve
                temperature=0.1
            )
            # safety_settings=... # Optional safety settings
        )
        if response.parts:
             return response.text
        else:
             # Check for specific block reasons if available
             block_reason = "Unknown"
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 block_reason = str(response.prompt_feedback.block_reason) # Convert enum to string
                 logging.warning(f"Gemini response blocked or empty. Reason: {block_reason}")
             else:
                 logging.warning("Gemini response was empty or potentially blocked without specific reason provided.")
             return None
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return None

# --- Helper Function - Parse LLM Ranking ---
def parse_llm_ranking(response_text: Optional[str], candidate_names: List[str]) -> List[str]:
    """Parses the LLM's ranked list of names."""
    if not response_text:
        return []

    ranked_names = []
    lines = response_text.strip().split('\n')
    candidate_names_set = set(candidate_names) # Original names sent to LLM
    # Create a lowercased map for more robust fuzzy matching
    candidate_lower_map = {name.lower(): name for name in candidate_names}

    for line in lines:
        # Clean up potential numbering/whitespace from LLM output
        name_from_llm = line.strip().lstrip('0123456789.- ').rstrip()
        if not name_from_llm: continue # Skip empty lines

        name_lower = name_from_llm.lower()

        # 1. Direct match (case-insensitive for robustness)
        matched = False
        if name_lower in candidate_lower_map:
            original_case_name = candidate_lower_map[name_lower]
            if original_case_name not in ranked_names: # Add only once
                ranked_names.append(original_case_name)
                matched = True
                # logging.debug(f"LLM Parse: Direct match for '{name_from_llm}'")

        # 2. Simple Substring match (if direct failed)
        if not matched:
            for candidate_name_lower, original_case_name in candidate_lower_map.items():
                if name_lower in candidate_name_lower or candidate_name_lower in name_lower:
                    if original_case_name not in ranked_names: # Add candidate name if match found
                        ranked_names.append(original_case_name)
                        logging.info(f"LLM Parse: Fuzzy matched LLM output '{name_from_llm}' to candidate '{original_case_name}'")
                        matched = True
                        break # Take the first fuzzy match

        if not matched:
             logging.warning(f"LLM Parse: LLM returned name not matched to candidates: '{name_from_llm}'")

    # 3. Ensure all original candidates are included if LLM missed some
    returned_names_set = set(ranked_names)
    missing_names = [name for name in candidate_names if name not in returned_names_set]
    if missing_names:
        logging.info(f"LLM Parse: Appending {len(missing_names)} candidates potentially missed by LLM: {missing_names}")
        ranked_names.extend(missing_names) # Append missing ones at the end (lowest rank)

    return ranked_names

# --- Helper Function - Parse Duration Constraint ---
def parse_duration_constraint(query_text: str) -> Optional[int]:
    """
    Parses query for duration constraints (e.g., "less than 40 minutes").
    Returns max duration in minutes, or None.
    """
    max_duration = None
    # Stricter patterns first (e.g., "less than", "max")
    match = re.search(r'\b(less than|under|at most|max|maximum|within|no more than)\s+(\d+)\s*(minutes?|mins?)\b', query_text, re.IGNORECASE)
    if match:
        max_duration = int(match.group(2))
        logging.info(f"Duration Parse: Found constraint: {match.group(1)} {max_duration} mins")
        return max_duration # Return immediately if a strict constraint is found

    # Less strict pattern (just number + minutes) - use only if no strict one found
    match = re.search(r'\b(\d+)\s*(minutes?|mins?)\b', query_text, re.IGNORECASE)
    if match:
        potential_duration = int(match.group(1))
        # Avoid overriding a stricter constraint found above
        if max_duration is None:
            # Check for context suggesting it's a limit (heuristic)
             if any(word in query_text.lower() for word in [" shorter ", " shorter,", " maximum ", " max ", " within "]):
                 max_duration = potential_duration
                 logging.info(f"Duration Parse: Found potential limit: {max_duration} mins")
             else:
                 logging.info(f"Duration Parse: Found duration '{potential_duration} mins', but no clear limit keyword. Ignoring as constraint.")

    return max_duration

# --- Main Logic Function - Gemini Re-ranking ---
def get_recommendations_gemini_rerank(
    query_text: str,
    assessments_data_local: List[Dict[str, Any]],
    sbert_model_local: SentenceTransformer,
    top_k: int = 10,
    initial_retrieve: int = 25
) -> List[Dict[str, Any]]:
    """
    Generates recommendations using SentenceBERT retrieval + Gemini re-ranking.
    Uses local copies of data/model passed as arguments.
    """
    global gemini_configured # Access global flag

    if not assessments_data_local or not sbert_model_local:
        logging.error("Cannot generate recommendations - data or SBERT model not ready.")
        return []

    start_time = time.time()

    # 1. Initial Candidate Retrieval (SentenceBERT)
    try:
        logging.info(f"Encoding query: '{query_text[:50]}...'")
        query_embedding = sbert_model_local.encode([query_text])[0]
        # Ensure embeddings are available in the data
        if not assessments_data_local or 'embedding' not in assessments_data_local[0]:
             logging.error("Embeddings not found in assessment data.")
             return []
        all_assessment_embeddings = np.array([item['embedding'] for item in assessments_data_local])
        logging.info(f"Calculating similarities against {len(all_assessment_embeddings)} assessments.")
        similarities = cosine_similarity(query_embedding.reshape(1, -1), all_assessment_embeddings)[0]
    except Exception as e:
        logging.error(f"Error during SBERT embedding or similarity calculation: {e}")
        return []

    # Ensure we don't request more candidates than available
    num_assessments = len(assessments_data_local)
    num_candidates_to_retrieve = min(initial_retrieve, num_assessments)

    if num_candidates_to_retrieve <= 0:
        logging.warning("No candidates to retrieve based on settings or data availability.")
        return []

    # Get indices of top N similar assessments
    semantic_ranked_indices = np.argsort(similarities)[::-1][:num_candidates_to_retrieve]
    candidate_assessments = [assessments_data_local[i] for i in semantic_ranked_indices]
    candidate_names = [item['assignment'] for item in candidate_assessments]

    retrieval_time = time.time()
    logging.info(f"Initial SBERT retrieval ({len(candidate_assessments)} candidates) took {retrieval_time - start_time:.2f}s")

    # --- LLM Re-ranking Section ---
    reranked_candidates = candidate_assessments # Default to semantic order
    if gemini_configured and candidate_assessments: # Only proceed if API key is set and candidates exist
        prompt = generate_llm_relevance_prompt(query_text, candidate_assessments)
        logging.info(f"Sending prompt for {len(candidate_assessments)} candidates to Gemini...")
        llm_start_time = time.time()
        llm_response_text = call_gemini_api(prompt)
        llm_call_time = time.time()
        logging.info(f"Gemini API call took {llm_call_time - llm_start_time:.2f}s")

        if llm_response_text:
            logging.info("Parsing Gemini response...")
            parse_start_time = time.time()
            ranked_names_from_llm = parse_llm_ranking(llm_response_text, candidate_names)
            parse_end_time = time.time()
            logging.info(f"LLM Response Parsing took {parse_end_time - parse_start_time:.2f}s")
            # logging.debug(f"LLM ranked names ({len(ranked_names_from_llm)}): {ranked_names_from_llm[:top_k]}...") # Optional full list log

            # Build the re-ranked list of full assessment data
            assessment_map = {item['assignment']: item for item in candidate_assessments}
            temp_reranked_list = []
            processed_names = set() # Ensure unique items
            for name in ranked_names_from_llm:
                if name in assessment_map and name not in processed_names:
                    temp_reranked_list.append(assessment_map[name])
                    processed_names.add(name)
                # else: # Log if a name from LLM wasn't in the original candidate map (should be handled by parse_llm_ranking warnings)
                #    logging.warning(f"LLM Re-ranking: Name '{name}' from LLM not found in candidate map.")

            # If re-ranking produced a list, use it
            if temp_reranked_list:
                 reranked_candidates = temp_reranked_list
                 logging.info(f"Successfully re-ranked {len(reranked_candidates)} candidates using Gemini.")
            else:
                 logging.warning("Gemini parsing resulted in an empty list. Falling back to SBERT semantic ranking.")
        else:
            logging.warning("Gemini did not return a valid response. Falling back to SBERT semantic ranking.")
    elif not candidate_assessments:
         logging.warning("No candidates retrieved by SBERT, skipping Gemini re-ranking.")
    else: # Gemini skipped (not configured)
        logging.info("Skipping Gemini re-ranking (API key not configured). Using SBERT semantic ranking.")

    # --- Duration Filtering ---
    filter_start_time = time.time()
    max_duration_filter = parse_duration_constraint(query_text)
    final_recommendations = []

    if max_duration_filter is not None:
        logging.info(f"Applying duration filter: <= {max_duration_filter} minutes")
        for assessment in reranked_candidates:
            duration_val = assessment.get('duration_minutes') # Already extracted
            if duration_val is not None and duration_val <= max_duration_filter:
                final_recommendations.append(assessment)
            # else: logging.debug(f"Filtering out '{assessment.get('assignment')}' (Duration: {duration_val})") # Optional debug
            if len(final_recommendations) >= top_k: # Stop if we have enough
                break
        logging.info(f"Found {len(final_recommendations)} assessments after duration filter.")
    else:
        # No duration filter, just take the top_k from the (potentially) re-ranked list
        final_recommendations = reranked_candidates[:top_k]
        logging.info("No duration filter applied.")

    filter_time = time.time()
    logging.info(f"Filtering & final selection took {filter_time - filter_start_time:.2f}s")
    total_time = time.time()
    logging.info(f"Total recommendation processing time: {total_time - start_time:.2f}s")

    return final_recommendations # Return final top K (already sliced or filtered)

# --- API Formatting Function ---
def format_recommendations_for_api(raw_recommendations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Formats the recommendation list to match the API specification."""
    output_recommendations = []
    # Define keys required in the final output
    required_output_keys = ['assignment', 'url', 'remote_support', 'adaptive_support', 'duration', 'test_type']

    for item in raw_recommendations:
        formatted_item = {}
        # Use .get() to safely access keys and provide defaults
        formatted_item['assignment'] = item.get('assignment', 'N/A')
        formatted_item['url'] = item.get('url', '#') # Default URL if missing
        formatted_item['remote_support'] = item.get('remote_support', 'N/A')
        formatted_item['adaptive_support'] = item.get('adaptive_support', 'N/A')
        formatted_item['duration'] = item.get('duration', 'N/A') # Keep original duration string
        formatted_item['test_type'] = item.get('test_type', []) # Default to empty list

        # You could add a check here to ensure essential keys like 'assignment' are present
        # if formatted_item['assignment'] == 'N/A':
        #    logging.warning("Skipping item with missing assignment name.")
        #    continue

        output_recommendations.append(formatted_item)

    # Wrap in the final response structure
    final_response = {"recommendations": output_recommendations}
    return final_response

# --- Initialization Function (Load Data and Models) ---
def load_resources():
    """Loads assessment data, SBERT model, and configures Gemini."""
    global assessments_data, sbert_model, gemini_configured, sbert_model_loaded, data_loaded

    # 1. Configure Gemini
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Optional: Add a simple test call if needed, but configure is usually enough
            # list_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # if not list_models: raise Exception("No models available")
            gemini_configured = True
            logging.info("Gemini API Key configured successfully.")
        except Exception as e:
            logging.error(f"An error occurred during Gemini API key configuration: {e}")
            gemini_configured = False
    else:
        logging.warning("GEMINI_API_KEY not found in environment variables. Gemini re-ranking will be skipped.")
        gemini_configured = False

    # 2. Load SBERT Model
    try:
        logging.info(f"Loading Sentence Transformer model '{SBERT_MODEL_NAME}'...")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        sbert_model_loaded = True
        logging.info("SBERT Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load SBERT model: {e}")
        sbert_model = None
        sbert_model_loaded = False

    # 3. Load and Preprocess Assessment Data
    if sbert_model_loaded: # Only proceed if model loaded
        try:
            logging.info(f"Loading assessment data from '{ASSESSMENT_DATA_FILE}'...")
            with open(ASSESSMENT_DATA_FILE, 'r', encoding='utf-8') as f:
                assessments_data_raw_list = json.load(f)
            logging.info(f"Loaded {len(assessments_data_raw_list)} raw assessment entries.")

            processed_data = []
            texts_to_embed = []
            logging.info("Preprocessing assessment data (duration, text for embedding)...")
            for item in assessments_data_raw_list:
                duration_minutes = extract_minutes(item.get('duration'))
                item['duration_minutes'] = duration_minutes # Store calculated minutes

                # Basic validation - must have assignment, description, and parsable duration
                if not item.get('assignment') or not item.get('description'): # Removed duration check here, handle later
                    logging.warning(f"Skipping assessment due to missing name/description: {item.get('assignment', 'Unknown')}")
                    continue

                # Prepare text for embedding (handle potentially missing 'test_type')
                test_types = item.get('test_type', [])
                test_type_str = ', '.join(test_types) if isinstance(test_types, list) else 'N/A'
                text = f"Assessment: {item['assignment']}. Type: {test_type_str}. Description: {item['description']}"
                texts_to_embed.append(text)
                processed_data.append(item) # Add item to the list of valid data

            if processed_data and texts_to_embed:
                logging.info(f"Generating SBERT embeddings for {len(processed_data)} valid assessments...")
                assessment_embeddings = sbert_model.encode(texts_to_embed, show_progress_bar=True)

                # Add embeddings to the data structure
                for i, item in enumerate(processed_data):
                    item['embedding'] = assessment_embeddings[i]

                assessments_data = processed_data # Assign the final processed list
                data_loaded = True
                logging.info(f"Embeddings generated. Total assessments ready: {len(assessments_data)}")
            else:
                 logging.error("No valid assessments found after preprocessing or no texts to embed.")
                 data_loaded = False

        except FileNotFoundError:
            logging.error(f"Assessment data file not found: '{ASSESSMENT_DATA_FILE}'")
            assessments_data = []
            data_loaded = False
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from '{ASSESSMENT_DATA_FILE}': {e}")
            assessments_data = []
            data_loaded = False
        except Exception as e:
            logging.error(f"An unexpected error occurred during data loading/preprocessing: {e}")
            assessments_data = []
            data_loaded = False
    else:
        logging.error("SBERT model not loaded. Cannot preprocess data or generate embeddings.")
        data_loaded = False

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Resources on Startup ---
# This ensures models and data are loaded only once when the app starts
with app.app_context():
    load_resources()

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health Check Endpoint: Verifies the API is running.
    """
    # Check if essential components are loaded
    if data_loaded and sbert_model_loaded:
         # Could add a Gemini check if it's considered essential for health
         # status = "healthy" if gemini_configured else "healthy_no_reranking"
        return jsonify({"status": "healthy"}), 200
    else:
        # Indicate a problem if core components failed to load
        return jsonify({"status": "unhealthy", "reason": "Core components failed to load."}), 503 # Service Unavailable

@app.route('/recommendations', methods=['POST'])
def get_assessment_recommendations():
    """
    Assessment Recommendation Endpoint: Accepts a query and returns relevant assessments.
    """
    # Check if resources are loaded
    if not data_loaded or not sbert_model_loaded:
        logging.error("Request received but core components (data/sbert) are not loaded.")
        return jsonify({"error": "Service Unavailable. Core components not loaded."}), 503

    # --- Input Validation ---
    if not request.is_json:
        logging.warning("Received non-JSON request.")
        return jsonify({"error": "Request must be JSON"}), 400 # Bad Request

    data = request.get_json()
    query_text = data.get('query_text')

    if not query_text or not isinstance(query_text, str):
        logging.warning(f"Received invalid query_text: {query_text}")
        return jsonify({"error": "Missing or invalid 'query_text' field in JSON body"}), 400 # Bad Request

    # --- Recommendation Logic ---
    try:
        # Pass the globally loaded data and model to the function
        # Ensure sbert_model is not None before passing
        if sbert_model is None:
             logging.error("SBERT model is None, cannot proceed.")
             return jsonify({"error": "Internal Server Error: Model not available."}), 500

        raw_recs = get_recommendations_gemini_rerank(
            query_text=query_text,
            assessments_data_local=assessments_data, # Use the pre-loaded data
            sbert_model_local=sbert_model,         # Use the pre-loaded model
            top_k=10,
            initial_retrieve=25 # Number of candidates for potential Gemini re-ranking
        )

        # Ensure at least 1 recommendation if possible, but max 10
        if not raw_recs:
            logging.info(f"No recommendations found for query: '{query_text[:50]}...'")
            # Return empty list as per requirement (min 1, max 10 implies 0 is acceptable if none found)
            formatted_output = {"recommendations": []}
        else:
             formatted_output = format_recommendations_for_api(raw_recs) # Already handles top_k slicing

        return jsonify(formatted_output), 200

    except Exception as e:
        logging.exception(f"An unexpected error occurred during recommendation generation for query '{query_text[:50]}...': {e}") # Log full traceback
        return jsonify({"error": "Internal Server Error"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to be accessible from other devices on the network
    # Use debug=True only for development (enables auto-reload and detailed errors)
    # For production/deployment, use a proper WSGI server like Gunicorn or Waitress
    app.run(host='0.0.0.0', port=5000, debug=False) # Turn Debug OFF for stable running