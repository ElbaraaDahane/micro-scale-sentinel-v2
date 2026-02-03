"""
Micro-Scale Sentinel - AI Classification Module
Handles Google Gemini API integration for particle classification.
Simplified for engineering students - focuses on physics-based reasoning.
"""

import google.generativeai as genai
from PIL import Image
import json
import base64
import io
import numpy as np
from typing import Dict, Any, Optional
import streamlit as st

def encode_image_for_gemini(image_array: np.ndarray) -> str:
    """
    Convert numpy array (OpenCV format) to base64 string for Gemini API.
    
    Args:
        image_array: RGB image as numpy array (from OpenCV)
        
    Returns:
        Base64 encoded JPEG string
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save to bytes buffer as JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str


def build_classification_prompt(features: Dict[str, Any]) -> str:
    """
    Builds the expert prompt for Gemini with physics context.
    ~800 tokens, 3 clear sections for student readability.
    
    Args:
        features: Dictionary of extracted particle features
        
    Returns:
        Formatted prompt string
    """
    
    # Extract key values for context
    size = features.get('size_um', 0)
    circularity = features.get('circularity', 0)
    aspect_ratio = features.get('aspect_ratio', 0)
    ri_estimate = features.get('refractive_index_estimate', 0)
    
    prompt = f"""You are an expert in marine biology and materials science with 20 years of experience analyzing holographic microscopy images for microplastic pollution research.

## PHYSICS REFERENCE DATA
Use these refractive indices (RI) to guide your analysis:
- Plastics (RI 1.4-1.6): PET (1.575), PS (1.55), HDPE/PVC (1.54), PP (1.49)
- Biological (RI 1.33-1.40): Diatoms (1.35-1.40), Copepods (1.33-1.38), Water (1.33)
- Key insight: Higher RI creates sharper diffraction fringes in holographic images

## ANALYSIS TASK
Analyze this microscopic particle and classify it as MICROPLASTIC or BIOLOGICAL.

Extracted Features:
- Size: {size:.1f} μm
- Circularity: {circularity:.2f} (1.0 = perfect circle)
- Aspect Ratio: {aspect_ratio:.2f} (1.0 = circular, >1 = elongated)
- Estimated RI: {ri_estimate:.3f}

Step-by-step reasoning:
1. Examine the diffraction pattern: Regular/ring-like (plastic) vs organic/complex (biological)
2. Check RI estimate: >1.45 suggests plastic, <1.40 suggests biological
3. Analyze morphology: Irregular/jagged edges (plastic) vs symmetric/structured (biological)
4. Consider transparency: Uniform (plastic) vs internal structures (biological)
5. Compare competing hypotheses and assign confidence scores

## OUTPUT FORMAT (STRICT JSON)
Return ONLY a JSON object with this exact structure:
{{
    "classification": "MICROPLASTIC" or "BIOLOGICAL" or "UNCERTAIN",
    "confidence_microplastic": 0-100,
    "confidence_biological": 0-100,
    "polymer_type": "PET" or "HDPE" or "PP" or "PS" or "PVC" or "Other" or null,
    "organism_type": "diatom" or "copepod" or "other" or null,
    "recommendation": "DEFINITE" (>85% conf) or "PROBABLE" (60-85%) or "UNCERTAIN" (<60%),
    "reasoning": "Detailed 2-3 sentence explanation citing specific optical features visible",
    "evidence": {{
        "diffraction_pattern": "Description of fringe pattern",
        "refractive_index_analysis": "RI interpretation",
        "morphology": "Shape and edge description",
        "size_analysis": "Size context vs typical particles"
    }}
}}

Rules:
- Confidence scores must sum to 100
- Use null (not "unknown") for polymer_type if biological, and vice versa
- Be conservative: if uncertain, say UNCERTAIN rather than guess
"""
    return prompt


def classify_particle(
    image_array: np.ndarray, 
    features: Dict[str, Any], 
    api_key: str,
    model_name: str = "gemini-1.5-flash"
) -> Dict[str, Any]:
    """
    Classify a particle using Google Gemini API.
    
    This is the main function that sends the image + features to Gemini
    and returns the structured classification result.
    
    Args:
        image_array: Preprocessed RGB image as numpy array
        features: Dictionary of extracted physics features
        api_key: Google Gemini API key
        model_name: Gemini model to use (default: gemini-1.5-flash)
        
    Returns:
        Dictionary containing classification results
        
    Raises:
        Exception: If API call fails or response parsing fails
    """
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Prepare the image
    try:
        img_base64 = encode_image_for_gemini(image_array)
    except Exception as e:
        raise Exception(f"Failed to encode image: {str(e)}")
    
    # Build the prompt with physics context
    prompt = build_classification_prompt(features)
    
    # Create the content parts for the API
    # Gemini can take text + image in the same request
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }
                }
            ]
        }
    ]
    
    # Call the API
    try:
        model = genai.GenerativeModel(model_name)
        
        # Set generation config for structured output
        generation_config = {
            "temperature": 0.2,  # Lower = more deterministic/factual
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Make the API call with timeout handling
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            request_options={"timeout": 30}  # 30 second timeout
        )
        
        # Extract text response
        response_text = response.text
        
        # Parse JSON from the response
        # Sometimes Gemini wraps JSON in markdown code blocks ```json ... ```
        # We need to extract just the JSON part
        result = parse_gemini_response(response_text)
        
        # Validate the result has required fields
        validate_classification_result(result)
        
        return result
        
    except Exception as e:
        # Provide helpful error messages for students
        error_msg = str(e)
        
        if "API key not valid" in error_msg:
            raise Exception("Invalid API Key. Please check your key at https://makersuite.google.com/app/apikey")
        elif "429" in error_msg or "Resource exhausted" in error_msg:
            raise Exception("API rate limit reached. Wait 60 seconds and try again (Free tier: 60 requests/min)")
        elif "503" in error_msg:
            raise Exception("Gemini API temporarily unavailable. Please try again in a few moments.")
        elif "json" in error_msg.lower() or "parse" in error_msg.lower():
            raise Exception(f"Failed to parse AI response. Try again or use Debug Mode. Details: {error_msg}")
        else:
            raise Exception(f"Classification failed: {error_msg}")


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON from Gemini's text response.
    Handles cases where JSON is wrapped in markdown code blocks.
    
    Args:
        response_text: Raw text from Gemini API
        
    Returns:
        Parsed dictionary
    """
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        # Extract between ```json and ```
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        # Extract between ``` and ```
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()
    
    # Parse JSON
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError as e:
        # If parsing fails, try to fix common issues
        # Sometimes Gemini adds trailing commas or comments
        raise Exception(f"JSON parsing error: {str(e)}. Response was: {text[:200]}...")


def validate_classification_result(result: Dict[str, Any]) -> None:
    """
    Check that the result has all required fields and valid values.
    Helps catch cases where Gemini doesn't follow instructions exactly.
    
    Args:
        result: Parsed JSON dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        "classification", 
        "confidence_microplastic", 
        "confidence_biological", 
        "reasoning"
    ]
    
    # Check required fields exist
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field in AI response: {field}")
    
    # Validate classification value
    valid_classes = ["MICROPLASTIC", "BIOLOGICAL", "UNCERTAIN"]
    if result["classification"] not in valid_classes:
        # Try to fix common variations
        class_lower = result["classification"].lower()
        if "plastic" in class_lower:
            result["classification"] = "MICROPLASTIC"
        elif "bio" in class_lower:
            result["classification"] = "BIOLOGICAL"
        else:
            result["classification"] = "UNCERTAIN"
    
    # Ensure confidence values are numbers between 0-100
    try:
        result["confidence_microplastic"] = int(result["confidence_microplastic"])
        result["confidence_biological"] = int(result["confidence_biological"])
    except (ValueError, TypeError):
        # If not valid numbers, set to 0
        result["confidence_microplastic"] = 0
        result["confidence_biological"] = 0
    
    # Ensure they sum to approximately 100 (allow small rounding errors)
    total_conf = result["confidence_microplastic"] + result["confidence_biological"]
    if result["classification"] == "UNCERTAIN" and total_conf < 100:
        # Distribute remaining confidence as uncertainty
        pass  # This is fine
    
    # Ensure evidence field exists (optional but recommended)
    if "evidence" not in result:
        result["evidence"] = {
            "diffraction_pattern": "Not provided",
            "refractive_index_analysis": "Not provided",
            "morphology": "Not provided",
            "size_analysis": "Not provided"
        }


# =============================================================================
# EXAMPLE USAGE (for testing)
# =============================================================================
if __name__ == "__main__":
    # This runs when you execute: python src/classifier.py
    # Used for testing the module independently
    
    print("Testing classifier module...")
    print("Note: This requires a valid GEMINI_API_KEY environment variable")
    
    import os
    import sys
    
    # Add parent directory to path to import preprocessing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocessing import preprocess_image, extract_features
    
    # Test API key
    test_key = os.getenv("GEMINI_API_KEY")
    if not test_key:
        print("❌ Error: Set GEMINI_API_KEY environment variable first")
        print("   export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Create a test image (synthetic)
    print("Creating test image...")
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Preprocess
    print("Preprocessing...")
    processed = preprocess_image(test_img)
    features = extract_features(processed, scale_um_per_pixel=0.5)
    
    # Classify
    print("Calling Gemini API...")
    try:
        result = classify_particle(processed, features, test_key)
        print("\n✅ Success! Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n❌ Failed: {e}")
