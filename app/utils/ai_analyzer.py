import os
import json
import requests
import re
from datetime import datetime

def analyze_blockage_with_groq(image_description, caption):
    """
    Use Groq AI to analyze both the image description and user caption
    to provide a comprehensive analysis of the road blockage situation
    
    Args:
        image_description (str): Description of the image content from image analysis
        caption (str): The user-provided caption for the road blockage
        
    Returns:
        dict: Analysis results including estimated time, severity, and detailed analysis
    """
    # Check if we have a Groq API key in environment
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if not groq_api_key:
        print("No Groq API key found in environment variables")
        # Fall back to rule-based estimation
        estimated_time = simulate_ai_response(caption)
        return {
            "estimated_time": estimated_time,
            "severity": "Unknown",
            "analysis": "Analysis not available - No Groq API key provided",
            "factors": []
        }
    
    try:
        # Actual API endpoint for Groq
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Prepare the prompt for the AI that includes both image description and caption
        system_message = """
        You are an expert traffic analyst specializing in road blockages and traffic disruptions. 
        Analyze the road blockage situation based on the image description and user report. 
        Consider factors like:
        1. Number of vehicles involved
        2. Presence of emergency vehicles
        3. Visible damage or obstacles
        4. Weather conditions
        5. Type of road and location
        
        Provide a comprehensive analysis including:
        1. Estimated blockage time in minutes
        2. Severity level (Minor, Moderate, Major, Severe)
        3. Key factors affecting the blockage
        4. Detailed analysis of the situation
        
        Format your response as valid JSON with the following structure:
        {{
            "estimated_time": <integer_minutes>,
            "severity": <string>,
            "analysis": <string>,
            "factors": [<string>, <string>, ...]
        }}
        """
        
        user_message = f"""
        Image description: {image_description}
        
        User report: {caption}
        
        Analyze this road blockage situation and estimate how long it will last.
        """
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        # Make the actual API call to Groq
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            response_content = result['choices'][0]['message']['content']
            
            # Extract the JSON from the response
            # Sometimes the model might wrap the JSON in markdown code blocks
            json_match = re.search(r'```json\s*(.+?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_content
            
            # Parse the JSON response
            try:
                analysis_result = json.loads(json_str)
                
                # Ensure all expected fields are present
                if 'estimated_time' not in analysis_result:
                    analysis_result['estimated_time'] = simulate_ai_response(caption)
                if 'severity' not in analysis_result:
                    analysis_result['severity'] = "Unknown"
                if 'analysis' not in analysis_result:
                    analysis_result['analysis'] = "Detailed analysis not provided"
                if 'factors' not in analysis_result:
                    analysis_result['factors'] = []
                
                return analysis_result
            except json.JSONDecodeError as e:
                print(f"Error parsing Groq response as JSON: {e}")
                print(f"Raw response: {response_content}")
                # Fall back to rule-based estimation
                estimated_time = simulate_ai_response(caption)
                # Generate a more helpful fallback analysis
                severity, analysis, factors = generate_fallback_explanation(caption, estimated_time)
                return {
                    "estimated_time": estimated_time,
                    "severity": severity,
                    "analysis": analysis,
                    "factors": factors
                }
        else:
            print(f"Error from Groq API: {response.status_code} - {response.text}")
            # Fall back to rule-based estimation
            estimated_time = simulate_ai_response(caption)
            # Generate a more helpful fallback analysis
            severity, analysis, factors = generate_fallback_explanation(caption, estimated_time)
            return {
                "estimated_time": estimated_time,
                "severity": severity,
                "analysis": analysis,
                "factors": factors
            }
            
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Fall back to rule-based estimation
        estimated_time = simulate_ai_response(caption)
        # Generate a more helpful fallback analysis
        severity, analysis, factors = generate_fallback_explanation(caption, estimated_time)
        return {
            "estimated_time": estimated_time,
            "severity": severity,
            "analysis": analysis,
            "factors": factors
        }

def estimate_blockage_time(caption, image_metadata=None):
    """
    Estimate the blockage time using Groq AI or fallback to rule-based system
    
    Args:
        caption (str): The user-provided caption for the road blockage
        image_metadata (dict): Optional metadata from image analysis
        
    Returns:
        int: Estimated blockage time in minutes
        dict: Full analysis results if image_metadata is provided
    """
    # If image metadata is provided, use Groq for comprehensive analysis
    if image_metadata and 'image_description' in image_metadata:
        analysis_result = analyze_blockage_with_groq(
            image_metadata['image_description'],
            caption
        )
        
        # Store the full analysis in the metadata for later use
        image_metadata['blockage_analysis'] = analysis_result
        
        return analysis_result['estimated_time'], analysis_result
    
    # Otherwise, use the simple rule-based system
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if groq_api_key:
        try:
            # Simplified analysis with just the caption
            analysis_result = analyze_blockage_with_groq("No image description available", caption)
            return analysis_result['estimated_time']
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            # Fall back to rule-based estimation
            return simulate_ai_response(caption)
    else:
        # If no API key is available, use rule-based estimation
        return simulate_ai_response(caption)

def generate_fallback_explanation(caption, estimated_time):
    """
    Generate a fallback explanation for why a specific blockage time was estimated
    
    Args:
        caption (str): The user-provided caption
        estimated_time (int): The estimated blockage time in minutes
        
    Returns:
        tuple: (severity, analysis, factors)
    """
    caption_lower = caption.lower()
    factors = []
    severity = "Moderate"
    
    # Determine severity based on estimated time
    if estimated_time >= 180:
        severity = "Major"
    elif estimated_time >= 120:
        severity = "Moderate to Major"
    elif estimated_time <= 30:
        severity = "Minor"
        
    # Check for keywords to determine factors
    if any(word in caption_lower for word in ['accident', 'crash', 'collision']):
        factors.append("Vehicle collision")
        if severity != "Major":
            severity = "Major"
    
    if any(word in caption_lower for word in ['construction', 'work', 'repair']):
        factors.append("Road construction or repair work")
    
    if any(word in caption_lower for word in ['flood', 'water', 'rain']):
        factors.append("Water logging or flooding")
    
    if any(word in caption_lower for word in ['tree', 'fallen', 'debris']):
        factors.append("Fallen tree or debris")
    
    if any(word in caption_lower for word in ['traffic', 'congestion', 'jam']):
        factors.append("Traffic congestion")
    
    if any(word in caption_lower for word in ['landslide', 'rockfall']):
        factors.append("Landslide or rockfall")
        severity = "Major"
    
    if any(word in caption_lower for word in ['protest', 'demonstration', 'rally']):
        factors.append("Protest or demonstration")
    
    # If no specific factors were identified, add a generic one
    if not factors:
        factors.append("Road obstruction")
    
    # Generate analysis text
    analysis = f"Based on the report description, this appears to be a {severity.lower()} road blockage."
    
    # Add explanation for the estimated time
    if estimated_time >= 180:
        analysis += f" The estimated clearance time of {estimated_time} minutes suggests a significant obstruction that requires substantial time to resolve."
    elif estimated_time >= 120:
        analysis += f" The estimated clearance time of {estimated_time} minutes indicates a moderate to major blockage that will take some time to clear."
    elif estimated_time >= 60:
        analysis += f" The estimated clearance time of {estimated_time} minutes suggests a moderate blockage that should be cleared within the hour."
    else:
        analysis += f" The estimated clearance time of {estimated_time} minutes indicates a minor blockage that should be cleared relatively quickly."
    
    # Add information about the factors
    if factors:
        analysis += f" The blockage appears to involve {', '.join(factors).lower()}."
    
    # Add recommendations based on severity
    if severity == "Major":
        analysis += " Drivers are advised to seek alternative routes as this blockage may persist for an extended period."
    elif severity == "Moderate to Major":
        analysis += " Consider alternative routes if available."
    else:
        analysis += " The route should clear relatively soon, but caution is advised."
    
    return severity, analysis, factors

def simulate_ai_response(caption):
    """
    Simulate an AI response for estimating blockage time based on keywords
    
    Args:
        caption (str): The user-provided caption
        
    Returns:
        int: Estimated blockage time in minutes
    """
    caption = caption.lower()
    
    # Check for keywords indicating severity
    if any(word in caption for word in ['accident', 'crash', 'collision']):
        return 120  # 2 hours
    
    if any(word in caption for word in ['major', 'severe', 'serious']):
        return 90  # 1.5 hours
    
    if any(word in caption for word in ['construction', 'work', 'repair']):
        return 180  # 3 hours
    
    if any(word in caption for word in ['police', 'emergency']):
        return 60  # 1 hour
    
    if any(word in caption for word in ['traffic', 'congestion', 'jam']):
        return 45  # 45 minutes
    
    if any(word in caption for word in ['minor', 'small']):
        return 30  # 30 minutes
    
    if any(word in caption for word in ['flood', 'water']):
        return 240  # 4 hours
    
    if any(word in caption for word in ['tree', 'fallen']):
        return 120  # 2 hours
    
    # Default estimate if no keywords match
    return 60  # 1 hour default
