import logging
from openai import OpenAI
from fastapi import HTTPException
import os

# API keys
XAI_API_KEY = os.getenv("XAI_API_KEY")
VISION_MODEL_NAME = "grok-vision-beta"
CHAT_MODEL_NAME = "grok-beta"

client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

logger = logging.getLogger(__name__)

def process_image_with_grok(base64_image: str) -> dict:
    try:
        logger.debug("Sending request to Grok Vision model.")

        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze the provided image and extract detailed attributes of the person in the image. "
                                "Categorize the output into distinct groups of physical features. Structure the response in JSON format with the following groups:\n\n"
                                "1. \"face_shape\": Identify the face shape (e.g., oval, round, square, heart-shaped).\n"
                                "2. \"hair_details\": Include hair color, texture (e.g., straight, wavy, curly), and length (short, medium, long).\n"
                                "3. \"eye_details\": Include eye color, shape (e.g., almond, round, hooded).\n"
                                "4. \"skin_tone\": Classify the skin tone (e.g., fair, medium, tan, dark).\n"
                                "5. \"other_features\": Note distinctive features like freckles, moles, scars, or makeup (e.g., lipstick, eyeshadow).\n\n"
                                "Return the results in the following JSON format:\n\n"
                                "{\n"
                                "  \"face_shape\": \"<value>\",\n"
                                "  \"hair_details\": {\n"
                                "    \"color\": \"<value>\",\n"
                                "    \"texture\": \"<value>\",\n"
                                "    \"length\": \"<value>\"\n"
                                "  },\n"
                                "  \"eye_details\": {\n"
                                "    \"color\": \"<value>\",\n"
                                "    \"shape\": \"<value>\"\n"
                                "  },\n"
                                "  \"skin_tone\": \"<value>\",\n"
                                "  \"other_features\": [\"<value>\", \"<value>\"]\n"
                                "}\n\n"
                                "Additionally, allow users to describe desired features (face shape, hair color, eye color, etc.) and match these descriptions with identified attributes to suggest the closest matches. "
                                "Provide the matching logic and confidence levels for each attribute."
                            )
                        }
                    ]
                }
            ],
        )

        return response.choices[0].message

    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def process_document_with_text_model(aggregated_results: list) -> dict:
    document_context = " ".join([str(result) for result in aggregated_results])
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
             messages=[
                {
                    "role": "system",
                    "content": """You are a helpful, friendly, and clear assistant with expertise in analyzing and solving form-related issues. 
                                Provide personalized guidance based on the extracted form data:

                    1. **Completed Fields**:
                       - Acknowledge the user's effort.
                       - Verify if the values provided are logical and valid.
                       ``` 
                       {completed_fields}
                       ```

                    2. **Empty Fields**:
                       - Explain the importance of each missing field.
                       - Provide instructions and examples to help complete it.
                       ``` 
                       {empty_fields}
                       ```

                    3. **Required Field Statuses**:
                       - Identify required fields that are incomplete.
                       - Prioritize missing required fields and guide the user to address them.
                       ``` 
                       {required_field_statuses}
                       ```

                    ### Output Structure:
                    - Start with an acknowledgment of the user's effort.
                    - Highlight completed fields and confirm their validity.
                    - Provide step-by-step guidance for each missing field, prioritizing required ones.
                    - Use a supportive tone with examples where relevant.
                    - End with encouragement to finish the form.

                    ### Example Output:
                    "Great work so far! Here's what I noticed:

                    ✅ **Completed Fields**:
                    - **Full Name**: John Doe
                    - **Date of Birth**: 1990-01-01
                       These look good!

                    ⚠️ **Fields That Need Attention**:
                    - **Email Address**: Missing. Please enter your email, e.g., john.doe@example.com.

                    🚨 **Required Fields Missing**:
                    - **Address**: Enter your full address, e.g., '123 Main St, Springfield, IL 12345'.

                    Keep going, you're almost there! 📝"

                    Generate helpful, supportive text based on the provided data."""
                },
                {"role": "user", "content": document_context},
            ],
        )
        return response.choices[0].message
    except Exception as e:
        logger.error("Error processing document: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


def generate_response(request: dict) -> str:
    """
    Generates a response based on the user's request and interaction.
    """
    base_messages = [
        {
            "role": "system",
            "content": """As a creative assistant in the partner matching process, 
                       the language model can check many things during a conversation with 
                       a client to best understand their needs and preferences. Here are some examples:

                       **Basic information:**
                       * Age, gender, location: This is basic demographic information that will help 
                         narrow down the pool of potential partners.
                       * Sexual orientation: It is important to know the client's preferences in this regard.
                       * Relationship status: Is the client single, widowed, divorced?
                       * Education, profession: This information can provide insight into the client's lifestyle and aspirations.

                       **Partner preferences:**
                       * Age: Does the client have preferences regarding the age of the potential partner?
                       * Physical appearance: Are there any specific features that the client finds attractive?
                       * Personality: Is the client looking for someone extroverted, introverted, spontaneous, or rather calm?
                       * Interests: Is the client looking for someone with similar interests, 
                         or perhaps someone who will introduce them to new hobbies?
                       * Values: Are there any values that are particularly important to the client in a relationship 
                         (e.g., honesty, loyalty, family)?

                       **Lifestyle and expectations:**
                       * Physical activity: Is the client physically active and looking for a partner with a similar lifestyle?
                       * Eating habits: Is the client a vegetarian, vegan, or do they have any food allergies?
                       * Attitude towards alcohol, cigarettes: Does the client drink alcohol, smoke cigarettes, 
                         and do they accept these habits in a partner?
                       * Plans for the future: Is the client planning to start a family, travel, or focus on their career?

                       **Additional aspects:**
                       * Level of commitment: Is the client looking for a serious relationship or a casual acquaintance?
                       * Past experiences: Has the client had any difficult experiences in previous relationships 
                         that might affect their current expectations?
                       * Openness to new experiences: Is the client open to meeting people from different backgrounds and cultures?

                       The language model can also analyze the client's way of speaking to capture nuances and emotions 
                       that can be helpful in finding the ideal partner. For example, tone of voice, choice of words, 
                       and speaking rate can provide information about the client's temperament and personality.

                       It is important that the language model asks questions in an empathetic and non-judgmental manner, 
                       so that the client feels comfortable and free to share their thoughts and feelings.
                       """
        }
    ]
    base_messages.append({"role": "user", "content": request['question']})

    try:
        # Initial API call to the chat model
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=base_messages,
        )

        # Extract the first response from the chat model
        initial_message = response.choices[0].message

        # Check if the user's query involves document-related topics
        requires_document = any(keyword in request['question'].lower() for keyword in ["form", "document", "application", "download"])

        # If documents are relevant, prepare document links HTML
        document_links_html = ""
        if requires_document:
            for doc_key, doc_info in DOCUMENTS_DB.items():
                document_links_html += f'<p><a href="{doc_info["url"]}" download="{doc_info["document_name"]}">{doc_info["document_name"]}</a></p>'

        # Create an interactive response depending on the context
        if requires_document:
            grok_response = (
                f"Sure thing! It sounds like you need some official documents. Here are the ones I think will help you: "
                f"{document_links_html} Let me know if you'd like help filling them out or understanding what to do next!"
            )
        else:
            grok_response = (
                f"Great question! {initial_message.content} "
                f"If at any point you think a DMV document might help, just let me know!"
            )

        # Prepare follow-up messages for continued conversation
        follow_up_messages = base_messages + [initial_message, {"role": "assistant", "content": grok_response}]

        # Make the second API call to refine or extend the response
        final_response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=follow_up_messages,
        )

        # Extract and process the final response content
        final_answer = final_response.choices[0].message.content
        return final_answer

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")