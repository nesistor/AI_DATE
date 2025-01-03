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

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")  # You can set your MongoDB URI in environment variables
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["user_database"]  # Select the database
users_collection = db["users"]

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
    Focuses on understanding user preferences, characteristics, and needs.
    """
    base_messages = [
        {
            "role": "system",
            "content": """As a funny and creative assistant in the partner matching process, 
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

        # Focus on analyzing and understanding user characteristics and needs
        grok_response = (
            f"Thank you for sharing! {initial_message.content} "
            f"Let's dive deeper into what you're looking for and how I can assist further."
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

def save_characteristics_to_db(user_id: str, characteristics: dict) -> None:
    """
    Saves a user's characteristics to MongoDB.
    
    :param user_id: User's ID
    :param characteristics: Dictionary containing the user's characteristics
    """
    try:
        # Check if the user already exists
        user = users_collection.find_one({"user_id": user_id})
        
        if user:
            # If the user exists, update their data
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"characteristics": characteristics}}
            )
        else:
            # If the user doesn't exist, insert a new document
            users_collection.insert_one({"user_id": user_id, "characteristics": characteristics})
        
        logger.debug(f"Successfully saved characteristics for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving characteristics to DB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving characteristics to DB: {str(e)}")

def get_characteristics_from_db(user_id: str) -> dict:
    """
    Retrieves a user's characteristics from MongoDB.
    
    :param user_id: User's ID
    :return: Dictionary containing the user's characteristics
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        if user:
            return user.get("characteristics", {})
        else:
            logger.warning(f"User {user_id} not found in database.")
            return {}
    except Exception as e:
        logger.error(f"Error retrieving characteristics from DB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving characteristics from DB: {str(e)}")

        
def match_users(user1_id: str, user2_id: str, user1_intentions: dict, user2_intentions: dict) -> dict:
    """
    Compares the characteristics and intentions of two users based on their data from MongoDB.
    
    :param user1_id: ID of the first user (primary user)
    :param user2_id: ID of the second user
    :param user1_intentions: Intentions or goals of the first user
    :param user2_intentions: Intentions or goals of the second user
    
    :return: Dictionary with the comparison results
    """
    try:
        # Get the characteristics of the users from the database
        user1_characteristics = get_characteristics_from_db(user1_id)
        user2_characteristics = get_characteristics_from_db(user2_id)
        
        if not user1_characteristics or not user2_characteristics:
            raise HTTPException(status_code=404, detail="One or both users do not have stored characteristics.")
        
        # Combine the characteristics and intentions of both users into a single comparison text
        comparison_text = f"""
        User 1 (Primary) Characteristics: {json.dumps(user1_characteristics)}
        User 1 (Primary) Intentions: {json.dumps(user1_intentions)}
        
        User 2 Characteristics: {json.dumps(user2_characteristics)}
        User 2 Intentions: {json.dumps(user2_intentions)}
        
        Please analyze these two users based on their characteristics and intentions. 
        Compare them in different areas like personality, lifestyle, goals, and physical traits. 
        Provide a compatibility score (0-100) for each category and an overall compatibility score.
        """
        
        # Make the request to the Grok model with an updated prompt including intentions
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert matchmaker who analyzes users' characteristics, intentions, and finds compatibility."},
                {"role": "user", "content": comparison_text}
            ]
        )

        # Extract the comparison result
        matching_result = response.choices[0].message.content
        logger.debug(f"Matching result: {matching_result}")
        
        # Assuming the response from Grok is in JSON format, parse the result
        match_details = json.loads(matching_result)
        return match_details
    
    except Exception as e:
        logger.error(f"Error matching users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching users: {str(e)}")
