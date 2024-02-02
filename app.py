import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Initialize tokenizer and model for Microsoft's Phi-2
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", device_map="auto", trust_remote_code=True)

# Function to generate text using Phi-2 model
def generate_text(prompt):
    
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4
        )
        
    return tokenizer.decode(output_ids[0][token_ids.size(1):])

# Streamlit app interface
st.title("Lesson Plan Generator")

# User input for the lesson generation
lesson_plan_input = st.text_input("Lesson Plan Request:", "Please generate a lesson plan on Mathematics for a 7th Grade level in the us on Linear Algebra")

#Template for the lesson plan, used in the system prompt
FORMATTED_RESPONSE="""
        Title: 
        Grade Level: 
        Subject:  
        Objectives: 
        Activities: 
        Assessment: Application of knowledge and skills through a task
        Procedure: Formatted as a list of steps and substeps
        Resources:
        Notes:
        """

# Generate Lesson Plan Button
if st.button("Generate Lesson Plan"):
    
    # Construct the prompt for lesson plan
    lesson_plan_prompt = f"""
    You are assisting the user in lesson plan generation.
    USER: {lesson_plan_input}
    Formulate the lesson in the following format {FORMATTED_RESPONSE}
    """

    # Generate lesson plan
    lesson_plan = generate_text(lesson_plan_prompt)
    st.write(f"{lesson_plan.strip()}")
