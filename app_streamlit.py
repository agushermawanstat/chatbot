# Streamlit UI
st.title("CIT-Knowledge Management Chatbot")

# Get user input with wider input box and the same prompt as before
user_input = st.text_area("To provide a more accurate answer, please provide details of your question or issue (type 'exit' to exit):", key='user_input')
st.markdown(
    """
    <style>
        textarea {
            width: 100%; /* Set the width of the textarea to 100% of the container */
            border: none; /* Remove the textarea border */
            padding: 10px; /* Add padding for a better appearance */
            font-size: 16px; /* Adjust font size as needed */
        }
        .response-box {
            border-radius: 15px;
            padding: 10px;
            margin: 10px 0;
        }
        .submit-button {
            background-color: #4CAF50; /* Green background color */
            color: white; /* White text color */
            padding: 10px 20px; /* Add padding to the button */
            border: none; /* Remove button border */
            border-radius: 5px; /* Add border radius to the button */
            font-size: 16px; /* Adjust font size as needed */
        }
        .custom-warning {
            background-color: #4CAF50; /* Green background color */
            color: white; /* White text color */
            padding: 10px; /* Add some padding */
            margin: 10px 0; /* Add some margin */
            border-radius: 15px; /* Add border radius to the warning box */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a Submit button with a green background
if st.button("Submit", key='submit_button', class="submit-button"):
    if user_input.lower() != 'exit':
        response_options = generate_response_tfidf_with_probability_and_detail(user_input, df)
        if response_options:
            for i, (response, probability) in enumerate(response_options, start=1):
                # Define response box color based on probability
                if probability >= 0.8:
                    color = "#ADFF2F"  # Green
                elif probability >= 0.5:
                    color = "#FFD700"  # Yellow
                else:
                    color = "#F08080"  # Red

                # Display response with colored box
                st.markdown(
                    f"""
                    <div class="response-box" style="background-color: {color};">
                        Option {i}: (Prob.: {probability:.0%}) {response.capitalize()}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            # Display the custom warning message
            st.markdown(
                """
                <div class="custom-warning">
                    Kindly provide a comprehensive and detailed description of the issue you are facing, and I will offer the solution as accurately as possible!
                </div>
                """,
                unsafe_allow_html=True
            )
