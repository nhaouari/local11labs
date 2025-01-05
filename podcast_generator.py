import google.generativeai as genai
import json
import os

def generate_podcast_script(api_key, news_content, host_names=["Noureddine", "Noura"]):
  """Generates a podcast script using the Gemini API with a dynamic list of host names."""
  if len(host_names) < 2:
    raise ValueError("At least two host names are required")

  genai.configure(api_key=api_key)

  generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
  }

  gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
  )

  chat_session = gemini_model.start_chat(history=[])

  # Create host introductions dynamically
  host_intro = ", ".join(host_names[:-1]) + f" and {host_names[-1]}"

  prompt = f"""
You are an AI assistant tasked with creating a podcast script about recent developments in artificial intelligence. The script should be engaging, informative, and structured as a conversation between the following hosts: {host_intro}.

Here is the AI news content to base the podcast on:

<ai_news_content>
{news_content}
</ai_news_content>

Before generating the final script, please analyze the content and structure your thoughts.

In your content breakdown:
1. Extract and list 5-7 key quotes from the AI news content, citing their relevance.
2. Identify the main topics and any recurring elements in the news content.
3. Categorize these topics into 2-3 broader themes for discussion.
4. Select 2-3 impactful excerpts that highlight key developments.
5. Analyze these excerpts, noting their implications for the AI field.
6. Identify any contrasting or complementary information between the excerpts.
7. Note any common assumptions or expectations related to the topics.
8. Summarize the most important insights from the content.
9. For each main topic, brainstorm 2-3 potential questions or discussion points.
10. Formulate a thought-provoking question that extends beyond the discussed material.
11. Create a brief outline of the podcast structure, including the order of topics and key points to cover.

After your content breakdown, generate a podcast script that follows these guidelines:
1. Structure the script as a conversation between multiple hosts, rotating between them.
2. Begin with a brief introduction to the AI news roundup.
3. Cover the main topics identified in your analysis, alternating between hosts.
4. Include the impactful excerpts and your analysis of them in the conversation.
5. Discuss any contrasts or complementary information between topics.
6. Address common assumptions or expectations related to the topics.
7. Incorporate the brainstormed questions and discussion points to create a dynamic conversation.
8. Conclude with a summary of key insights and the thought-provoking question.
9. Keep the tone conversational and engaging throughout.
10. Make sure the content and dialogue are clean of any special characters and ready to be read by a TTS.

Format the script as a list of JSON objects, where each object represents a dialogue turn with "host" and "dialogue" keys. Ensure that the hosts rotate throughout the script.

Example structure (do not use this content, it's just to illustrate the format):
{{
  "content breakdown": [
  {{
    "step":"",
    "content": ""
  }}
  ...
  ],
  "script":[
  {{
    "host": "{host_names[0]}",
    "dialogue": "Welcome to our AI news roundup. Today, we have some exciting developments to discuss."
  }},
  ...
]
}}

Remember to maintain this structure throughout the entire script, rotating between all hosts, and covering all the key points from your content breakdown.
"""

  response = chat_session.send_message(prompt)
  podcast_data = json.loads(response.text)

  return podcast_data