from utils import *
import cohere

SUMMARY_PROMPT = (
    """
    Concisely summarize the following chat history. The length of the summary should be a maximum of 100 words.

    Do not include any other text. Respone ONLY with the content of your summary.

    Here is the chat history:

    \n\n
    {messages}
    \n\n

    """
)
def summarize_text(messages):
    """call the model to summarize text if the text is long enough"""
    co = cohere.ClientV2()
    msgs = "\n\n".join(message["content"] for message in messages)
    prompt = SUMMARY_PROMPT.format(messages=msgs)
    summary = co.chat(
        model = "command-a-03-2025",
        messages = [{"role": "user", "content":prompt}],
    )
    # [{"role":"assistant", "content":summary.text}]
    return summary.message.content[0].text
