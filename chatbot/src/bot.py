from chainlit.server import app
from chainlit.context import init_http_context

from fastapi import Request

from utils.connection import AnswerGenerator
from utils.config import DEBUG, LOCAL


conn = AnswerGenerator()


@app.post("/sms")
async def chat(request: Request) -> dict[str, str]:
    """Respond to incoming text messages with a text message."""
    # receive question in SMS
    form = await request.form()
    to = form.get("From")
    question = form.get("Body")
    # set http context
    init_http_context(user=to)
    # get answers and send SMS back
    response = await conn.send_sms(input_message=question, to=to)
    if DEBUG:
        print("From {} to {}".format(to, form.get("To")))
        print(f"Question: {question}\nAnswer: {response.body}")
        print(f"Message status: {response.status}, SID: {response.sid}\n")
    return {"question": question, "answer": response.body}


@app.get("/answer/")
async def chat_query(
    question: str,
    to_phone_number: str = "test"
) -> dict[str, str]:
    """Respond queries with a LLM generated answer."""
    # set http context
    init_http_context()
    # get answers
    answer = await conn.create_answer(
        question=question, session=to_phone_number
    )
    return {"question": question, "answer": answer}


if LOCAL:
    import chainlit as cl

    @cl.on_chat_start
    async def main() -> None:
        await cl.Message(content="Ask me anything!", author="Avvi").send()

    @cl.on_message
    async def on_message(msg: cl.Message) -> None:
        question = msg.content
        answer = await conn.create_answer(question=question, session="test")
        await cl.Message(content=answer, author="Avvi").send()

    @cl.on_stop
    async def on_exit() -> None:
        conn.vector_store_client.close()
        conn.checkpoint_client.close()
        conn.chat_history_client.close()
        print("Exit chainlit!")
