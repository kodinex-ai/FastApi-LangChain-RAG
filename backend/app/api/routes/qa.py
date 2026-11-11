import os
import yaml

from app.core.config import logger, settings


from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document

from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory

from langchain.prompts.prompt import PromptTemplate
from app.schemas.chat_schema import ChatBody
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.api.deps import CurrentUser, SessionDep, get_current_user
from app.crud import chat_crud
from app.models.user_model import User

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
router = APIRouter()

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config/chat.yml")
with open(config_path, "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

chat_config = config.get("CHAT_CONFIG", None)

logger.info(f"Chat config: {chat_config}")


def get_context_retriever_chain(vector_store):
    logger.info("Creating context retriever chain")
    llm = ChatOpenAI(model_name="gpt-4-turbo")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_vector_store():
    embeddings = OpenAIEmbeddings()
    return PGVector(
        collection_name="docs",
        connection_string=settings.SYNC_DATABASE_URI,
        embedding_function=embeddings,
    )


def to_messages(rows) -> list:
    return [
        HumanMessage(content=row.content)
        if row.role == chat_crud.ROLE_HUMAN
        else AIMessage(content=row.content)
        for row in rows
    ]


@router.post("/chat")
def chat_action(
    request: ChatBody,
    session: SessionDep,
    current_user: User = Depends(get_current_user),
):
    # Sync endpoint (runs in the threadpool) so a request waiting on the
    # conversation lock does not block the event loop.
    user_id = current_user.id

    if request.conversation_id is None:
        conversation = chat_crud.create_conversation(session=session, user_id=user_id)
        chat_history = []
    else:
        try:
            session.exec(
                text(
                    f"SET LOCAL lock_timeout = '{int(settings.CHAT_LOCK_TIMEOUT_SECONDS)}s'"
                )
            )
            # Ownership-checked lookup + row lock: serializes requests on the
            # same conversation across workers until commit/rollback.
            conversation = chat_crud.get_user_conversation(
                session=session,
                conversation_id=request.conversation_id,
                user_id=user_id,
                lock=True,
            )
        except OperationalError:
            session.rollback()
            raise HTTPException(status_code=409, detail="Conversation is busy")
        if conversation is None:
            # Same response whether it does not exist or belongs to someone else.
            raise HTTPException(status_code=403, detail="Conversation not accessible")
        chat_history = to_messages(
            chat_crud.get_recent_messages(
                session=session,
                conversation_id=conversation.id,
                limit=settings.CHAT_HISTORY_MAX_MESSAGES,
            )
        )

    try:
        retriever_chain = get_context_retriever_chain(get_vector_store())
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        response = conversation_rag_chain.invoke(
            {"chat_history": chat_history, "input": request.message}
        )
        answer = response["answer"]
    except Exception:
        # Nothing is persisted (incl. a newly created conversation) on failure.
        session.rollback()
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    chat_crud.add_message_pair(
        session=session,
        conversation_id=conversation.id,
        question=request.message,
        answer=answer,
    )
    conversation_id = conversation.id
    session.commit()

    return {"data": answer, "conversation_id": str(conversation_id)}

    # # Load prompts from configuration
    # _template_condense = chat_config["PROMPTS"]["CONDENSE_QUESTION"]
    # _template_answer = chat_config["PROMPTS"]["ANSWER_QUESTION"]
    # _template_default_document = chat_config["PROMPTS"]["DEFAULT_DOCUMENT"]

    # # Your existing logic here, replace hardcoded prompt templates with loaded ones
    # # Example of using loaded prompts:
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template_condense)

    # ANSWER_PROMPT = ChatPromptTemplate.from_template(_template_answer)
    # DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(_template_default_document)
    # logger.info(f"CONDENSE_QUESTION_PROMPT: {CONDENSE_QUESTION_PROMPT}")
    # logger.info(f"ANSWER_PROMPT: {ANSWER_PROMPT}")
    # logger.info(f"DEFAULT_DOCUMENT_PROMPT: {DEFAULT_DOCUMENT_PROMPT}")

    # def _combine_documents(
    #     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    # ):
    #     doc_strings = [format_document(doc, document_prompt) for doc in docs]

    #     return document_separator.join(doc_strings)

    # memory = ConversationBufferMemory(
    #     return_messages=True, output_key="answer", input_key="question"
    # )

    # # First we add a step to load memory
    # # This adds a "memory" key to the input object
    # loaded_memory = RunnablePassthrough.assign(
    #     chat_history=RunnableLambda(memory.load_memory_variables)
    #     | itemgetter("history"),
    # )
    # # Now we calculate the standalone question
    # standalone_question = {
    #     "standalone_question": {
    #         "question": lambda x: x["question"],
    #         "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    #     }
    #     | CONDENSE_QUESTION_PROMPT
    #     | ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
    #     | StrOutputParser(),
    # }
    # # Now we retrieve the documents
    # retrieved_documents = {
    #     "docs": itemgetter("standalone_question") | retriever,
    #     "question": lambda x: x["standalone_question"],
    # }
    # # Now we construct the inputs for the final prompt
    # final_inputs = {
    #     "context": lambda x: _combine_documents(x["docs"]),
    #     "question": itemgetter("question"),
    # }

    # test = final_inputs["context"]

    # logger.info(f"Final inputs: {test}")
    # # And finally, we do the part that returns the answers
    # answer = {
    #     "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    #     "docs": itemgetter("docs"),
    # }

    # final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    # inputs = {"question": request.message}
    # logger.info(f"Inputs: {inputs}")
    # result = final_chain.invoke(inputs)

    # test2 = result["answer"]

    # logger.info(f"Result: {test2}")

    # test3 = result["answer"].content

    # logger.info(f"Result: {test3}")

    # return {"data": result["answer"].content}
