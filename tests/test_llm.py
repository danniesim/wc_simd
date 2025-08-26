from wc_simd.llm import ChatSession


def test_chat_session():
    session = ChatSession()
    user_input = "Hello, how are you?"
    response = [x for x in session.response_generator(user_input, stdout=True)]
    assert response is not None


def test_chat_instruction():
    session = ChatSession(
        system_instruction="You are a helpful assistant called THE GOAT.")
    user_input = "Who are you?"
    response = session.send(user_input, stdout=True)
    assert response is not None


def test_chat_session_history():
    session = ChatSession()
    user_input = "Hello, my name is Dan."
    session.send(user_input, stdout=True)
    response = session.send("What is my name?", stdout=True)
    # check if "Dan" is in response
    assert any("Dan" in msg for msg in response)
    session.clear_history()
    response2 = session.send("What is my name?", stdout=True)
    # check if "Dan" is in response
    assert not any("Dan" in msg for msg in response2)
