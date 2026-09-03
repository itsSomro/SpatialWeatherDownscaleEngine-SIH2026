from backend.ai_agent import get_assistant_reply


def main():
    print("running")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("terminated")
            break

        response = get_assistant_reply(user_input)
        print("\nAssistant:")
        print(response)


if __name__ == "__main__":
    main()
