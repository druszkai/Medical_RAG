import os

from config import *

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def main():
    print("Rendszer inicializálás.\n")
    print("Rendszer inicializálása megtörtént.\n")

    while True:
        user_input = input("\nPÁCIENS KÉRDÉSE:\n")
        if user_input.lower() in {"exit", "quit", "q", "kilép", "kilépés"}:
            break

    print("Rendszer leáll.\n")


if __name__ == "__main__":
    main()