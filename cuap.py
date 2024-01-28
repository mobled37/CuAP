import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def Attribute_Description(client, attribute):
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Make the attribute_description"},
            {"role": "user", "content": f"What does a {attribute} looks like?"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def main():
    source = "Church"
    attribute = "fire"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    while True:
        attribute_description = Attribute_Description(client, attribute)
        if attribute_description.find("."):
            attribute_description = attribute_description[
                attribute_description.find(attribute)
                + len(attribute)
                + 1 : attribute_description.find(".")
                + 1
            ]
            break

    CuAP_sentence = f"{source} with {attribute} that {attribute_description}"

    print("cuap :", CuAP_sentence)


if __name__ == "__main__":
    main()
